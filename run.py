#!/usr/bin/env python3
"""
HackRF FM-AFSK Transmitter
Transmits digital data over FM using AFSK (Audio Frequency Shift Keying)
"""

import numpy as np
import time
import struct
import subprocess
import tempfile
import os
from scipy import signal
from scipy.io import wavfile


def xor(a, b):
    result = []
    for i in range(len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
    return ''.join(result)


# Reverse string
def reverse(data):
    return ''.join(data[::-1])


# Perform modulo-2 division on two strings of binary symbols
def mod2div(dividend, divisor):

    # Number of bits to be XORed at a time.
    pick = len(divisor)

    # Slicing the dividend to appropriate
    # length for particular step
    tmp = dividend[0:pick]

    while pick < len(dividend):

        if tmp[0] == '1':

            # replace the dividend by the result
            # of XOR and pull 1 bit down
            tmp = xor(divisor[1:], tmp[1:]) + dividend[pick]

        else:   # If leftmost bit is '0'
            # If the leftmost bit of the dividend (or the
            # part used in each step) is 0, the step cannot
            # use the regular divisor; we need to use an
            # all-0s divisor.
            tmp = xor(('0'*pick)[1:], tmp[1:]) + dividend[pick]

        # increment pick to move further
        pick += 1

    # For the last n bits, we have to carry it out
    # normally as increased value of pick will cause
    # Index Out of Bounds.
    if tmp[0] == '1':
        tmp = xor(divisor[1:], tmp[1:])
    else:
        tmp = xor(('0'*pick)[1:], tmp[1:])

    remainder = tmp
    return remainder


class EOTPacket:
    def __init__(self):
        pass

    # Calculate BCH checkbits
    def calc_checkbits(self, data, key):
        appended_data = data + '0'*(len(key)-1)  # Appends n-1 zeros at end of data
        remainder = mod2div(appended_data, key)
        return ''.join(remainder)

    def eot_encode(self):
        bit_sync = '01' * (1000//2) + '0'
        packet = bytearray(75)
        packet[0:11] = b'11100010010' # frame_sync
        packet[11:13] = b'11' # chaining bits
        packet[13:15] = b'11' # batt_cond
        packet[15:18] = b'000' # message_type
        packet[18:35] = "{:017b}".format(0xabcd)[-17:].encode('ascii') # unit_addr
        packet[35:42] = b'1110000' # self.pressure
        packet[42:49] = b'1110000' # self.batt_charge
        packet[49:56] = b'1' * 7
        data_block = packet[11:56].decode('ascii')

        generator = '1111001101000001111'  # BCH generator polynomial
        cipher_key = '101011011101110000'  # XOR cipher key
        data_block = reverse(data_block)
        checkbits = self.calc_checkbits(data_block, generator)
        checkbits_cipher = xor(checkbits, cipher_key)
        packet[56:74] = checkbits_cipher.encode('ascii')  # append checkbits

        packet[74:75] = b'1' # dummy bit

        print('raw packet:', packet)

        result = (bit_sync + packet.decode('ascii')) * 2 # repeat twice due to redundancy
        # convert ascii string of 1s and 0s to list of integers
        result = [int(bit) for bit in result]

        return result


class AFSKEncoder:
    """AFSK encoder for converting digital data to audio tones"""
    
    def __init__(self, sample_rate=48000, baud_rate=1200, mark_freq=1200, space_freq=2200):
        self.sample_rate = sample_rate
        self.baud_rate = baud_rate
        self.mark_freq = mark_freq    # Frequency for '1' bit
        self.space_freq = space_freq  # Frequency for '0' bit
        self.samples_per_bit = int(sample_rate / baud_rate)
        
    def encode_byte(self, byte_val):
        """Encode a single byte with start/stop bits"""
        # Start bit (0), 8 data bits (LSB first), stop bit (1)
        bits = [0]  # Start bit
        for i in range(8):
            bits.append((byte_val >> i) & 1)
        bits.append(1)  # Stop bit
        return bits
    
    def bits_to_audio(self, bits):
        """Convert bit sequence to AFSK audio signal"""
        audio = np.array([])
        
        last_phase = 0
        for bit in bits:
            freq = self.mark_freq if bit == 1 else self.space_freq
            t = np.linspace(0, 1/self.baud_rate, self.samples_per_bit, False)
            tone = np.sin(2 * np.pi * freq * t + last_phase)
            audio = np.concatenate([audio, tone])
            last_phase += 2 * np.pi * freq * (1 / self.baud_rate)
        
        return audio
    
    def encode_data(self, data):
        """Encode data string to AFSK audio"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        all_bits = []
        for byte in data:
            all_bits.extend(self.encode_byte(byte))
        
        return self.bits_to_audio(all_bits)

class FMModulator:
    """FM modulator for audio signals"""
    
    def __init__(self, sample_rate=2000000, audio_sample_rate=48000, deviation=75000):
        self.sample_rate = sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation
        self.upsample_ratio = sample_rate // audio_sample_rate
        
    def modulate(self, audio_signal):
        """Apply FM modulation to audio signal"""
        # Upsample audio to RF sample rate
        upsampled = signal.resample(audio_signal, len(audio_signal) * self.upsample_ratio)
        
        # Pre-emphasis filter (6dB/octave above 3.18kHz)
        b, a = signal.butter(1, 3180 / (self.sample_rate / 2), 'high')
        emphasized = signal.lfilter(b, a, upsampled)
        
        # FM modulation
        phase = np.cumsum(2 * np.pi * self.deviation * emphasized / self.sample_rate)
        fm_signal = np.exp(1j * phase)
        
        return fm_signal

class HackRFTransmitter:
    """HackRF transmitter interface"""
    
    def __init__(self, frequency=144390000, sample_rate=2000000, gain=14):
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        
    def transmit_iq(self, iq_data, duration=None):
        """Transmit IQ data using hackrf_transfer"""
        # Convert complex IQ to interleaved int8 format
        iq_int8 = self._complex_to_int8(iq_data)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.iq') as f:
            f.write(iq_int8.tobytes())
            temp_file = f.name
        
        try:
            # Build hackrf_transfer command
            cmd = [
                'hackrf_transfer',
                '-t', temp_file,
                '-f', str(self.frequency),
                '-s', str(self.sample_rate),
                '-x', str(self.gain),
                '-a', '1'  # Enable amp
            ]
            
            if duration:
                # Calculate number of samples for duration
                samples = int(duration * self.sample_rate)
                cmd.extend(['-n', str(samples)])
            
            print(f"Transmitting on {self.frequency/1e6:.3f} MHz...")
            print(f"Sample rate: {self.sample_rate/1e6:.1f} MS/s")
            print(f"TX gain: {self.gain} dB")
            
            # Execute transmission
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return False
            
            print("Transmission completed successfully")
            return True
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def _complex_to_int8(self, iq_data):
        """Convert complex IQ data to int8 format for HackRF"""
        # Normalize to [-1, 1]
        iq_normalized = iq_data / np.max(np.abs(iq_data))
        
        # Convert to int8 range [-127, 127]
        real_int8 = (iq_normalized.real * 127).astype(np.int8)
        imag_int8 = (iq_normalized.imag * 127).astype(np.int8)
        
        # Interleave I and Q
        iq_int8 = np.empty(len(iq_data) * 2, dtype=np.int8)
        iq_int8[0::2] = real_int8
        iq_int8[1::2] = imag_int8
        
        return iq_int8

def main():
    """Main transmission function"""
    
    # Configuration
    RF_FREQUENCY = 457937500  # 144.39 MHz (2m amateur band)
    RF_SAMPLE_RATE = 2000000  # 2 MS/s
    AUDIO_SAMPLE_RATE = 44100
    TX_GAIN = 40
    
    # AFSK configuration (Bell 202 standard)
    BAUD_RATE = 1200
    MARK_FREQ = 1200   # '1' bit frequency
    SPACE_FREQ = 1800  # '0' bit frequency
    
    print("HackRF FM-AFSK Transmitter")
    print("=" * 40)
    print(f"Frequency: {RF_FREQUENCY/1e6:.3f} MHz")
    print(f"AFSK: {BAUD_RATE} baud, Mark={MARK_FREQ}Hz, Space={SPACE_FREQ}Hz")
    print()
    
    # Initialize components
    afsk_encoder = AFSKEncoder(
        sample_rate=AUDIO_SAMPLE_RATE,
        baud_rate=BAUD_RATE,
        mark_freq=MARK_FREQ,
        space_freq=SPACE_FREQ
    )
    
    fm_modulator = FMModulator(
        sample_rate=RF_SAMPLE_RATE,
        audio_sample_rate=AUDIO_SAMPLE_RATE
    )
    
    transmitter = HackRFTransmitter(
        frequency=RF_FREQUENCY,
        sample_rate=RF_SAMPLE_RATE,
        gain=TX_GAIN
    )
    
    # Encode message
    message_bits = EOTPacket().eot_encode()  # Example EOT packet
    # message_bits = [1] * 1000
    message_audio = afsk_encoder.bits_to_audio(message_bits)
    
    # Add silence padding
    silence_duration = 0.2 # 200ms
    silence_samples = int(AUDIO_SAMPLE_RATE * silence_duration)
    silence = np.zeros(silence_samples)
    
    # Combine audio segments
    full_audio = np.concatenate([
        silence,
        message_audio,
        message_audio,
        message_audio,
        silence
    ])

    # Save audio to pcm (int16 format)
    audio_file = 'output.wav'
    print(f"Saving audio to {audio_file}...")
    wavfile.write(audio_file, AUDIO_SAMPLE_RATE, (full_audio * (2**15-1)).astype(np.int16))
    
    # # Apply windowing to reduce spectral splatter
    # window_size = int(AUDIO_SAMPLE_RATE * 0.01)  # 10ms window
    # full_audio[:window_size] *= np.hanning(window_size)[:window_size//2]
    # full_audio[-window_size:] *= np.hanning(window_size)[window_size//2:]
    
    print(f"Audio duration: {len(full_audio)/AUDIO_SAMPLE_RATE:.2f} seconds")
    
    # FM modulate
    print("Applying FM modulation...")
    fm_signal = fm_modulator.modulate(full_audio)
    
    # Transmit
    duration = len(fm_signal) / RF_SAMPLE_RATE
    success = transmitter.transmit_iq(fm_signal, duration)
    
    if success:
        print(f"Successfully transmitted message")
    else:
        print("Transmission failed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTransmission interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        # print backtrace for debugging
        import traceback
        traceback.print_exc()
        print("Make sure HackRF is connected and hackrf_transfer is in PATH")
