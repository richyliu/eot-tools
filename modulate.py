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
import sys
from scipy import signal
from scipy.io import wavfile


def eprint(*args, **kwargs):
    """Print to stderr"""
    print(*args, file=sys.stderr, **kwargs)


class AFSKEncoder:
    """AFSK encoder for converting digital data to audio tones"""

    def __init__(self, sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=2200):
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

    def __init__(self, sample_rate=2000000, audio_sample_rate=44100, deviation=75000):
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

    def __init__(self, frequency=144390000, sample_rate=2000000, gain=10):
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

            eprint(f"Transmitting on {self.frequency/1e6:.3f} MHz...")
            eprint(f"Sample rate: {self.sample_rate/1e6:.1f} MS/s")
            eprint(f"TX gain: {self.gain} dB")

            # Execute transmission
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                eprint(f"Error: {result.stderr}")
                return False

            eprint("Transmission completed successfully")
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


class EOTRF:
    def __init__(self, freq=457937500):
        self.message = ''

        # Configuration
        RF_FREQUENCY = freq  # 144.39 MHz (2m amateur band)
        RF_SAMPLE_RATE = 2000000  # 2 MS/s
        AUDIO_SAMPLE_RATE = 44100
        TX_GAIN = 40

        # AFSK configuration (Bell 202 standard)
        BAUD_RATE = 1200
        MARK_FREQ = 1200   # '1' bit frequency
        SPACE_FREQ = 1800  # '0' bit frequency

        eprint("HackRF FM-AFSK Transmitter")
        eprint("=" * 40)
        eprint(f"Frequency: {RF_FREQUENCY/1e6:.3f} MHz")
        eprint(f"AFSK: {BAUD_RATE} baud, Mark={MARK_FREQ}Hz, Space={SPACE_FREQ}Hz")
        eprint()

        # Initialize components
        self.afsk_encoder = AFSKEncoder(
            sample_rate=AUDIO_SAMPLE_RATE,
            baud_rate=BAUD_RATE,
            mark_freq=MARK_FREQ,
            space_freq=SPACE_FREQ
        )

        self.fm_modulator = FMModulator(
            sample_rate=RF_SAMPLE_RATE,
            audio_sample_rate=AUDIO_SAMPLE_RATE
        )

        self.transmitter = HackRFTransmitter(
            frequency=RF_FREQUENCY,
            sample_rate=RF_SAMPLE_RATE,
            gain=TX_GAIN
        )


    def with_message(self, packet, padded_silence=0.5):
        packet += '0' * 5 # it tends to truncate the last few bits
        assert all(c in ['0', '1'] for c in packet), "Packet must be a binary string"
        message = map(int, list(packet))

        message_audio = self.afsk_encoder.bits_to_audio(message)

        silence_samples = int(self.afsk_encoder.sample_rate * padded_silence)
        silence = np.zeros(silence_samples)
        # silence = np.sin(2 * np.pi * 700 * np.linspace(0, padded_silence, silence_samples, False)) * 0.4

        self.full_audio = np.concatenate([
            silence,
            message_audio,
            silence,
            message_audio,
            silence,
            message_audio,
            silence,
            # message_audio,
            # silence,
            # message_audio,
            # silence,
        ])

        # FM modulate
        eprint("Applying FM modulation...")
        self.fm_signal = self.fm_modulator.modulate(self.full_audio)

    def save_audio(self, filename='output.wav'):
        """Save the generated audio to a WAV file"""
        eprint(f"Saving audio to {filename}...")
        wavfile.write(filename, self.afsk_encoder.sample_rate, (self.full_audio * (2**15-1)).astype(np.int16))
        eprint(f"Audio duration: {len(self.full_audio)/self.afsk_encoder.sample_rate:.2f} seconds")

    def fm_transmit(self):
        duration = len(self.fm_signal) / self.transmitter.sample_rate
        success = self.transmitter.transmit_iq(self.fm_signal, duration)

        if success:
            eprint(f"Successfully transmitted message")
        else:
            eprint("Transmission failed")
