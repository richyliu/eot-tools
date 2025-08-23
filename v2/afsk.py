#!/usr/bin/env python3
"""
FM-AFSK encoder and decoder
"""

import numpy as np
from scipy import signal


class AFSKEncoder:
    """AFSK encoder and decoder for converting digital data to audio tones or vice versa"""

    def __init__(self, sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=1800):
        self.sample_rate = sample_rate
        self.internal_sample_rate = sample_rate // baud_rate * baud_rate  # Adjust to be multiple of baud rate
        self.baud_rate = baud_rate
        self.mark_freq = mark_freq    # Frequency for '1' bit
        self.space_freq = space_freq  # Frequency for '0' bit
        self.samples_per_bit = sample_rate // baud_rate

    def encode(self, bitstring):
        """Convert bit string to AFSK audio signal"""
        audio = np.array([])

        last_phase = 0
        t_err = 0
        for bit in bitstring:
            freq = self.mark_freq if bit == '1' else self.space_freq
            t = np.linspace(0, 1/self.baud_rate, self.samples_per_bit, False)
            tone = np.sin(2 * np.pi * freq * t + last_phase)
            audio = np.concatenate([audio, tone])
            last_phase += 2 * np.pi * freq * (1 / self.baud_rate)

        # resample to desired sample rate if changed
        if self.internal_sample_rate != self.sample_rate:
            audio = signal.resample(audio, int(len(audio) * self.sample_rate / self.internal_sample_rate))

        return audio

    def decode(self, audio_signal, frame_sync='0101', debug=False):
        """Decode AFSK audio signal back to bits"""

        # resample if needed
        if self.internal_sample_rate != self.sample_rate:
            audio_signal = signal.resample(audio_signal, int(len(audio_signal) * self.internal_sample_rate / self.sample_rate))

        # convert to complex waveform with shifted frequencies
        audio_center = (self.mark_freq + self.space_freq) / 2
        sig = audio_signal * np.exp(-2j * np.pi * audio_center * np.arange(len(audio_signal)) / self.internal_sample_rate)

        # low pass filter to isolate frequencies
        filter_freq = abs(self.mark_freq - self.space_freq) / 2 + 300
        sig = signal.lfilter(signal.firwin(81, cutoff=filter_freq, fs=self.internal_sample_rate), 1, sig)

        # quadrature demodulation
        scale = 10
        sig = np.angle(sig[1:] * np.conj(sig[:-1])) * scale
        sig = np.concatenate(([0], sig))  # prepend zero for first sample

        # ignore extreme values
        sig[np.abs(sig) > 1] = 0

        # find the bit boundaries with convolution
        num_ones = int(self.samples_per_bit * 0.1)
        num_zeros = self.samples_per_bit - num_ones
        bitbounds_mask = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))
        mask = []
        for bit in frame_sync:
            if (bit == '1') == (self.mark_freq > self.space_freq):
                mask.append(bitbounds_mask)
            else:
                mask.append(-bitbounds_mask)
        all_mask = np.concatenate(mask)
        bitbounds = np.convolve(sig, all_mask[::-1], mode='valid')

        start = np.argmax(bitbounds)

        # get the bit values
        total_len = ((len(sig) - start) // self.samples_per_bit - 1) * self.samples_per_bit
        if total_len <= 0:
            raise ValueError("Not enough data to decode AFSK signal")
        indices = np.linspace(start, start + total_len, total_len // self.samples_per_bit + 1).astype(int)
        if self.mark_freq < self.space_freq:
            sig = -sig  # Invert if space (0) frequency is higher than mark (1) frequency
        vals = ((np.sign(sig[indices]) + 1) / 2).astype(int)
        vals = ''.join(map(str, vals))

        if debug:
            print(start, len(sig), self.samples_per_bit, total_len, (indices[5] - indices[0]) / 5)
            print(vals)
            # TEMP: plot to file for debugging
            import matplotlib.pyplot as plt
            plt.figure(figsize=(15, 6))
            plt.plot(sig, label='Demodulated Signal')
            plt.plot(bitbounds/40, label='Bit Boundaries', color='red')
            # plt.plot(np.linspace(start, start + len(all_mask), len(all_mask)), all_mask, label='Mask', color='green')
            # plt.plot(audio_signal * 0.5, label='Original Audio Signal', alpha=0.5)
            plt.axvline(start, color='green', linestyle='--', label='Start Index')
            for i in indices:
                plt.axvline(i, color='orange', linestyle=':')
            plt.title('Demodulated Signal')
            plt.xlabel('Sample Index')
            plt.ylabel('Phase')
            plt.legend()
            plt.grid()
            plt.savefig('demodulated_signal.png')
            plt.close()

        return vals



def test():
    afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=1800)
    test_bits = '0' * 5 + '01' * 20 + '0' + '1110010' + '0' * 2
    audio_signal = afsk.encode(test_bits)

    decoded_bits = afsk.decode(audio_signal, frame_sync='0101010101')
    print('decoded:', decoded_bits)
    print('original:', test_bits)
    assert decoded_bits in test_bits, "Decoded bits do not match original bits"

    print('Test OK')


if __name__ == "__main__":
    test()
