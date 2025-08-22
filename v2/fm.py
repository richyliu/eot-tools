#!/usr/bin/env python3
"""
FM encoder and decoder
"""

import numpy as np
from scipy import signal


class FMModulator:
    """FM modulator for audio signals"""

    def __init__(self, sample_rate=2000000, audio_sample_rate=44100, deviation=2500):
        self.sample_rate = sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation

    def modulate(self, audio_signal):
        """Apply FM modulation to audio signal"""
        # Upsample audio to RF sample rate
        upsampled = signal.resample(audio_signal, len(audio_signal) * self.sample_rate // self.audio_sample_rate)

        # Pre-emphasis filter (6dB/octave above 3.18kHz)
        b, a = signal.butter(1, 3180 / (self.sample_rate / 2), 'high')
        emphasized = signal.lfilter(b, a, upsampled)

        # FM modulation
        phase = np.cumsum(2 * np.pi * self.deviation * emphasized / self.sample_rate)
        fm_signal = np.exp(1j * phase)

        return fm_signal

    def demodulate(self, fm_signal, shift=0):
        """Demodulate FM signal back to audio"""
        # shift the signal to center frequency
        fm_signal = fm_signal * np.exp(-2j * np.pi * shift * np.arange(len(fm_signal)) / self.sample_rate)

        # downsample to audio sample rate
        downsampled = signal.resample(fm_signal, len(fm_signal) * self.audio_sample_rate // self.sample_rate)

        # Quadrature demodulation
        sig = np.angle(downsampled[1:] * np.conj(downsampled[:-1]))
        sig = np.concatenate(([0], sig))

        # Scale the demodulated signal
        scale = 10
        sig = sig * scale
        sig = np.clip(sig, -1, 1)
        return sig


def test():
    from afsk import AFSKEncoder

    afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=1800)
    test_bits = '0' * 5 + '01' * 20 + '0' + '1110010' + '0' * 2
    audio_signal = afsk.encode(test_bits)

    fm_modulator = FMModulator(sample_rate=2000000, audio_sample_rate=44100, deviation=2500)
    fm_signal = fm_modulator.modulate(audio_signal)
    # shift by 25kHz to simulate RF transmission
    fm_signal = fm_signal * np.exp(2j * np.pi * 25000 * np.arange(len(fm_signal)) / 2000000)

    audio_signal_demod = fm_modulator.demodulate(fm_signal, shift=25000)

    decoded_bits = afsk.decode(audio_signal_demod, frame_sync='0101010101')
    print('decoded:', decoded_bits)
    print('original:', test_bits)


if __name__ == "__main__":
    test()
