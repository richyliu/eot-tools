#!/usr/bin/env python3
"""
FM encoder and decoder
"""

import numpy as np
from scipy import signal

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        """Dummy profile decorator if line_profiler is not installed"""
        return func

class FMModulator:
    """FM modulator for audio signals"""

    def __init__(self, sample_rate=6000000, audio_sample_rate=44100, deviation=2500):
        self.sample_rate = sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.deviation = deviation

    def modulate(self, audio_signal, shift=0):
        """Apply FM modulation to audio signal"""
        # Upsample audio to RF sample rate
        upsampled = signal.resample(audio_signal, len(audio_signal) * self.sample_rate // self.audio_sample_rate)

        # Pre-emphasis filter (6dB/octave above 3.18kHz)
        b, a = signal.butter(1, 3180 / (self.sample_rate / 2), 'high')
        emphasized = signal.lfilter(b, a, upsampled)

        # FM modulation
        phase = np.cumsum(2 * np.pi * self.deviation * emphasized / self.sample_rate)
        fm_signal = np.exp(1j * phase)

        # shift the signal to center frequency
        fm_signal = fm_signal * np.exp(-2j * np.pi * shift * np.arange(len(fm_signal)) / self.sample_rate)

        return fm_signal

    def demodulate(self, fm_signal, shift=0):
        """Demodulate FM signal back to audio"""
        # shift the signal to center frequency
        fm_signal = fm_signal * np.exp(-2j * np.pi * shift * np.arange(len(fm_signal)) / self.sample_rate)

        # # low pass filter to remove high frequency noise
        # b, a = signal.butter(1, self.audio_sample_rate / self.sample_rate, 'low')
        # fm_signal = signal.lfilter(b, a, fm_signal)

        # downsample to audio sample rate
        intermediate_rate = self.sample_rate / np.ceil(self.sample_rate / self.audio_sample_rate)
        downsampled1 = signal.decimate(fm_signal, int(self.sample_rate / intermediate_rate))
        downsampled = signal.resample(downsampled1, int(len(downsampled1) * self.audio_sample_rate / intermediate_rate))

        # Quadrature demodulation
        sig = np.angle(downsampled[1:] * np.conj(downsampled[:-1]))
        sig = np.concatenate(([0], sig))

        # Scale the demodulated signal
        scale = 10
        sig = sig * scale
        sig = np.clip(sig, -1, 1)
        return sig


class SignalDetector:
    def __init__(self, target_freqs, N, fs=6000000, deviation=2500):
        self.target_freqs = target_freqs
        self.fs = fs
        self.N = N
        self.deviation = deviation

        freqs = np.fft.fftfreq(N, 1/self.fs)
        mask_signals = []
        self.mask_noise = np.ones(N, dtype=bool)
        for f in self.target_freqs:
            mask = (freqs >= f - self.deviation/2) & (freqs <= f + self.deviation/2)
            mask_signals.append(mask)
            self.mask_noise &= ~mask  # keep noise mask as the complement of all signal masks

        self.mask_signals_matrix = np.array(mask_signals, dtype=bool).astype(np.float32)
        # Normalize the mask signals matrix
        self.mask_signals_matrix /= (np.sum(self.mask_signals_matrix, axis=1, keepdims=True) + 1e-10)  # avoid division by zero

        self.mask_noise = self.mask_noise.astype(np.float32) / np.sum(self.mask_noise)


    @profile
    def calc_snr(self, x):
        # make sure signal is complex
        assert np.iscomplexobj(x), "Input signal must be complex"

        # Compute FFT
        assert len(x) == self.N, f"Input signal length must be {self.N}, got {len(x)}"
        # no need to do fftshift because the order doesn't matter for SNR calculation
        X = np.fft.fft(x)

        psd = np.abs(X)**2 / self.N  # power spectrum estimate

        signals = psd @ self.mask_signals_matrix.T
        noise = psd @ self.mask_noise
        snrs = 10 * np.log10(signals / noise + 1e-10)  # avoid log(0)
        return snrs

def test():
    from afsk import AFSKEncoder

    afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=1800)
    test_bits = '0' * 5 + '01' * 20 + '0' + '1110010' + '0' * 2
    audio_signal = afsk.encode(test_bits)

    fm_modulator = FMModulator(sample_rate=2000000, audio_sample_rate=44100, deviation=2500)
    # shift by 25kHz to simulate RF transmission
    fm_signal = fm_modulator.modulate(audio_signal, shift=25000)

    audio_signal_demod = fm_modulator.demodulate(fm_signal, shift=-25000)

    decoded_bits = afsk.decode(audio_signal_demod, frame_sync='0101010101')
    print('decoded:', decoded_bits)
    print('original:', test_bits)

    assert decoded_bits in test_bits, "Decoded bits do not match original bits"


if __name__ == "__main__":
    test()
