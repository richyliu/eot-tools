#!/usr/bin/env -S python3 -u

from datetime import datetime
import time
import sys
import wave
import numpy as np
from scipy.io import wavfile
import os

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        """Dummy profile decorator if line_profiler is not installed"""
        return func

from parser import EOTParser, HOTParser, DPUParser
from afsk import AFSKEncoder
from fm import FMModulator, SignalDetector


def open_recording_as_raw(filename):
    """
    Open a HackRF or similar recording of I/Q data.
    """
    _, _, _, center_freq_raw, sample_rate_raw, num_format = filename.split('_')
    center_freq = int(center_freq_raw)
    fs = int(sample_rate_raw)

    type_ = None
    scale = 1.0
    if num_format.startswith('i8c'):
        type_ = np.int8
        scale = 1/2**7
    elif num_format.startswith('i16c'):
        type_ = np.int16
        scale = 1/2**15
    elif num_format.startswith('fc'):
        type_ = np.float32
        scale = 1.0
    else:
        raise Exception(f'Unknown recording number format {num_format} in file "{filename}"')

    # Read raw interleaved IQ data
    raw_data = np.memmap(filename, dtype=type_, mode='r')

    print(f'File: {filename}, sample count: {len(raw_data)} ({len(raw_data)/fs/2:.2f}) seconds)')

    return center_freq, fs, scale, raw_data


def power_spectrogram_one_line(sig, sample_rate, num_buckets=100, power_symbols=' .:+oO#@'):
    X = np.fft.fftshift(np.fft.fft(sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), 1 / sample_rate))
    psd = np.abs(X)**2 / len(sig)  # power spectrum estimate

    # divide the spectrum into buckets and get the average power in each bucket
    bucket_size = len(psd) // num_buckets
    psd_buckets = np.array([np.mean(10 * np.log10(psd[i * bucket_size:(i + 1) * bucket_size])) for i in range(num_buckets)])
    # bucket_freqs = np.array([np.mean(freqs[i * bucket_size:(i + 1) * bucket_size]) for i in range(num_buckets)])

    # plot the power spectrum by using a symbol for each bucket
    max_power = np.max(psd_buckets)
    min_power = np.min(psd_buckets)
    psd_buckets_scaled = (psd_buckets - min_power) / (max_power - min_power) * (len(power_symbols) - 1)
    power_plot = ''.join(power_symbols[int(p)] for p in psd_buckets_scaled)

    return power_plot, max_power, min_power


@profile
def main(argv):
    if len(argv) > 1:
        center_freq = int(argv[1])
    else:
        center_freq = 455_500_000  # in Hz

    if len(argv) > 2:
        fs = int(argv[2])  # sample rate in Hz
    else:
        fs = 6_000_000  # in Hz

    if len(argv) > 3:
        num_format = argv[3]
    else:
        num_format = 'i16c'

    if num_format.startswith('i8c'):
        dtype = np.int8
        scale = 1/2**7
    elif num_format.startswith('i16c'):
        dtype = np.int16
        scale = 1/2**15
    elif num_format.startswith('fc'):
        dtype = np.float32
        scale = 1.0
    else:
        raise Exception(f'Unknown number format "{num_format}"')

    print(f'Streaming mode: sample rate={fs} Hz, center frequency={center_freq/1e6:.3f} MHz, dtype={dtype}, scale={scale}')

    chunk_size = 1 << 15 # in bytes
    chunk_size //= dtype().itemsize
    print_every = int(.5 * fs / (chunk_size // dtype().itemsize // 2))
    print(f'{print_every=}')

    counter = 0
    parsed = 0
    fd = sys.stdin.buffer.fileno()

    fm_modulator = FMModulator(sample_rate=fs, audio_sample_rate=44100)
    eot = EOTParser()
    hot = HOTParser()
    dpu = DPUParser()

    decoders = [eot, hot, dpu]
    snr_threshold = 16  # dB
    max_last_signal_length = 1.0 # in seconds
    min_signal_length = 0.1  # in seconds

    target_freqs = []
    for decoder in decoders:
        target_freqs.extend(decoder.freqs)
    target_freqs.sort()
    target_freqs = np.array(target_freqs, dtype=np.float32)
    signal_detector = SignalDetector(target_freqs - center_freq, chunk_size // dtype().itemsize // 2, fs=fs, deviation=2500)

    last_signal_vals = [[] for _ in range(len(target_freqs))]

    t = 0

    start_time = time.perf_counter()

    while True:
        raw = os.read(fd, chunk_size)
        if not raw:
            break  # EOF
        if len(raw) < chunk_size:
            print(f'[WARNING] read({fd}, {chunk_size}) only returned length {len(raw)}')
            continue

        data = np.frombuffer(raw, dtype=dtype)
        parsed += len(data) // 2  # each sample is a pair of values (I and Q)

        t += len(data) // 2 / fs  # update time in seconds

        section = (data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)) * scale

        counter += 1
        if counter == print_every:
            counter = 0

            power_plot, max_power, min_power = power_spectrogram_one_line(section, fs, num_buckets=60)

            timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            now = time.perf_counter()
            rate = parsed / (now - start_time)
            print(f'{timestr} t={t:5.1f} ({rate/1e6:4.1f} MSPS) |{power_plot}| min={min_power:.1f}, max={max_power:.1f}, snrs=', end='')
            for j in range(len(snrs)):
                print(f'{snrs[j]:5.1f}', end=',')
            # print('  ', end='')
            # for i in range(len(last_signal_vals)):
            #     print(f'{len(last_signal_vals[i])}', end=',')
            print()
            sys.stdout.flush()

        snrs = signal_detector.calc_snr(section)
        if np.all(snrs < snr_threshold):
            for i in range(len(last_signal_vals)):
                if len(last_signal_vals[i]) > 0:
                    break
            else:
                # No signals detected, skip this section
                continue

        # print(f't={t:.3f} len(last_signal_vals[..])=', end='')
        # for i in range(len(last_signal_vals)):
        #     print(f'{len(last_signal_vals[i])}', end=',')
        # print()

        for j in range(len(snrs)):
            for k in range(len(decoders)):
                if target_freqs[j] in decoders[k].freqs:
                    decoder = decoders[k]
                    break
            else:
                raise ValueError(f'No decoder found for frequency {target_freqs[j]} Hz')

            if snrs[j] > snr_threshold:
                if len(last_signal_vals[j]) == 0:
                    print(f't={t:.3f} got signal freq={target_freqs[j]/1e3:.1f}kHz snr={snrs[j]}')
                if len(last_signal_vals[j]) < int(max_last_signal_length * fs) // len(section):
                    last_signal_vals[j].append(section)
                else:
                    print(f't={t:.3f} signal overflow freq={target_freqs[j]/1e3:.1f}kHz snr={snrs[j]}')
            elif len(last_signal_vals[j]) > 0:
                if len(last_signal_vals[j]) >= int(min_signal_length * fs) // len(section):
                    # end of a signal segment
                    print(
                        f't={t:.3f}s',
                        f'length={len(last_signal_vals[j])*len(section)/fs:.2f}s',
                        f'freq={target_freqs[j]/1e3:.1f}kHz',
                        f'decoder={decoder}'
                    )
                    try:
                        afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=decoder.afsk_mark_freq, space_freq=decoder.afsk_space_freq)
                        vals = np.concatenate(last_signal_vals[j])
                        audio_signal_demod = fm_modulator.demodulate(vals, shift=target_freqs[j] - center_freq)
                        decoded_bits = afsk.decode(audio_signal_demod, frame_sync=decoder.frame_sync)
                        print(f'Decoded bits: {decoded_bits}')
                        data = decoder.decode(decoded_bits)
                        decoder.pretty_print(data)
                    except Exception as ex:
                        print('[WARNING] Decoding error:', ex)
                else:
                    print(f't={t:.3f} signal too short freq={target_freqs[j]/1e3:.1f}kHz length={len(last_signal_vals[j])}')
                last_signal_vals[j] = []


if __name__ == "__main__":
    main(sys.argv)
