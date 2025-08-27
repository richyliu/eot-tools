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
    power_plot = ''.join(power_symbols[int(p)] if not np.isnan(p) else '!' for p in psd_buckets_scaled)

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

    signal_on_snr_threshold = 10  # dB relative to baseline
    signal_off_snr_threshold = 5  # dB relative to baseline
    signal_hysteresis_duration = 0.01 # seconds
    max_last_signal_length = 1.0 # in seconds
    snr_ema_alpha = 1/5

    chunk_size = 1 << 15 # in bytes
    chunk_size //= dtype().itemsize
    print_interval_secs = 0.5
    print_every = int(print_interval_secs * fs / (chunk_size // dtype().itemsize // 2))
    print(f'{print_every=}')

    counter = 0
    parsed = 0
    fd = sys.stdin.buffer.fileno()

    fm_modulator = FMModulator(sample_rate=fs, audio_sample_rate=44100)
    eot = EOTParser()
    hot = HOTParser()
    dpu = DPUParser()

    decoders = [eot, hot, dpu]
    decoder_per_freq = []

    target_freqs = []
    for decoder in decoders:
        target_freqs.extend(decoder.freqs)
    target_freqs.sort()
    for f in target_freqs:
        for decoder in decoders:
            if f in decoder.freqs:
                decoder_per_freq.append(decoder)
                break
        else:
            raise Exception(f'No decoder found for frequency {f}')
    target_freqs = np.array(target_freqs, dtype=np.float32)
    signal_detector = SignalDetector(target_freqs - center_freq, chunk_size // dtype().itemsize // 2, fs=fs, deviation=1500)

    last_signal_vals = [[] for _ in range(len(target_freqs))]
    snr_ema = np.ones(len(target_freqs), dtype=np.float32) * -100.0
    signal_snr_baseline = 100.0  # dB
    snrs_above_on_threshold = np.zeros(len(target_freqs), dtype=np.int32)
    snrs_below_off_threshold = np.zeros(len(target_freqs), dtype=np.int32)
    squelch_open = [False] * len(target_freqs)

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
        num_chunk_samples = len(data) // 2  # each sample is a pair of values (I and Q)
        parsed += num_chunk_samples

        chunk_time = len(data) // 2 / fs  # update time in seconds
        t += chunk_time

        section = (data[0::2].astype(np.float32) + 1j * data[1::2].astype(np.float32)) * scale

        counter += 1
        if counter == print_every:
            counter = 0

            power_plot, max_power, min_power = power_spectrogram_one_line(section, fs, num_buckets=30)

            timestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            now = time.perf_counter()
            rate = parsed / (now - start_time)
            print(f'{timestr} t={t:5.1f} ({rate/1e6:4.1f} MSPS) |{power_plot}| {min_power:.1f}->{max_power:.1f}dB snrs_ema=', end='')
            for i in range(len(snr_ema)):
                print(f'{int(snr_ema[i])}', end=',')
            print()
            sys.stdout.flush()

        snrs = signal_detector.calc_snr(section)

        # update exponential moving average of snrs
        snr_ema = snr_ema_alpha * snrs + (1 - snr_ema_alpha) * snr_ema

        # initialize snr_threshold dynamically
        if parsed > int(fs * 1) and signal_snr_baseline > 90:
            signal_snr_baseline = np.median(snr_ema)
            print(f'Baseline SNR established at {signal_snr_baseline:.1f} dB (ON SNR: {signal_snr_baseline + signal_on_snr_threshold:.1f} dB, OFF SNR: {signal_snr_baseline + signal_off_snr_threshold:.1f} dB)')

        # DEBUG snrs

        # print(f'l={parsed//num_chunk_samples} snrs=', end='')
        # for j in range(len(snrs)):
        #     print(f'{snrs[j]:.1f}', end=',')
        # print(f'snr_ema=', end='')
        # for j in range(len(snrs)):
        #     print(f'{snr_ema[j]:.1f}', end=',')
        # print()
        # continue

        for i in range(len(snr_ema)):
            if snr_ema[i] > signal_snr_baseline + signal_on_snr_threshold:
                snrs_above_on_threshold[i] += 1
            else:
                snrs_above_on_threshold[i] = 0
            if snr_ema[i] < signal_snr_baseline + signal_off_snr_threshold:
                snrs_below_off_threshold[i] += 1
            else:
                snrs_below_off_threshold[i] = 0

            decoder = decoder_per_freq[i]

            if not squelch_open[i]:
                if snrs_above_on_threshold[i] >= signal_hysteresis_duration * fs / num_chunk_samples:
                    # open the squelch (start recording signal)
                    squelch_open[i] = True
                    print(f't={t:.3f} (l={parsed//(len(data)//2)}) got signal freq={target_freqs[i]/1e3:.1f}kHz snr_ema={snr_ema[i]:.1f} decoder={decoder}')
                    last_signal_vals[i].append(section)
            else:
                # recording signal as long as it is not too long
                if len(last_signal_vals[i]) < int(max_last_signal_length * fs) // len(section):
                    last_signal_vals[i].append(section)
                else:
                    print(f't={t:.3f} (l={parsed//(len(data)//2)}) signal overflow freq={target_freqs[i]/1e3:.1f}kHz snr_ema={snr_ema[i]:.1f} decoder={decoder}')

                if snrs_below_off_threshold[i] >= signal_hysteresis_duration * fs / num_chunk_samples:
                    # close the squelch (stop recording signal)
                    print(
                        f't={t:.3f}s',
                        f'length={len(last_signal_vals[i])*len(section)/fs:.2f}s',
                        f'freq={target_freqs[i]/1e3:.1f}kHz',
                        f'decoder={decoder}'
                    )

                    try:
                        afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=decoder.afsk_mark_freq, space_freq=decoder.afsk_space_freq)
                        vals = np.concatenate(last_signal_vals[i])
                        audio_signal_demod = fm_modulator.demodulate(vals, shift=target_freqs[i] - center_freq)
                        decoded_bits = afsk.decode(audio_signal_demod, frame_sync=decoder.frame_sync)
                        print(f'Decoded bits: {decoded_bits}')
                        buf = decoded_bits
                        while decoder.frame_sync in buf:
                            print(f'Attempting to decode frame at index {len(decoded_bits) - len(buf)}')
                            data = decoder.decode(buf)
                            decoder.pretty_print(data)
                            buf = buf[buf.index(decoder.frame_sync) + len(decoder.frame_sync):]
                        else:
                            print('No complete frame sync found in the decoded bits')
                    except Exception as ex:
                        print('[WARNING] Decoding error:', ex)

                    last_signal_vals[i] = []
                    squelch_open[i] = False

if __name__ == "__main__":
    main(sys.argv)
