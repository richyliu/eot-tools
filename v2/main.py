#!/usr/bin/env python3

from datetime import datetime
import time
import sys
import wave
import numpy as np
from scipy.io import wavfile
import os

from parser import EOTParser, HOTParser, DPUParser
from afsk import AFSKEncoder
from fm import FMModulator


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


def power_spectrogram_one_line(sig, sample_rate, num_buckets=50, power_symbols=' .,:ilwW'):
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


def fm_demod(fname):
    center_freq, sample_rate, scale, raw_data = open_recording_as_raw(fname)
    
    fm_modulator = FMModulator(sample_rate=sample_rate, audio_sample_rate=44100, deviation=2500)
    afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=1800)

    import matplotlib.pyplot as plt
    # also plot fft

    start = int(6_000_000 * 4.0)  # 100 ms
    duration = int(6_000_000 * 0.4)  # 100 ms
    data_section = raw_data[start*2:(start + duration)*2]  # 2 bytes per sample (I and Q)
    data_section = data_section.astype(np.float32) * scale  # scale to float32
    section = data_section[::2] + 1j * data_section[1::2]  # convert interleaved IQ to complex signal

    X = np.fft.fftshift(np.fft.fft(section))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(section), 1 / sample_rate))
    psd = np.abs(X)**2 / len(section)  # power spectrum estimate

    plt.figure(figsize=(15, 6))
    plt.plot((freqs + center_freq) / 1e6, 10 * np.log10(psd), label='Power Spectrum')
    plt.title(f'Power Spectrum of Signal Section @ {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.grid()
    plt.legend()
    plt.savefig('power_spectrum.png')
    plt.close()


    duration = int(6_000_000 * 0.4)
    for i in range(0, len(raw_data), duration * 2):
        if i + duration * 2 > len(raw_data):
            break
        data_section = raw_data[i:i + duration*2]
        data_section = data_section.astype(np.float32) * scale  # scale to float32
        section = data_section[::2] + 1j * data_section[1::2]  # convert interleaved IQ to complex signal

        power_plot, max_power, min_power = power_spectrogram_one_line(section, sample_rate)

        time = i / (sample_rate * 2)  # convert to seconds
        print(f'{time:.1f} |{power_plot}| min={min_power:.1f}, max={max_power:.1f}')


def streaming(argv):
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

    chunk_size = 65536 # in bytes
    skip = 200

    skip_counter = 0
    parsed = 0
    fd = sys.stdin.buffer.fileno()

    fm_modulator = FMModulator(sample_rate=fs, audio_sample_rate=44100)
    eot = EOTParser()
    hot = HOTParser()
    dpu = DPUParser()

    decoders = [eot, hot, dpu]
    snr_threshold = 10  # dB
    max_last_signal_length = 1.0 # in seconds
    min_signal_length = 0.1  # in seconds
    window_size = int(fs * 0.02)  # 20 ms window size

    target_freqs = []
    for decoder in decoders:
        target_freqs.extend(decoder.freqs)
    target_freqs = np.array(target_freqs, dtype=np.float32)

    last_signal_vals = [np.array([], dtype=np.float32)] * len(target_freqs)

    t = 0

    start_time = time.perf_counter()

    while True:
        raw = os.read(fd, chunk_size)
        if not raw:
            break  # EOF

        data = np.frombuffer(raw, dtype=dtype)
        parsed += len(data) // 2  # each sample is a pair of values (I and Q)

        t += len(data) // 2 / fs  # update time in seconds

        # Convert to complex64
        data = data.astype(np.float32) * scale  # scale to float32
        section = data[0::2] + 1j * data[1::2]

        snrs = fm_modulator.calc_snr(section, fs, target_freqs - center_freq, deviation=7500)
        for j in range(len(snrs)):
            for k in range(len(decoders)):
                if target_freqs[j] in decoders[k].freqs:
                    decoder = decoders[k]
                    break
            else:
                raise ValueError(f'No decoder found for frequency {target_freqs[j]} Hz')

            if snrs[j] > snr_threshold and len(last_signal_vals[j]) < int(max_last_signal_length * fs):
                # print(f'SNR: {snrs[j]:.1f}dB, freq={target_freqs[j]/1e3:.1f}kHz, decoder={decoder}')
                last_signal_vals[j] = np.concatenate((last_signal_vals[j], section))
            elif len(last_signal_vals[j]) > 0:
                if len(last_signal_vals[j]) >= int(min_signal_length * fs):
                    # end of a signal segment
                    print(
                        f't={t:.2f}s',
                        f'length={len(last_signal_vals[j])/fs:.2f}s',
                        f'freq={target_freqs[j]/1e3:.1f}kHz',
                        f'decoder={decoder}'
                    )
                    sys.stdout.flush()
                    try:
                        afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=decoder.afsk_mark_freq, space_freq=decoder.afsk_space_freq)
                        audio_signal_demod = fm_modulator.demodulate(last_signal_vals[j], shift=target_freqs[j] - center_freq)
                        decoded_bits = afsk.decode(audio_signal_demod, frame_sync=decoder.frame_sync)
                        print(f'Decoded bits: {decoded_bits}')
                        data = decoder.decode(decoded_bits)
                        decoder.pretty_print(data)
                    except Exception as ex:
                        print('[WARNING] Decoding error:', ex)
                last_signal_vals[j] = np.array([], dtype=np.float32)

        skip_counter += 1
        if skip_counter < skip:
            continue
        skip_counter = 0

        power_plot, max_power, min_power = power_spectrogram_one_line(section, fs, num_buckets=100)

        timestr = datetime.now().strftime('%H:%M:%S.%f')[:-4]  # HH:MM:SS.ss
        now = time.perf_counter()
        rate = parsed / (now - start_time)
        print(f'{timestr} ({rate/1e6:.2f} MSPS) |{power_plot}| min={min_power:.1f}, max={max_power:.1f}, snrs=', end='')
        for j in range(len(snrs)):
            print(f'{snrs[j]:5.1f}dB', end=',')
        print()
        sys.stdout.flush()


if __name__ == "__main__":
    streaming(sys.argv)
    # main(sys.argv[1])
    # fm_demod(sys.argv[1])
