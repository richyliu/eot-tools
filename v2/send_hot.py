#!/usr/bin/env -S python3 -u

import argparse
import sys
import numpy as np

from parser import EOTParser, HOTParser, DPUParser
from afsk import AFSKEncoder
from fm import FMModulator

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Send HOT (Head of Train) packets with customizable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --unit-addr 12345 --command EMR
        '''
    )

    parser.add_argument('-c', '--command',
                        choices=['SRQ', 'EMR'], default='SRQ',
                        help='Command type (SRQ or EMR, default: SRQ)')

    parser.add_argument('-u', '--unit-addr',
                        type=int, default=12345,
                        help='Unit address (0-99999, default: 12345)')

    parser.add_argument('-f', '--frequency',
                        type=int, default=455_500_000,
                        help='Center frequency in Hz (default: 455500000)')

    parser.add_argument('-a', '--sample-rate',
                        type=int, default=6_000_000,
                        help='Sample rate in Hz (default: 6000000)')

    parser.add_argument('-t', '--num-format',
                        choices=['i8c', 'i16c', 'fc'], default='i16c',
                        help='Number format (i8c, i16c, fc, default: i16c)')

    return parser.parse_args()



def iq_stream_out(signal, dtype=np.int16, scale=1/2**15, noise_level=0.01):
    signal += (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * noise_level
    # convert to I/Q format (interleaved) and write to stdout
    iq_signal = np.empty((signal.size * 2,), dtype=dtype)
    iq_signal[0::2] = np.real(signal) / scale
    iq_signal[1::2] = np.imag(signal) / scale
    sys.stdout.buffer.write(iq_signal.tobytes())
    # print(f'[SEND] sent {len(signal)} complex samples ({len(iq_signal) * iq_signal.itemsize / 1e3:.2f} kB)', file=sys.stderr)

def main(argv):
    args = parse_arguments()

    center_freq = args.frequency
    fs = args.sample_rate
    num_format = args.num_format

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

    print(f'[SEND] sample rate={fs} Hz, center frequency={center_freq/1e6:.3f} MHz, dtype={dtype}, scale={scale}', file=sys.stderr)

    hot = HOTParser()
    afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=hot.afsk_mark_freq, space_freq=hot.afsk_space_freq)
    fm_modulator = FMModulator(sample_rate=fs, audio_sample_rate=44100)

    bitstr = hot.encode_fields(unit_addr=args.unit_addr, command=args.command)
    print('[SEND]', hot.decode(bitstr), file=sys.stderr)
    audio_signal = afsk.encode(bitstr)
    fm_signal, _ = fm_modulator.modulate(audio_signal, shift=center_freq - hot.freqs[0])
    print(f'[SEND] Generated FM signal with {len(fm_signal)} complex samples', file=sys.stderr)

    # pad with silence
    silence = np.zeros(int(fs * 0.5), dtype=np.complex64)
    signal = np.concatenate([silence, fm_signal, silence])

    iq_stream_out(signal, dtype=dtype, scale=scale, noise_level=0.01)


if __name__ == "__main__":
    main(sys.argv)
