#!/usr/bin/env -S python3 -u

from datetime import datetime
import time
import sys
import numpy as np
import scipy.signal as signal
import os

from fm import FMModulator

def main(argv):
    chunk_size = 4096  # bytes
    in_dtype = np.int32
    in_scale = np.iinfo(in_dtype).max
    in_fs = 48000

    out_dtype = np.int16
    out_scale = np.iinfo(out_dtype).max
    out_fs = 800000

    fd = sys.stdin.buffer.fileno()

    phase = 0
    while True:
        raw = os.read(fd, chunk_size)
        if not raw:
            break  # EOF
        if len(raw) < chunk_size:
            print(f'[WAV] read({fd}, {chunk_size}) only returned length {len(raw)}', file=sys.stderr)
            continue

        data = np.frombuffer(raw, dtype=in_dtype)
        data = data.astype(np.float32) / in_scale

        fm_modulator = FMModulator(sample_rate=out_fs, audio_sample_rate=in_fs, deviation=2500)
        iq_data, phase = fm_modulator.modulate(data, shift=0, start_phase=phase)
        # print(iq_data, file=sys.stderr)

        # convert to packed int16 I/Q
        # interleave I and Q
        output = np.empty((iq_data.size * 2,), dtype=out_dtype)
        output[0::2] = np.clip(iq_data.real * out_scale, np.iinfo(out_dtype).min, np.iinfo(out_dtype).max).astype(out_dtype)
        output[1::2] = np.clip(iq_data.imag * out_scale, np.iinfo(out_dtype).min, np.iinfo(out_dtype).max).astype(out_dtype)
        os.write(sys.stdout.buffer.fileno(), output.tobytes())

if __name__ == "__main__":
    main(sys.argv)
