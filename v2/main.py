#!/usr/bin/env python3

import wave
import numpy as np

from parser import EOTParser, HOTParser
from afsk import AFSKEncoder

def main():
    packets = []

    data = {
        'chaining_bits': 0b11,
        'batt_cond': 3,
        'message_type': 'normal',
        'unit_addr': 43210,
        'pressure': 73,
        'batt_charge': 23,
        'discretionary': 0,
        'valve_circuit_operational': 1,
        'confirmation_indicator': 0,
        'turbine_status': 1,
        'motion_detection': 0,
        'marker_light_battery_weak': 0,
        'marker_light_status': 0,
        'dummy': 1
    }
    eot = EOTParser()
    packets.append(eot.encode(data))

    data = {
        'chaining_bits': 0b11,
        'message_type': 'normal',
        'unit_addr': 54321,
        'command': 0b10101010,
        'dummy': 0
    }
    hot = HOTParser()
    packets.append(hot.encode(data))

    encoded = '0' * 500 + ('0' * 100).join(packets) + '0' * 500
    
    afsk = AFSKEncoder(sample_rate=44100, baud_rate=1200, mark_freq=1200, space_freq=1800)
    audio_signal = afsk.encode(encoded)

    with wave.open('out.wav', 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes((audio_signal * 32767).astype(np.int16).tobytes())
        print("Audio signal saved to out.wav")


if __name__ == "__main__":
    main()
