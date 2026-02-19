#!/usr/bin/env python3

import serial
import time

def crc8(buf: bytes) -> int:
    uVar1 = 0
    data = bytearray(buf)  # make a mutable copy
    data[-1] = 0           # match buf[len-1] = 0
    
    for b in data:
        uVar2 = 0x80
        for _ in range(8):
            uVar1 <<= 1
            if b & uVar2:
                uVar1 |= 1
            uVar2 >>= 1
            if (uVar1 << 23) & 0x80000000:  # equivalent to (int)(uVar1 << 0x17) < 0
                uVar1 ^= 0x135
    return uVar1 & 0xFF

ser = None
try:
    # Open the serial port
    ser = serial.Serial('/dev/cu.usbserial-1320', 115200, timeout=1) # Adjust port and baud rate as needed
    print(f"Serial port {ser.name} opened successfully.")

    # first, second, rest = 2, 6, bytes([1, 2, 3, 4])
    first, second, rest = 0, 0, bytes()

    data = bytes([first, second]) + rest
    length = len(data)
    crc_input = bytes([length]) + data + bytes([0])
    crc = crc8(crc_input)
    packet = bytes([length]) + data + bytes([crc])
    print(f'Sending packet: {[f"0x{b:02X}" for b in packet]}')
    ser.write(packet)

    while True:
        got = ser.read(1)
        if len(got) == 0:
            print('No data received, exiting.')
            break
        char = got[0]
        print(f'0x{char:02X} ({char:08b})')
        time.sleep(0.1)

except serial.SerialException as e:
    print(f"Error opening or communicating with serial port: {e}")
except KeyboardInterrupt:
    print("Exiting program.")
finally:
    if ser and ser.is_open:
        ser.close()
        print(f"Serial port {ser.name} closed.")
