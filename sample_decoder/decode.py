#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyEOT End-of-Train Device Decoder
Copyright (c) 2018 Eric Reuter

This source file is subject of the GNU general public license

history:    2018-08-09 Initial Version

purpose:    Receives demodulated FFSK bitstream from GNU Radio, indentifes
            potential packets, and passes them to decoder classes for
            parsing and verification.  Finally human-readable data are printed
            to stdout.

            Requires eot_decoder.py and helpers.py
"""

import datetime
from eot_decoder import EOT_decode
import subprocess
import sys

def printEOT(EOT):
    localtime = str(datetime.datetime.now().
                    strftime('%Y-%m-%d %H:%M:%S.%f'))[:-3]
    print("")
    print("EOT {}".format(localtime))
    #   print(EOT.get_packet())
    print("---------------------")
    print("Unit Address:   {}".format(EOT.unit_addr))
    print("Pressure:       {} psig".format(EOT.pressure))
    print("Motion:         {}".format(EOT.motion))
    print("Marker Light:   {}".format(EOT.mkr_light))
    print("Marker Battery: {}".format(EOT.mkr_batt))
    print("Turbine:        {}".format(EOT.turbine))
    print("Battery Cond:   {}".format(EOT.batt_cond_text))
    print("Battery Charge: {}".format(EOT.batt_charge))
    print("Arm Status:     {}".format(EOT.arm_status))
    print("Conf Ind:       {}".format(EOT.conf_ind))
    print("Valve Circuit:  {}".format(EOT.valve_ckt_stat))
    print("Spare:          {}".format(EOT.spare))
    print("Message Type:   {}".format(EOT.message_type))
    print("Checkbits Rx:   {}".format(EOT.checkbitsRx))


def main(file_name):
    # run minimodem to read raw bits from wav file
    # minimodem -f output.wav -M 1200 -S 1800 --sync-byte 0xaa -c 1.0 --binary-raw 4 1200 | tr -d '\n'
    res = subprocess.run([
        'minimodem',
        '-f', file_name,
        '-M', '1200',
        '-S', '1800',
        # '--sync-byte', '0xaa',
        '-c', '1.1',
        '--binary-raw', '8',
        '-q',
        '1200'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("Error running minimodem: ", res.stderr.decode())
        return
    buffer = res.stdout.decode().replace('\n', '').replace(' ', '')

    # frame_sync = '1011100010010'
    frame_sync = '10101011100010010'
    while buffer.find(frame_sync) >= 0:
        buffer = buffer[buffer.find(frame_sync):]
        EOT = EOT_decode(buffer[6:])  # first 6 bits are bit sync
        if (EOT.valid):
            printEOT(EOT)
        else:
            print('invalid EOT packet found')
        buffer = buffer[1:]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} <wav_file>".format(sys.argv[0]))
        sys.exit(1)
    wav_file = sys.argv[1]
    main(wav_file)
