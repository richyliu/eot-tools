#!/usr/bin/env python3

import argparse
import sys

def xor(a, b):
    result = []
    for i in range(len(b)):
        if a[i] == b[i]:
            result.append('0')
        else:
            result.append('1')
    return ''.join(result)


# Reverse string
def reverse(data):
    return ''.join(data[::-1])


# Perform modulo-2 division on two strings of binary symbols
def mod2div(dividend, divisor):

    # Number of bits to be XORed at a time.
    pick = len(divisor)

    # Slicing the dividend to appropriate
    # length for particular step
    tmp = dividend[0:pick]

    while pick < len(dividend):

        if tmp[0] == '1':

            # replace the dividend by the result
            # of XOR and pull 1 bit down
            tmp = xor(divisor[1:], tmp[1:]) + dividend[pick]

        else:   # If leftmost bit is '0'
            # If the leftmost bit of the dividend (or the
            # part used in each step) is 0, the step cannot
            # use the regular divisor; we need to use an
            # all-0s divisor.
            tmp = xor(('0'*pick)[1:], tmp[1:]) + dividend[pick]

        # increment pick to move further
        pick += 1

    # For the last n bits, we have to carry it out
    # normally as increased value of pick will cause
    # Index Out of Bounds.
    if tmp[0] == '1':
        tmp = xor(divisor[1:], tmp[1:])
    else:
        tmp = xor(('0'*pick)[1:], tmp[1:])

    remainder = tmp
    return remainder


class EOTPacket:
    def __init__(self,
                 batt_cond=3,
                 message_type='normal',
                 unit_addr=123456,
                 pressure=60,
                 batt_charge=70,
                 valve_circuit_operational=True,
                 confirmation_indicator=True,
                 turbine_status=True,
                 motion_detection=True,
                 marker_light_battery_weak=False,
                 marker_light_status=True):
        assert batt_cond in [0, 1, 2, 3], "Invalid battery condition"
        assert message_type in ['normal', 'arm'], "Invalid message type"
        assert 0 <= unit_addr <= 0x1FFFF, "Unit address must be between 0 and 0x1FFFF"
        assert 0 <= pressure <= 127, "Pressure must be between 0 and 127"
        assert 0 <= batt_charge <= 127, "Battery charge must be between 0 and 127"

        self.batt_cond = batt_cond
        self.message_type = 0 if message_type == 'normal' else 0b111
        self.unit_addr = unit_addr
        self.pressure = pressure
        self.batt_charge = batt_charge
        self.valve_circuit_operational = int(valve_circuit_operational)
        self.confirmation_indicator = int(confirmation_indicator)
        self.turbine_status = int(turbine_status)
        self.motion_detection = int(motion_detection)
        self.marker_light_battery_weak = int(marker_light_battery_weak)
        self.marker_light_status = int(marker_light_status)

    # Calculate BCH checkbits
    def calc_checkbits(self, data, key):
        appended_data = data + '0'*(len(key)-1)  # Appends n-1 zeros at end of data
        remainder = mod2div(appended_data, key)
        return ''.join(remainder)

    def encode_message(self, bit_sync_bits=1000):
        data = self.encode_data_block()
        frame_sync = '11100010010'  # Frame sync bits

        packet = ''
        packet += '01' * (bit_sync_bits//2) + '0'
        packet += frame_sync  # Frame sync
        packet += data

        # duplicate the packet for redundancy
        packet += packet

        return packet

    def encode_data_block(self):
        # note that all values are in binary little endian format
        
        packet = ''
        packet += '11' # chaining bits
        packet += '{:02b}'.format(self.batt_cond)[::-1] # battery condition
        packet += '{:03b}'.format(self.message_type)[::-1] # message type
        packet +='{:017b}'.format(self.unit_addr)[::-1] # unit address
        packet += '{:07b}'.format(self.pressure)[::-1] # pressure
        packet += '{:07b}'.format(self.batt_charge)[::-1] # battery charge
        packet += '0' # discretionary
        packet += '{:01b}'.format(self.valve_circuit_operational) # valve circuit operational
        packet += '{:01b}'.format(self.confirmation_indicator)
        packet += '{:01b}'.format(self.turbine_status)
        packet += '{:01b}'.format(self.motion_detection)
        packet += '{:01b}'.format(self.marker_light_battery_weak)
        packet += '{:01b}'.format(self.marker_light_status)

        # generate BCH code
        data_block = packet
        assert len(data_block) == 45, "Data block must be 45 bits long"
        generator = '1111001101000001111'  # BCH generator polynomial
        cipher_key = '101011011101110000'  # XOR cipher key
        data_block = reverse(data_block)
        checkbits = self.calc_checkbits(data_block, generator)
        checkbits_cipher = xor(checkbits, cipher_key)
        packet += checkbits_cipher  # append checkbits

        packet += '1' # dummy bit

        assert len(packet) == 64, "Encoded packet must be 64 bits long"

        print('raw packet:', packet, file=sys.stderr)

        return packet

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
        description='Generate EOT (End of Train) packets with customizable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --unit-addr 654321 --pressure 80 --message-type arm
  %(prog)s -u 100000 -p 45 -c 85 --no-turbine --marker-battery-weak
  %(prog)s -b 2 -m arm -s 800
        '''
    )
    
    # Battery condition
    parser.add_argument('-b', '--battery-condition', 
                        type=int, choices=[0, 1, 2, 3], default=3,
                        help='Battery condition (0-3, default: 3)')
    
    # Message type
    parser.add_argument('-m', '--message-type', 
                        choices=['normal', 'arm'], default='normal',
                        help='Message type (normal or arm, default: normal)')
    
    # Unit address
    parser.add_argument('-u', '--unit-addr', 
                        type=int, default=123456,
                        help='Unit address (0-131071, default: 123456)')
    
    # Pressure
    parser.add_argument('-p', '--pressure', 
                        type=int, default=60,
                        help='Pressure value (0-127, default: 60)')
    
    # Battery charge
    parser.add_argument('-c', '--battery-charge', 
                        type=int, default=70,
                        help='Battery charge (0-127, default: 70)')
    
    # Valve circuit operational
    parser.add_argument('--valve-circuit', '--valve', 
                        type=str2bool, nargs='?', const=True, default=True,
                        help='Valve circuit operational (default: True)')
    parser.add_argument('--no-valve-circuit', '--no-valve', 
                        dest='valve_circuit', action='store_false',
                        help='Set valve circuit as non-operational')
    
    # Confirmation indicator
    parser.add_argument('--confirmation-indicator', '--confirm', 
                        type=str2bool, nargs='?', const=True, default=True,
                        help='Confirmation indicator (default: True)')
    parser.add_argument('--no-confirmation-indicator', '--no-confirm', 
                        dest='confirmation', action='store_false',
                        help='Disable confirmation indicator')
    
    # Turbine status
    parser.add_argument('--turbine', 
                        type=str2bool, nargs='?', const=True, default=True,
                        help='Turbine status (default: True)')
    parser.add_argument('--no-turbine', 
                        dest='turbine', action='store_false',
                        help='Set turbine as non-operational')
    
    # Motion detection
    parser.add_argument('--motion', 
                        type=str2bool, nargs='?', const=True, default=True,
                        help='Motion detection (default: True, meaning motion detected)')
    parser.add_argument('--no-motion', 
                        dest='motion', action='store_false',
                        help='Set no motion detected')
    
    # Marker light battery weak
    parser.add_argument('--marker-battery-weak', '--marker-weak', 
                        action='store_true', default=False,
                        help='Marker light battery weak (default: False)')
    parser.add_argument('--no-marker-battery-weak', '--no-marker-weak', 
                        dest='marker_battery_weak', action='store_false',
                        help='Marker light battery OK or not monitored')
    
    # Marker light status
    parser.add_argument('--marker-light', '--marker', 
                        type=str2bool, nargs='?', const=True, default=True,
                        help='Marker light status (default: True)')
    parser.add_argument('--no-marker-light', '--no-marker', 
                        dest='marker_light', action='store_false',
                        help='Set marker light status to false')
    
    # Bit sync bits
    parser.add_argument('-s', '--bit-sync-bits', 
                        type=int, default=1000,
                        help='Number of bit sync bits (default: 1000)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Validate unit address range
    if not (0 <= args.unit_addr <= 0x1FFFF):
        print(f"Error: Unit address must be between 0 and {0x1FFFF} (131071)", file=sys.stderr)
        sys.exit(1)
    
    # Validate pressure range
    if not (0 <= args.pressure <= 127):
        print("Error: Pressure must be between 0 and 127", file=sys.stderr)
        sys.exit(1)
    
    # Validate battery charge range
    if not (0 <= args.battery_charge <= 127):
        print("Error: Battery charge must be between 0 and 127", file=sys.stderr)
        sys.exit(1)
    
    # Validate bit sync bits (must be even)
    if args.bit_sync_bits % 2 != 0:
        print("Error: Bit sync bits must be even", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create EOTPacket with parsed arguments
        eot = EOTPacket(
            batt_cond=args.battery_condition,
            message_type=args.message_type,
            unit_addr=args.unit_addr,
            pressure=args.pressure,
            batt_charge=args.battery_charge,
            valve_circuit_operational=args.valve_circuit,
            confirmation_indicator=args.confirmation,
            turbine_status=args.turbine,
            motion_detection=args.motion,
            marker_light_battery_weak=args.marker_battery_weak,
            marker_light_status=args.marker_light
        )
        
        # Generate and print the encoded packet
        encoded_packet = eot.encode_message(bit_sync_bits=args.bit_sync_bits)
        print(encoded_packet)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
