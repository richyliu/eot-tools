#!/usr/bin/env python3

import argparse
import sys
from encode_eot import EOTPacket


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
  %(prog)s -a out.wav
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
    
    # Output options
    parser.add_argument('-a', '--save-audio-file', 
                        type=str, default=None,
                        help='Save the generated audio to a WAV file (default: None)')
    
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

        # Save to audio file if specified
        if args.save_audio_file:
            from modulate import EOTRF
            eotrf = EOTRF()
            eotrf.with_message(encoded_packet, padded_silence=0.2)

            eotrf.save_audio(args.save_audio_file)
            print(f"Audio saved to {args.save_audio_file}", file=sys.stderr)
        else:
            print("No audio file specified, skipping save.", file=sys.stderr)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
