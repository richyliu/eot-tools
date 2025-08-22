#!/usr/bin/env python3

import enum
import sys

from bch_utils import mod2div, xor


class BinaryPacketFieldType(enum.Enum):
    UINTLE = 1
    UINTBE = 2
    INTLE = 3
    INTBE = 4
    ENUM = 5
    BCH = 6


def bch_encode(bin_str, generator, cipher_key):
    appended_data = bin_str[::-1] + '0'*(len(generator)-1)  # Appends n-1 zeros at end of data
    remainder = mod2div(appended_data, generator)
    checkbits = ''.join(remainder)
    if cipher_key is not None:
        checkbits = xor(checkbits, cipher_key)
    return checkbits


class BinaryPacketEncoderDecoder:
    def __init__(self):
        self.fields = []

    def add_field(self, name, size, type_, addl=None):
        if type_ not in BinaryPacketFieldType:
            raise ValueError("Invalid field type")

        if type_ == BinaryPacketFieldType.ENUM and addl is None:
            raise ValueError("Enum type requires an additional mapping dictionary (from LE binary values to enum strings)")
        if type_ == BinaryPacketFieldType.BCH and addl is None:
            raise ValueError("BCH type requires generator and cipher key as additional parameters")
        self.fields.append((name, size, type_, addl))

    def add_fields(self, fields):
        for field in fields:
            self.add_field(*field)

    def encode(self, data, debug=False):
        bin_str = ''

        for name, size, type_, addl in self.fields:
            if type_ == BinaryPacketFieldType.BCH:
                generator, cipher_key = addl
                bin_value = bch_encode(bin_str, generator, cipher_key)
                if len(bin_value) != size:
                    raise ValueError(f"BCH encoded value for {name} has incorrect size")
            else:
                if name not in data:
                    raise ValueError(f"Missing field: {name}")
                value = data[name]
                if type_ == BinaryPacketFieldType.UINTLE:
                    if not (0 <= value < (1 << size)):
                        raise ValueError(f"Value for {name} out of range")
                    bin_value = format(value, f'0{size}b')[::-1]
                elif type_ == BinaryPacketFieldType.UINTBE:
                    if not (0 <= value < (1 << size)):
                        raise ValueError(f"Value for {name} out of range")
                    bin_value = format(value, f'0{size}b')
                elif type_ == BinaryPacketFieldType.INTLE:
                    if not (-(1 << (size - 1)) <= value < (1 << (size - 1))):
                        raise ValueError(f"Value for {name} out of range")
                    if value < 0:
                        value = (1 << size) + value
                    bin_value = format(value, f'0{size}b')[::-1]
                elif type_ == BinaryPacketFieldType.INTBE:
                    if not (-(1 << (size - 1)) <= value < (1 << (size - 1))):
                        raise ValueError(f"Value for {name} out of range")
                    if value < 0:
                        value = (1 << size) + value
                    bin_value = format(value, f'0{size}b')
                elif type_ == BinaryPacketFieldType.ENUM:
                    if value not in addl.values():
                        raise ValueError(f"Invalid enum value for {name}")
                    inv_map = {v: k for k, v in addl.items()}
                    bin_value = format(inv_map[value], f'0{size}b')[::-1]

                if debug:
                    print(f"[DEBUG]: Field: {name}, Value: {value}, Binary: {bin_value}")

            bin_str += bin_value

        return bin_str

    def decode(self, bin_str):
        assert all(c in '01' for c in bin_str), "Input must be a binary string"

        data = {}
        index = 0
        for name, size, type_, addl in self.fields:
            if index + size > len(bin_str):
                raise ValueError("Binary string too short for decoding")
            bin_value = bin_str[index:index+size]
            index += size

            if type_ == BinaryPacketFieldType.UINTLE:
                value = int(bin_value[::-1], 2)
            elif type_ == BinaryPacketFieldType.UINTBE:
                value = int(bin_value, 2)
            elif type_ == BinaryPacketFieldType.INTLE:
                rev_bin = bin_value[::-1]
                if rev_bin[0] == '1':
                    value = int(rev_bin, 2) - (1 << size)
                else:
                    value = int(rev_bin, 2)
            elif type_ == BinaryPacketFieldType.INTBE:
                if bin_value[0] == '1':
                    value = int(bin_value, 2) - (1 << size)
                else:
                    value = int(bin_value, 2)
            elif type_ == BinaryPacketFieldType.ENUM:
                if addl is None:
                    raise ValueError(f"Enum mapping not provided for {name}")
                le_value = int(bin_value[::-1], 2)
                if le_value not in addl:
                    raise ValueError(f"Invalid enum binary value for {name}")
                value = addl[le_value]
            elif type_ == BinaryPacketFieldType.BCH:
                generator, cipher_key = addl
                value = bin_value
                calc_checkbits = bch_encode(bin_str[:index - size], generator, cipher_key)
                if value != calc_checkbits:
                    print(f"[WARNING] BCH checkbits do not match. Expected: {calc_checkbits}, got: {value}", file=sys.stderr)

            data[name] = value

        return data


class GenericParser:
    def __init__(self):
        self.encoder_decoder = BinaryPacketEncoderDecoder()
        self.frame_sync = ''

    def encode(self, data, bit_sync_bits=69, debug=False):
        raise NotImplementedError("Subclasses must implement encode method")

    def decode(self, bitstr, debug=False):
        if debug:
            print(f"[DEBUG] Decoding bit string: {bitstr}, frame sync: {self.frame_sync}")
        frame_sync_index = bitstr.find(self.frame_sync)
        if frame_sync_index == -1:
            if debug:
                print("[DEBUG] Frame sync not found in bit string")
            return None
        data_start = frame_sync_index + len(self.frame_sync)
        data_end = data_start + sum(size for _, size, _, _ in self.encoder_decoder.fields)
        if data_end > len(bitstr):
            raise ValueError("Bit string too short for decoding")

        data_bits = bitstr[data_start:data_end]
        data = self.encoder_decoder.decode(data_bits)
        return data


class EOTParser(GenericParser):
    def __init__(self):
        super().__init__()
        self.encoder_decoder.add_fields([
            ('chaining_bits', 2, BinaryPacketFieldType.UINTLE),
            ('batt_cond', 2, BinaryPacketFieldType.UINTLE),
            ('message_type', 3, BinaryPacketFieldType.ENUM, {0: 'normal', 0b111: 'arm'}),
            ('unit_addr', 17, BinaryPacketFieldType.UINTLE),
            ('pressure', 7, BinaryPacketFieldType.UINTLE),
            ('batt_charge', 7, BinaryPacketFieldType.UINTLE),
            ('discretionary', 1, BinaryPacketFieldType.UINTLE),
            ('valve_circuit_operational', 1, BinaryPacketFieldType.UINTLE),
            ('confirmation_indicator', 1, BinaryPacketFieldType.UINTLE),
            ('turbine_status', 1, BinaryPacketFieldType.UINTLE),
            ('motion_detection', 1, BinaryPacketFieldType.UINTLE),
            ('marker_light_battery_weak', 1, BinaryPacketFieldType.UINTLE),
            ('marker_light_status', 1, BinaryPacketFieldType.UINTLE),
            ('bch_checkbits', 18, BinaryPacketFieldType.BCH, ('1111001101000001111', '101011011101110000')),
            ('dummy', 1, BinaryPacketFieldType.UINTLE),
        ])
        self.frame_sync = '11100010010'

    def encode(self, data, bit_sync_bits=69, debug=False):
        bitstr = ''
        bitstr += '01' * (bit_sync_bits // 2) + '0'
        bitstr += self.frame_sync
        bitstr += self.encoder_decoder.encode(data, debug)
        # duplicate per EOT spec
        bitstr += bitstr
        return bitstr


class HOTParser(GenericParser):
    def __init__(self):
        super().__init__()
        self.encoder_decoder.add_fields([
            ('chaining_bits', 2, BinaryPacketFieldType.UINTLE),
            ('message_type', 3, BinaryPacketFieldType.ENUM, {0: 'normal'}), # note: 0b111 ('arm') not used in HOT
            ('unit_addr', 17, BinaryPacketFieldType.UINTLE),
            ('command', 8, BinaryPacketFieldType.UINTLE),
            ('bch_checkbits', 33, BinaryPacketFieldType.BCH, ('1110011011010111000010110011111011', None)),
            ('dummy', 1, BinaryPacketFieldType.UINTLE),
        ])
        self.frame_sync = '100011110001000100101001'

    def encode(self, data, bit_sync_bits=456, debug=False):
        bitstr = ''
        bitstr += '01' * (bit_sync_bits // 2) + '0'
        bitstr += self.frame_sync
        data = self.encoder_decoder.encode(data, debug)
        bitstr += data * 3
        if debug:
            print(f"[DEBUG] HOT data portion: {data}")
        return bitstr


def test():
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
    encoded = eot.encode(data, debug=True)
    print(f"[TEST]: Encoded: {encoded}")

    decoded = eot.decode(encoded)
    print(f"[TEST]: Decoded: {decoded}")
    for k in decoded:
        if k not in data:
            print(f"[TEST]: New item in decoded: {k} = {decoded[k]}")
        else:
            assert data[k] == decoded[k], f"Mismatch for {k}: {data[k]} != {decoded[k]}"


    data = {
        'chaining_bits': 0b11,
        'message_type': 'normal',
        'unit_addr': 54321,
        'command': 0b00000001,
        # 'bch_checkbits': 0,
        'dummy': 1
    }
    hot = HOTParser()
    encoded = hot.encode(data, debug=True)
    print(f"[TEST]: Encoded HOT: {encoded}")
    decoded = hot.decode(encoded)
    print(f"[TEST]: Decoded HOT: {decoded}")
    for k in decoded:
        if k not in data:
            print(f"[TEST]: New item in decoded: {k} = {decoded[k]}")
        else:
            assert data[k] == decoded[k], f"Mismatch for {k}: {data[k]} != {decoded[k]}"


    raw_data = '''
1100011110001000100101001110001100101001011111010101010110000100101010001101101001011010011000110010100101111101010101011000010010101000110110100101101001100011001010010111110101010101100001001010100011011010010110100000
1100011110001000100101001110000011111011100100110101010011111001111101100010001011101111011000001111101110010011010101001111100111110110001000101110111101100000111110111001001101010100111110011111011000100010111011111101
110001111000100010010100111000110010100101111101010101011000010010101000110110100101101001100011001010010111110101010101100001001010100011011010010110100110001100101001011111010101010110000100101010001101101001
1100011110001000100101001110000011111011100100110101010011111001111101100010001011101111011000001111101110010011010101001111100111110110001000101110111101100000111110111001001101010100111110011111011000100010111011110111
11000111100010001001010011100011001010010111110101010101100001001010100011011010010110100110001100101001011111010101010110000100101010001101101001011010011000110010100101111101010101011000010001

1100011110001000100101001110001001001100101101010101010011111011010101110001111010011101011000100100110010110101010101001111101101010111000111101001110101100010010011001011010101010100111110110101011100011110100111010111
1100011110001000100101001110001001001100101101010101010011111011010101110001111110011101011000100100110010110101010101001111101101010111000111101001110101100010010011001011010101010100111110110101011100011110100111010111

    '''.strip().split('\n')
    for raw in raw_data:
        raw = raw.strip()
        if not raw:
            continue
        print(f"[TEST]: Raw: {raw}")
        decoded = hot.decode(raw, debug=True)
        print(f"[TEST]: Decoded from raw: {decoded}")

        if decoded is not None:
            encoded = hot.encode(decoded, debug=True)
            print(f"[TEST]: Re-encoded: {encoded}")


if __name__ == "__main__":
    test()
