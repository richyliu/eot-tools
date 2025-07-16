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

        print('[INFO] raw packet:', packet, file=sys.stderr)

        return packet
