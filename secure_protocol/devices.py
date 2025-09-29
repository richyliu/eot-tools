#!/usr/bin/env python3

import asyncio
import secrets
import sys
from enum import Enum, auto
from hashlib import sha256

import ecdsa
from ecdsa import SigningKey, VerifyingKey, NIST256p


class EOTMsgType(Enum):
    PUBKEY = auto()
    NONCE = auto()

class HOTMsgType(Enum):
    ADV = auto()
    PUBKEY_AND_COMMIT = auto()
    NONCE = auto()


class BigPacketDevice:
    def __init__(self):
        pass

    def send(self, hot_id, msg_type, data):
        # TODO: implement send (broadcast)
        pass

    def recv(self, hot_id, msg_type, data):
        raise NotImplementedError


class Nonce:
    @staticmethod
    def generate():
        return secrets.token_bytes(16)


class PIN:
    @staticmethod
    def compute(pubkey1, pubkey2, nonce1, nonce2):
        hasher = sha256()
        hasher.update(pubkey1)
        hasher.update(pubkey2)
        hasher.update(nonce1)
        hasher.update(nonce2)
        digest = hasher.digest()
        pin_int = int.from_bytes(digest, 'big') % 100000
        return pin_int


class Commitment:
    @staticmethod
    def create(pubkey1, pubkey2, nonce):
        hasher = sha256()
        hasher.update(pubkey1)
        hasher.update(pubkey2)
        hasher.update(nonce)
        return hasher.digest()

    @staticmethod
    def verify(commitment, pubkey1, pubkey2, nonce):
        return commitment == Commitment.create(pubkey1, pubkey2, nonce)
        


class PubDSA:
    def __init__(self, key):
        self.key = VerifyingKey.from_string(key, curve=NIST256p, hashfunc=sha256)

    def verify(self, msg, sig):
        return self.key.verify(sig, msg)

    def serialize(self):
        return self.key.to_string('compressed')
    

class PubPrivDSA:
    def __init__(self):
        self.priv_key = SigningKey.generate(curve=NIST256p, hashfunc=sha256)
        self.pub_key = self.priv_key.get_verifying_key()

    def serialize_pub(self):
        return self.pub_key.to_string('compressed')

    @staticmethod
    def pubkey_len():
        return 33  # compressed key length for NIST256p


class EOTState(Enum):
    IDLE = auto()        # initial
    WAIT_ADV = auto()    # waiting for advertisement from HOT
    KEY_EX_1 = auto()    # sent our pubkey, waiting for theirs
    KEY_EX_2 = auto()    # received their pubkey and commitment, sending our nonce and waiting for theirs
    PIN_DISPLAY = auto() # received their nonce, displaying PIN and waiting for user to confirm



class EOTDevice:
    def __init__(self, big_packet_device):
        self._state = EOTState.IDLE
        self.big_packet_device = big_packet_device
        self.big_packet_device.register_recv_callback(self.recv)
        
        self.current_hot = None
        self.current_priv_dsa = None
        self.current_hot_pub_dsa = None
        self.current_hot_commitment = None
        self.our_nonce = None
        self.pin = None

    def reset(self):
        self.current_hot = None
        self.current_priv_dsa = None
        self.current_hot_pub_dsa = None
        self.current_hot_commitment = None
        self.our_nonce = None
        self.pin = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        print('EOT: transitioning from', self._state, 'to', value, file=sys.stderr)
        self._state = value

    def press_arm_button(self):
        if self.state == EOTState.IDLE:
            self.state = EOTState.WAIT_ADV
        else:
            print('[WARN] EOT: ignoring arm button press in state', self.state, file=sys.stderr)

    def send(self, hot_id, msg_type, data):
        self.big_packet_device.send(hot_id, msg_type, data)

    def recv(self, hot_id, msg_type, data):
        print('EOT: received message of type', msg_type, 'in state', self.state, f'(message: {data.hex()})', file=sys.stderr)
        if msg_type == HOTMsgType.ADV and self.state == EOTState.WAIT_ADV:
            self.state = EOTState.KEY_EX_1
            self.current_priv_dsa = PubPrivDSA()
            pubkey = self.current_priv_dsa.serialize_pub()
            self.send(self.current_hot, EOTMsgType.PUBKEY, pubkey)
            self.current_hot = hot_id
        elif msg_type == HOTMsgType.PUBKEY_AND_COMMIT and self.state == EOTState.KEY_EX_1:
            self.state = EOTState.KEY_EX_2
            self.current_hot_pub_dsa = PubDSA(data[:PubPrivDSA.pubkey_len()])
            self.current_hot_commitment = data[PubPrivDSA.pubkey_len():]
            self.our_nonce = Nonce.generate()
            self.send(hot_id, EOTMsgType.NONCE, self.our_nonce)
        elif msg_type == HOTMsgType.NONCE and self.state == EOTState.KEY_EX_2:
            their_nonce = data
            if Commitment.verify(self.current_hot_commitment,
                                 self.current_priv_dsa.serialize_pub(),
                                 self.current_hot_pub_dsa.serialize(),
                                 their_nonce):
                print('[INFO] EOT: commitment verified, displaying PIN', file=sys.stderr)
                self.pin = PIN.compute(
                    self.current_priv_dsa.serialize_pub(),
                    self.current_hot_pub_dsa.serialize(),
                    self.our_nonce,
                    their_nonce
                )
                self.state = EOTState.PIN_DISPLAY
            else:
                print('[ERROR] EOT: commitment verification failed', file=sys.stderr)
                self.state = EOTState.IDLE
        else:
            print('[WARN] EOT: ignoring message of type', msg_type, 'in state', self.state, file=sys.stderr)

    async def run(self):
        while True:
            if self.state == EOTState.IDLE:
                self.reset()
            elif self.state == EOTState.WAIT_ADV:
                pass
            elif self.state == EOTState.KEY_EX_1:
                pass
            elif self.state == EOTState.KEY_EX_2:
                pass
            elif self.state == EOTState.PIN_DISPLAY:
                print(f'[PIN] Enter this PIN on the HOT device: {self.pin:05d}', file=sys.stderr)
                await asyncio.sleep(2)  # simulate user reading the PIN
            else:
                print('[ERROR] EOT: invalid state', self.state, file=sys.stderr)
                self.state = EOTState.IDLE
            await asyncio.sleep(0.01)


class HOTState(Enum):
    IDLE = auto()         # initial
    ADV = auto()          # sending periodic advertisements
    KEY_EX_1 = auto()     # sent our pubkey and commitment, waiting for their nonce
    KEY_EX_2 = auto()     # received their nonce, sending our nonce
    WAIT_FOR_PIN = auto() # sent our nonce, waiting for user to input PIN


class HOTDevice:
    def __init__(self, big_packet_device):
        self._state = HOTState.IDLE
        self.my_id = 1  # Example ID
        self.big_packet_device = big_packet_device
        self.big_packet_device.register_recv_callback(self.recv)
        self.current_priv_dsa = None
        self.current_eot_pub_dsa = None
        self.our_nonce = None
        self.current_eot_nonce = None
        self.expected_pin = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        print('HOT: transitioning from', self._state, 'to', value, file=sys.stderr)
        self._state = value

    def press_arm_button(self):
        if self.state == HOTState.IDLE:
            self.state = HOTState.ADV
        else:
            print('[WARN] HOT: ignoring arm button press in state', self.state, file=sys.stderr)

    def input_code(self, code):
        # simulate inputting the 5-digit code
        pass

    def send(self, hot_id, msg_type, data):
        self.big_packet_device.send(hot_id, msg_type, data)

    def recv(self, hot_id, msg_type, data):
        print('HOT: received message of type', msg_type, 'in state', self.state, f'(message: {data.hex()})', file=sys.stderr)
        if msg_type == EOTMsgType.PUBKEY and self.state == HOTState.ADV:
            self.state = HOTState.KEY_EX_1
            self.current_eot_pub_dsa = PubDSA(data)
            self.current_priv_dsa = PubPrivDSA()
            pubkey = self.current_priv_dsa.serialize_pub()
            self.our_nonce = Nonce.generate()
            commitment = Commitment.create(
                self.current_eot_pub_dsa.serialize(),
                self.current_priv_dsa.serialize_pub(),
                self.our_nonce
            )
            self.send(hot_id, HOTMsgType.PUBKEY_AND_COMMIT, pubkey + commitment)
        elif msg_type == EOTMsgType.NONCE and self.state == HOTState.KEY_EX_1:
            self.state = HOTState.WAIT_FOR_PIN
            self.current_eot_nonce = data
            self.expected_pin = PIN.compute(
                self.current_eot_pub_dsa.serialize(),
                self.current_priv_dsa.serialize_pub(),
                self.current_eot_nonce,
                self.our_nonce
            )
            self.send(hot_id, HOTMsgType.NONCE, self.our_nonce)
        else:
            print('[WARN] HOT: ignoring message of type', msg_type, 'in state', self.state, file=sys.stderr)

    async def run(self):
        while True:
            if self.state == HOTState.IDLE:
                pass
            elif self.state == HOTState.ADV:
                self.send(self.my_id, HOTMsgType.ADV, b'')
                await asyncio.sleep(1)
            elif self.state == HOTState.KEY_EX_1:
                pass
            elif self.state == HOTState.KEY_EX_2:
                pass
            elif self.state == HOTState.WAIT_FOR_PIN:
                print(f'[INFO] HOT: waiting for PIN input (expected PIN: {self.expected_pin:05d})', file=sys.stderr)
                await asyncio.sleep(1)
            else:
                print('[ERROR] HOT: invalid state', self.state, file=sys.stderr)
                self.state = HOTState.IDLE
            await asyncio.sleep(0.01)


class BigPacketDevice:
    def __init__(self):
        self.recv_callback = None
        self.upstream_send_fn = None

    def register_recv_callback(self, callback):
        self.recv_callback = callback

    def send(self, hot_id, msg_type, data):
        self.upstream_send_fn(hot_id, msg_type, data)

    def recv(self, hot_id, msg_type, data):
        if self.recv_callback:
            self.recv_callback(hot_id, msg_type, data)


async def main():
    device_a = BigPacketDevice()
    device_b = BigPacketDevice()

    # Link the devices
    device_a.upstream_send_fn = device_b.recv
    device_b.upstream_send_fn = device_a.recv

    eot = EOTDevice(device_a)
    hot = HOTDevice(device_b)

    eot.press_arm_button()
    hot.press_arm_button()

    await asyncio.gather(eot.run(), hot.run())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('User pressed Ctrl-C, exiting...')
