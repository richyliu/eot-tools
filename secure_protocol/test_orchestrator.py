#!/usr/bin/env python3.13
"""
Test orchestrator for EOT/HOT device protocol testing.

Runs ./eot and ./hot in parallel, allowing:
- Assertion of output patterns
- Input injection to stdin
- Logging to file for debugging
- ARM/QEMU testing with UART socket bridging
"""

import asyncio
import os
import pty
import re
import socket
import sys
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


DEFAULT_TIMEOUT = 5.0

SOCKET_PATHS = [
    "/tmp/eot_to_hot.sock",
    "/tmp/hot_to_eot.sock",
]


class DeviceRunner:
    """Manages a single device subprocess (eot or hot)."""

    def __init__(self, executable: str, packet_drops: Optional[list[int]] = None):
        self.executable = executable
        self.packet_drops = packet_drops or []
        self.process: Optional[asyncio.subprocess.Process] = None
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self._output_history: list[str] = []
        self.log_path: Optional[Path] = None
        self._pty_master: int = -1
        self._pty_master_consumed: bool = False
        self._transport: Optional[asyncio.BaseTransport] = None
        self.start_time: float = 0.0
        self._device_name: str = Path(executable).stem

    def _build_args(self) -> list[str]:
        return [str(p) for p in self.packet_drops]

    async def start(self, log_dir: Path) -> None:
        """Start the device subprocess with PTY for proper buffering."""
        self.start_time = time.time()
        self.log_path = log_dir / f"{self._device_name}.log"

        args = self._build_args()

        master_fd, slave_fd = pty.openpty()
        self._pty_master = master_fd

        self.process = await asyncio.create_subprocess_exec(
            self.executable,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)

        self._reader_task = asyncio.create_task(self._read_output_pty())
        await asyncio.sleep(0.2)

    async def _read_output_pty(self) -> None:
        """Read output from PTY master."""
        loop = asyncio.get_running_loop()

        log_path = self.log_path
        if log_path is None:
            return

        with open(log_path, "w") as log_f:
            log_f.write(
                f"=== {self._device_name.upper()} started at {datetime.now()} ===\n"
            )
            log_f.flush()

            reader = asyncio.StreamReader()
            read_protocol = asyncio.StreamReaderProtocol(reader)
            self._pty_master_consumed = True
            self._transport, _ = await loop.connect_read_pipe(
                lambda: read_protocol, os.fdopen(self._pty_master, "r")
            )

            while True:
                try:
                    line = await reader.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if decoded:
                        self._output_history.append(decoded)
                        if len(self._output_history) > 1000:
                            self._output_history = self._output_history[-500:]
                        await self._output_queue.put(decoded)
                        log_f.write(decoded + "\n")
                        log_f.flush()
                except Exception:
                    break
            if self._transport:
                self._transport.close()
                self._transport = None

    async def wait_for_output(
        self, pattern: str, timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        """
        Wait for a line matching the pattern.

        Args:
            pattern: Regex pattern to match
            timeout: Maximum seconds to wait

        Returns:
            The matched line

        Raises:
            asyncio.TimeoutError: If pattern not found within timeout
        """
        regex = re.compile(pattern, re.IGNORECASE)

        for line in self._output_history:
            if regex.search(line):
                return line

        try:
            async with asyncio.timeout(timeout):
                while True:
                    line = await self._output_queue.get()
                    if regex.search(line):
                        return line
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"[{self._device_name}] Pattern '{pattern}' not found within {timeout}s."
            )

    async def assert_output(
        self, pattern: str, timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        """
        Assert that a line matching the pattern appears within timeout.

        Raises:
            AssertionError: If pattern not found within timeout
        """
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.2f}s] [{self._device_name}] Waiting for pattern: {pattern}")
        try:
            return await self.wait_for_output(pattern, timeout)
        except asyncio.TimeoutError as e:
            raise AssertionError(str(e))

    async def send_input(self, text: str) -> None:
        """Send input to the device's stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"[{self._device_name}] Process not started")
        self.process.stdin.write(text.encode())
        await self.process.stdin.drain()

    async def stop(self) -> None:
        """Terminate the device subprocess."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._transport:
            try:
                self._transport.close()
            except Exception:
                pass
            self._transport = None

        if self._pty_master >= 0 and not self._pty_master_consumed:
            try:
                os.close(self._pty_master)
            except OSError:
                pass
            self._pty_master = -1

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.process.terminate()
                await self.process.wait()
            except ProcessLookupError:
                pass

    def get_recent_output(self, n: int = 20) -> list[str]:
        """Get the last n lines of output."""
        return self._output_history[-n:]


class UartBridge:
    """Bridges UART sockets between EOT and HOT QEMU instances."""

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._eot_reader: Optional[asyncio.StreamReader] = None
        self._eot_writer: Optional[asyncio.StreamWriter] = None
        self._hot_reader: Optional[asyncio.StreamReader] = None
        self._hot_writer: Optional[asyncio.StreamWriter] = None

    async def start(self, eot_socket_path: str, hot_socket_path: str) -> None:
        """Start bridging between the two UART sockets."""
        loop = asyncio.get_running_loop()

        for _ in range(50):
            try:
                eot_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                eot_sock.connect(eot_socket_path)
                break
            except (ConnectionRefusedError, FileNotFoundError):
                await asyncio.sleep(0.1)
        else:
            raise RuntimeError(
                f"Could not connect to EOT UART socket: {eot_socket_path}"
            )

        for _ in range(50):
            try:
                hot_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                hot_sock.connect(hot_socket_path)
                break
            except (ConnectionRefusedError, FileNotFoundError):
                await asyncio.sleep(0.1)
        else:
            raise RuntimeError(
                f"Could not connect to HOT UART socket: {hot_socket_path}"
            )

        self._eot_reader, self._eot_writer = await asyncio.open_unix_connection(
            sock=eot_sock
        )
        self._hot_reader, self._hot_writer = await asyncio.open_unix_connection(
            sock=hot_sock
        )

        self._running = True
        self._task = asyncio.create_task(self._bridge_loop())

    async def _forward(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Forward data from reader to writer."""
        try:
            while self._running:
                data = await reader.read(4096)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _bridge_loop(self) -> None:
        """Bidirectional forwarding between sockets."""
        if (
            not self._eot_reader
            or not self._hot_reader
            or not self._eot_writer
            or not self._hot_writer
        ):
            return

        eot_reader = self._eot_reader
        eot_writer = self._eot_writer
        hot_reader = self._hot_reader
        hot_writer = self._hot_writer

        await asyncio.gather(
            self._forward(eot_reader, hot_writer),
            self._forward(hot_reader, eot_writer),
        )

    async def stop(self) -> None:
        """Stop the bridge."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._eot_writer:
            self._eot_writer.close()
            try:
                await self._eot_writer.wait_closed()
            except Exception:
                pass
        if self._hot_writer:
            self._hot_writer.close()
            try:
                await self._hot_writer.wait_closed()
            except Exception:
                pass


class QemuDeviceRunner:
    """Manages a QEMU subprocess with UART communication."""

    def __init__(self, device_type: str, packet_drops: Optional[list[int]] = None):
        self.device_type = device_type
        self.packet_drops = packet_drops or []
        self.process: Optional[asyncio.subprocess.Process] = None
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self._output_history: list[str] = []
        self.log_path: Optional[Path] = None
        self._uart_socket_dir: Optional[str] = None
        self.start_time: float = 0.0

    async def start(self, log_dir: Path) -> None:
        """Start the QEMU subprocess."""
        self.start_time = time.time()
        self.log_path = log_dir / f"{self.device_type}.log"

        args = ["./run_qemu.sh", f"--{self.device_type}"]

        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

        self._reader_task = asyncio.create_task(self._read_output())

        for _ in range(100):
            if self._uart_socket_dir:
                break
            await asyncio.sleep(0.05)

    async def _read_output(self) -> None:
        """Read output from QEMU process stdout."""
        if not self.process or not self.process.stdout:
            return

        log_path = self.log_path
        if log_path is None:
            return

        with open(log_path, "w") as log_f:
            log_f.write(
                f"=== {self.device_type.upper()} QEMU started at {datetime.now()} ===\n"
            )
            log_f.flush()

            while True:
                try:
                    line = await self.process.stdout.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")

                    if decoded.startswith("UART_SOCKET_DIR="):
                        self._uart_socket_dir = decoded.split("=", 1)[1]
                        log_f.write(
                            f"[INFO] UART socket dir: {self._uart_socket_dir}\n"
                        )
                        log_f.flush()
                        continue

                    if decoded:
                        self._output_history.append(decoded)
                        if len(self._output_history) > 1000:
                            self._output_history = self._output_history[-500:]
                        await self._output_queue.put(decoded)
                        log_f.write(decoded + "\n")
                        log_f.flush()
                except Exception:
                    break

    @property
    def uart_socket_path(self) -> Optional[str]:
        """Get the UART socket path."""
        if self._uart_socket_dir:
            return f"{self._uart_socket_dir}/{self.device_type}_uart.sock"
        return None

    async def wait_for_output(
        self, pattern: str, timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        """Wait for a line matching the pattern."""
        regex = re.compile(pattern, re.IGNORECASE)

        for line in self._output_history:
            if regex.search(line):
                return line

        try:
            async with asyncio.timeout(timeout):
                while True:
                    line = await self._output_queue.get()
                    if regex.search(line):
                        return line
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"[{self.device_type}] Pattern '{pattern}' not found within {timeout}s."
            )

    async def assert_output(
        self, pattern: str, timeout: float = DEFAULT_TIMEOUT
    ) -> str:
        """Assert that a line matching the pattern appears within timeout."""
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.2f}s] [{self.device_type}] Waiting for pattern: {pattern}")
        try:
            return await self.wait_for_output(pattern, timeout)
        except asyncio.TimeoutError as e:
            raise AssertionError(str(e))

    async def send_input(self, text: str) -> None:
        """Send input via stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"[{self.device_type}] Process not started")
        self.process.stdin.write(text.encode())
        await self.process.stdin.drain()

    async def stop(self) -> None:
        """Terminate the QEMU subprocess."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                await self.process.wait()
            except ProcessLookupError:
                print(f"[{self.device_type}] Process already terminated")

    def get_recent_output(self, n: int = 20) -> list[str]:
        """Get the last n lines of output."""
        return self._output_history[-n:]


class TestOrchestrator:
    """Manages both EOT and HOT devices for testing."""

    def __init__(
        self, eot_bin: str = "./eot", hot_bin: str = "./hot", arm_mode: bool = False
    ):
        self.eot_bin = eot_bin
        self.hot_bin = hot_bin
        self.arm_mode = arm_mode
        self.eot: Union[DeviceRunner, QemuDeviceRunner, None] = None
        self.hot: Union[DeviceRunner, QemuDeviceRunner, None] = None
        self.log_dir: Optional[Path] = None
        self._test_name: str = "test"
        self._uart_bridge: Optional[UartBridge] = None

    def _clean_sockets(self) -> None:
        """Remove existing socket files."""
        for path in SOCKET_PATHS:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    async def setup(
        self,
        test_name: str,
        eot_drops: Optional[list[int]] = None,
        hot_drops: Optional[list[int]] = None,
    ) -> None:
        """Initialize test environment and start devices."""
        self._test_name = test_name

        if not self.arm_mode:
            self._clean_sockets()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"test_logs/{test_name}_{timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.arm_mode:
            self.eot = QemuDeviceRunner("eot", eot_drops)
            self.hot = QemuDeviceRunner("hot", hot_drops)
        else:
            self.eot = DeviceRunner(self.eot_bin, eot_drops)
            self.hot = DeviceRunner(self.hot_bin, hot_drops)

        await asyncio.gather(
            self.eot.start(self.log_dir),
            self.hot.start(self.log_dir),
        )

        if self.arm_mode:
            await asyncio.sleep(1.0)
            eot_socket = getattr(self.eot, "uart_socket_path", None)
            hot_socket = getattr(self.hot, "uart_socket_path", None)
            if eot_socket and hot_socket:
                self._uart_bridge = UartBridge()
                await self._uart_bridge.start(eot_socket, hot_socket)
                print(f"UART bridge started: {eot_socket} <-> {hot_socket}")

    async def teardown(self) -> None:
        """Stop devices and cleanup."""
        tasks = []
        if self._uart_bridge:
            tasks.append(self._uart_bridge.stop())
        if self.eot:
            tasks.append(self.eot.stop())
        if self.hot:
            tasks.append(self.hot.stop())

        if tasks:
            await asyncio.gather(*tasks)

    def elapsed_time(self) -> float:
        """Get elapsed time since test start."""
        if self.eot:
            return time.time() - self.eot.start_time
        return 0.0

    def print_header(self) -> None:
        """Print test header at the start of each test."""
        mode_str = " (ARM/QEMU)" if self.arm_mode else ""
        print(f"\n=== Test: {self._test_name}{mode_str} ===")
        if self.log_dir:
            print(f"Logs: {self.log_dir}")
        if self.eot:
            print(f"EOT log: {self.eot.log_path}")
        if self.hot:
            print(f"HOT log: {self.hot.log_path}")


async def test_full_pairing(orchestrator: TestOrchestrator) -> None:
    """Test complete pairing flow from idle to paired state."""
    await orchestrator.setup("test_full_pairing")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await eot.assert_output("waiting for HOT advertisement")

        await hot.send_input("-1\n")
        await hot.assert_output("sending advertisement")

        await eot.assert_output("received advertisement")
        await eot.assert_output("sent public key")

        await hot.assert_output("received EOT pubkey")
        await hot.assert_output("sent pubkey and commitment")

        await eot.assert_output("received HOT pubkey and commitment")
        await eot.assert_output("sent nonce")

        await hot.assert_output("received EOT nonce")
        await hot.assert_output("sent nonce to EOT")

        eot_pin_line = await eot.assert_output(r"PIN is \d{5}")
        eot_pin_match = re.search(r"PIN is (\d{5})", eot_pin_line)
        assert eot_pin_match, f"Could not extract PIN from: {eot_pin_line}"
        pin = eot_pin_match.group(1)
        print(f"EOT PIN: {pin}")

        hot_pin_line = await hot.assert_output(r"expected PIN is \d{5}")
        hot_pin_match = re.search(r"expected PIN is (\d{5})", hot_pin_line)
        assert hot_pin_match, f"Could not extract PIN from: {hot_pin_line}"
        hot_pin = hot_pin_match.group(1)
        print(f"HOT expected PIN: {hot_pin}")

        assert pin == hot_pin, f"PIN mismatch: EOT={pin}, HOT={hot_pin}"

        await hot.send_input(f"{pin}\n")
        await hot.assert_output("PIN correct")

        await eot.send_input("\n")
        await eot.assert_output("Pairing successful")

        print(f"[{orchestrator.elapsed_time():.2f}s] test_full_pairing PASSED")

    finally:
        await orchestrator.teardown()


async def test_basic_communication(orchestrator: TestOrchestrator) -> None:
    """Test pairing + status request + emergency brake."""
    await orchestrator.setup("test_basic_communication")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await hot.send_input("-1\n")

        eot_pin_line = await eot.assert_output(r"PIN is (\d{5})")
        pin_match = re.search(r"PIN is (\d{5})", eot_pin_line)
        pin = pin_match.group(1) if pin_match else None

        await hot.assert_output(r"expected PIN is \d{5}")

        if pin:
            await hot.send_input(f"{pin}\n")
            await hot.assert_output("PIN correct")

        await eot.send_input("\n")
        await eot.assert_output("Pairing successful")

        await hot.assert_output("Select an option")

        await hot.send_input("1\n")
        await hot.assert_output("sent status update request")
        await eot.assert_output("sent status update to HOT")
        await hot.assert_output("EOT Status:")

        await hot.assert_output("Select an option")

        await hot.send_input("2\n")
        await hot.assert_output("sent emergency brake request")
        await eot.assert_output("Emergency brake activated")
        await hot.assert_output("received emergency brake confirmation")

        print(f"[{orchestrator.elapsed_time():.2f}s] test_basic_communication PASSED")

    finally:
        await orchestrator.teardown()


async def test_wrong_pin(orchestrator: TestOrchestrator) -> None:
    """Test that wrong PIN entry fails and returns to idle."""
    await orchestrator.setup("test_wrong_pin")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await hot.send_input("-1\n")

        await hot.assert_output(r"expected PIN is (\d{5})", timeout=15)

        for attempt in range(3):
            await hot.send_input("00000\n")
            await asyncio.sleep(0.5)

        await hot.assert_output("Failed to enter correct PIN")
        await hot.assert_output("HOT_IDLE")

        print(f"[{orchestrator.elapsed_time():.2f}s] test_wrong_pin PASSED")

    finally:
        await orchestrator.teardown()


async def test_packet_drop(orchestrator: TestOrchestrator) -> None:
    """Test that packet drops cause timeout and recovery."""
    await orchestrator.setup("test_packet_drop", eot_drops=[1])
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("Will drop packet 1")

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")
        await hot.send_input("-1\n")

        await hot.assert_output("sending advertisement")

        await eot.assert_output("dropping packet 1 for testing")

        print(
            f"[{orchestrator.elapsed_time():.2f}s] test_packet_drop PASSED (packet drop observed)"
        )

    finally:
        await orchestrator.teardown()


async def test_timeout(orchestrator: TestOrchestrator) -> None:
    """Test that pairing timeout returns devices to idle."""
    await orchestrator.setup("test_timeout")
    orchestrator.print_header()
    assert orchestrator.eot is not None
    assert orchestrator.hot is not None

    try:
        eot = orchestrator.eot
        hot = orchestrator.hot

        await eot.assert_output("EOT_IDLE")
        await hot.assert_output("HOT_IDLE")

        await eot.send_input("1\n")

        timeout_sec = 35
        print(
            f"Waiting at most {timeout_sec} seconds for EOT to timeout waiting for HOT advertisement..."
        )
        await eot.assert_output("timed out", timeout=timeout_sec)

        print(f"[{orchestrator.elapsed_time():.2f}s] test_timeout PASSED")

    finally:
        await orchestrator.teardown()


TESTS = {
    "full_pairing": test_full_pairing,
    "basic_communication": test_basic_communication,
    "wrong_pin": test_wrong_pin,
    "packet_drop": test_packet_drop,
    "timeout": test_timeout,
}


async def run_tests(test_names: list[str], arm_mode: bool = False) -> bool:
    """Run specified tests. Returns True if all pass."""
    tests_status = []

    for name in test_names:
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available tests: {', '.join(TESTS.keys())}")
            tests_status.append((name, "SKIPPED"))
            continue

        orchestrator = TestOrchestrator(arm_mode=arm_mode)
        try:
            await TESTS[name](orchestrator)
            tests_status.append((name, "PASSED"))
        except AssertionError as e:
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            if orchestrator.eot:
                print("  EOT recent output:")
                for line in orchestrator.eot.get_recent_output(5):
                    print(f"    {line}")
            if orchestrator.hot:
                print("  HOT recent output:")
                for line in orchestrator.hot.get_recent_output(5):
                    print(f"    {line}")
            tests_status.append((name, "FAILED"))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            tests_status.append((name, "ERROR"))

        await asyncio.sleep(0.2)

    num_passed = sum(1 for _, status in tests_status if status == "PASSED")
    num_failed = sum(1 for _, status in tests_status if status == "FAILED")
    num_other = sum(
        1 for _, status in tests_status if status not in ("PASSED", "FAILED")
    )
    num_total = len(tests_status)

    print("\n=== Test Summary ===")
    print(
        f"Total: {num_total}, Passed: {num_passed}, Failed: {num_failed}, Other: {num_other}"
    )

    all_passed = num_passed == num_total

    if not all_passed:
        print("\nFailed/Errored tests:")
        for name, status in tests_status:
            if status != "PASSED":
                print(f"  {name}: {status}")

    return all_passed


def main():
    arm_mode = False
    args = sys.argv[1:]

    if "--arm" in args:
        arm_mode = True
        args.remove("--arm")

    if not args:
        print("Usage: python test_orchestrator.py [--arm] <test_name> [test_name...]")
        print(f"Available tests: {', '.join(TESTS.keys())}")
        print("Use 'all' to run all tests")
        print("Use --arm to run on QEMU/ARM instead of native")
        sys.exit(1)

    test_names = args
    if "all" in test_names:
        test_names = list(TESTS.keys())

    success = asyncio.run(run_tests(test_names, arm_mode=arm_mode))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
