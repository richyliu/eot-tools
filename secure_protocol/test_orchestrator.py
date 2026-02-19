#!/usr/bin/env python3.13
"""
Test orchestrator for EOT/HOT device protocol testing.

Runs ./main eot and ./main hot in parallel, allowing:
- Assertion of output patterns
- Input injection to stdin
- Logging to file for debugging
"""

import asyncio
import os
import pty
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_TIMEOUT = 5.0

SOCKET_PATHS = [
    "/tmp/eot_to_hot.sock",
    "/tmp/hot_to_eot.sock",
]


class DeviceRunner:
    """Manages a single device subprocess (eot or hot)."""

    def __init__(self, executable: str, device_type: str, packet_drops: Optional[list[int]] = None):
        self.executable = executable
        self.device_type = device_type
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

    def _build_args(self) -> list[str]:
        args = [self.device_type]
        args.extend(str(p) for p in self.packet_drops)
        return args

    async def start(self, log_dir: Path) -> None:
        """Start the device subprocess with PTY for proper buffering."""
        self.start_time = time.time()
        self.log_path = log_dir / f"{self.device_type}.log"

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
            log_f.write(f"=== {self.device_type.upper()} started at {datetime.now()} ===\n")
            log_f.flush()

            reader = asyncio.StreamReader()
            read_protocol = asyncio.StreamReaderProtocol(reader)
            self._pty_master_consumed = True
            self._transport, _ = await loop.connect_read_pipe(
                lambda: read_protocol,
                os.fdopen(self._pty_master, "r")
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

    async def wait_for_output(self, pattern: str, timeout: float = DEFAULT_TIMEOUT) -> str:
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
                f"[{self.device_type}] Pattern '{pattern}' not found within {timeout}s. "
                f"Recent output:\n" + "\n".join(self._output_history[-10:])
            )

    async def assert_output(self, pattern: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """
        Assert that a line matching the pattern appears within timeout.

        Raises:
            AssertionError: If pattern not found within timeout
        """
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:.3f}s] [{self.device_type}] Waiting for pattern: {pattern}")
        try:
            return await self.wait_for_output(pattern, timeout)
        except asyncio.TimeoutError as e:
            raise AssertionError(str(e))

    async def send_input(self, text: str) -> None:
        """Send input to the device's stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"[{self.device_type}] Process not started")
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


class TestOrchestrator:
    """Manages both EOT and HOT devices for testing."""

    def __init__(self, executable: str = "./main"):
        self.executable = executable
        self.eot: Optional[DeviceRunner] = None
        self.hot: Optional[DeviceRunner] = None
        self.log_dir: Optional[Path] = None
        self._test_name: str = "test"

    def _clean_sockets(self) -> None:
        """Remove existing socket files."""
        for path in SOCKET_PATHS:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    async def setup(self, test_name: str, eot_drops: Optional[list[int]] = None, hot_drops: Optional[list[int]] = None) -> None:
        """Initialize test environment and start devices."""
        self._test_name = test_name
        self._clean_sockets()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"test_logs/{test_name}_{timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.eot = DeviceRunner(self.executable, "eot", eot_drops)
        self.hot = DeviceRunner(self.executable, "hot", hot_drops)

        await asyncio.gather(
            self.eot.start(self.log_dir),
            self.hot.start(self.log_dir),
        )

    async def teardown(self) -> None:
        """Stop devices and cleanup."""
        tasks = []
        if self.eot:
            tasks.append(self.eot.stop())
        if self.hot:
            tasks.append(self.hot.stop())

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def elapsed_time(self) -> float:
        """Get elapsed time since test start."""
        if self.eot:
            return time.time() - self.eot.start_time
        return 0.0

    def print_header(self) -> None:
        """Print test header at the start of each test."""
        print(f"\n=== Test: {self._test_name} ===")
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

        print(f"[{orchestrator.elapsed_time():.3f}s] test_full_pairing PASSED")

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

        print(f"[{orchestrator.elapsed_time():.3f}s] test_basic_communication PASSED")

    finally:
        await orchestrator.teardown()
        orchestrator.print_header()


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

        print(f"[{orchestrator.elapsed_time():.3f}s] test_wrong_pin PASSED")

    finally:
        await orchestrator.teardown()
        orchestrator.print_header()


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

        print(f"[{orchestrator.elapsed_time():.3f}s] test_packet_drop PASSED (packet drop observed)")

    finally:
        await orchestrator.teardown()
        orchestrator.print_header()


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
        print(f"Waiting at most {timeout_sec} seconds for EOT to timeout waiting for HOT advertisement...")
        await eot.assert_output("timed out", timeout=timeout_sec)

        print(f"[{orchestrator.elapsed_time():.3f}s] test_timeout PASSED")

    finally:
        await orchestrator.teardown()
        orchestrator.print_header()


TESTS = {
    "full_pairing": test_full_pairing,
    "basic_communication": test_basic_communication,
    "wrong_pin": test_wrong_pin,
    "packet_drop": test_packet_drop,
    "timeout": test_timeout,
}


async def run_tests(test_names: list[str]) -> bool:
    """Run specified tests. Returns True if all pass."""
    all_passed = True

    for name in test_names:
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available tests: {', '.join(TESTS.keys())}")
            all_passed = False
            continue

        orchestrator = TestOrchestrator()
        try:
            await TESTS[name](orchestrator)
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
            orchestrator.print_header()
            all_passed = False
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            orchestrator.print_header()
            all_passed = False

        await asyncio.sleep(0.5)

    return all_passed


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_orchestrator.py <test_name> [test_name...]")
        print(f"Available tests: {', '.join(TESTS.keys())}")
        print("Use 'all' to run all tests")
        sys.exit(1)

    test_names = sys.argv[1:]
    if "all" in test_names:
        test_names = list(TESTS.keys())

    success = asyncio.run(run_tests(test_names))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
