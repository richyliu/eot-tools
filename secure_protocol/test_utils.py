import asyncio
import os
import pty
import re
import socket
import time
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

DEFAULT_TIMEOUT = 2.0


class BaseDeviceRunner:
    """Base class for device runners (Unix and QEMU)."""

    def __init__(self, device_name: str, mode: str = "default", log_callback: Optional[callable] = None):
        self._device_name = device_name
        self.mode = mode
        self.process: Optional[asyncio.subprocess.Process] = None
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self._output_history: List[str] = []
        self.log_path: Optional[Path] = None
        self.start_time: float = 0.0
        self.log_callback = log_callback or (lambda x: None)

    async def wait_for_output(self, pattern: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Wait for a line matching the pattern."""
        regex = re.compile(pattern, re.IGNORECASE)

        for line in self._output_history:
            if regex.search(line):
                return line

        try:
            async def _reader():
                while True:
                    line = await self._output_queue.get()
                    if regex.search(line):
                        return line
            return await asyncio.wait_for(_reader(), timeout=timeout)
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(
                f"[{self._device_name}] Pattern '{pattern}' not found within {timeout}s."
            )

    async def assert_output(self, pattern: str, timeout: float = DEFAULT_TIMEOUT) -> str:
        """Assert that a line matching the pattern appears within timeout."""
        elapsed = time.time() - self.start_time
        self.log_callback(f"[{elapsed:5.2f}s] [{self._device_name}] Waiting for pattern: {pattern}")
        try:
            return await self.wait_for_output(pattern, timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"[{self._device_name}] Pattern '{pattern}' not found within {timeout}s.")

    async def send_input(self, text: str) -> None:
        """Send input to the device's stdin."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"[{self._device_name}] Process not started")
        self.process.stdin.write(text.encode())
        await self.process.stdin.drain()

    def get_recent_output(self, n: int = 20) -> List[str]:
        """Get the last n lines of output."""
        return self._output_history[-n:]

    def _log_line(self, log_f, line: str):
        """Write a line to the log file with a timestamp."""
        elapsed = time.time() - self.start_time
        log_f.write(f"[{elapsed:.3f}s] {line}\n")
        log_f.flush()

class UnixDeviceRunner(BaseDeviceRunner):
    """Manages a single device subprocess (eot or hot) on Unix."""

    def __init__(self, executable: str, socket_paths: List[str], packet_drops: Optional[List[int]] = None, mode: str = "default", log_callback: Optional[callable] = None):
        super().__init__(Path(executable).stem, mode, log_callback)
        self.executable = executable
        self.socket_paths = socket_paths
        self.packet_drops = packet_drops or []
        self._pty_master: int = -1
        self._pty_master_consumed: bool = False
        self._transport: Optional[asyncio.BaseTransport] = None

    def _build_args(self) -> List[str]:
        args = [self.mode]
        args.extend(self.socket_paths)
        args.extend([str(p) for p in self.packet_drops])
        return args

    async def start(self, log_dir: Path) -> None:
        """Start the device subprocess with PTY."""
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
        await asyncio.sleep(0.1)

    async def _read_output_pty(self) -> None:
        """Read output from PTY master."""
        loop = asyncio.get_running_loop()
        if not self.log_path: return

        with open(self.log_path, "w") as log_f:
            log_f.write(f"=== {self._device_name.upper()} started at {datetime.now()} ===\n")
            log_f.flush()

            reader = asyncio.StreamReader()
            read_protocol = asyncio.StreamReaderProtocol(reader)
            self._pty_master_consumed = True
            try:
                self._transport, _ = await loop.connect_read_pipe(
                    lambda: read_protocol, os.fdopen(self._pty_master, "r")
                )
            except Exception:
                return

            while True:
                try:
                    line = await reader.readline()
                    if not line: break
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")
                    if decoded:
                        self._output_history.append(decoded)
                        if len(self._output_history) > 1000:
                            self._output_history = self._output_history[-500:]
                        await self._output_queue.put(decoded)
                        self._log_line(log_f, decoded)
                        elapsed = time.time() - self.start_time
                        self.log_callback(f"[{elapsed:5.2f}s] [target {self._device_name}] {decoded}")
                except Exception:
                    break
            if self._transport:
                self._transport.close()
                self._transport = None

    async def stop(self) -> None:
        """Terminate the device subprocess."""
        if self._reader_task:
            self._reader_task.cancel()
            try: await self._reader_task
            except asyncio.CancelledError: pass

        if self._transport:
            try: self._transport.close()
            except Exception: pass
            self._transport = None

        if self._pty_master >= 0 and not self._pty_master_consumed:
            try: os.close(self._pty_master)
            except OSError: pass
            self._pty_master = -1

        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                if self.process:
                    try: self.process.kill()
                    except ProcessLookupError: pass
                    await self.process.wait()

class QemuDeviceRunner(BaseDeviceRunner):
    """Manages a QEMU subprocess with UART communication."""

    def __init__(self, device_type: str, packet_drops: Optional[List[int]] = None, seed: Optional[int] = None, mode: str = "default", log_callback: Optional[callable] = None):
        super().__init__(device_type, mode, log_callback)
        self.packet_drops = packet_drops or []
        self.seed = seed
        self._uart_socket_dir: Optional[str] = None

    async def start(self, log_dir: Path) -> None:
        """Start the QEMU subprocess."""
        self.start_time = time.time()
        self.log_path = log_dir / f"{self._device_name}.log"

        args = ["./run_qemu.sh", f"--{self._device_name}"]
        self.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        self._reader_task = asyncio.create_task(self._read_output())

        for _ in range(100):
            if self._uart_socket_dir: break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError(f"Timeout waiting for {self._device_name} QEMU to provide UART socket info")

        await self._init_process()

    async def _init_process(self) -> None:
        """Initialize the QEMU process with RNG seed and packet drop arguments."""
        if not self.process or not self.process.stdin:
            raise RuntimeError(f"[{self._device_name}] Process not started")

        await self.wait_for_output("Select protocol mode")
        mode_val = {"default": "0", "test_profile": "1", "test_timing": "2", "legacy_only": "3"}.get(self.mode, "0")
        self.process.stdin.write(f"{mode_val}\n".encode())
        await self.process.stdin.drain()

        if self.mode in ("test_profile", "test_timing"): return

        await self.wait_for_output("Seed for RNG:")
        seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)
        self.process.stdin.write(f"{seed}\n".encode())
        await self.process.stdin.drain()

        for pkt_num in self.packet_drops:
            await self.wait_for_output("Enter packet number to drop")
            self.process.stdin.write(f"{pkt_num}\n".encode())
            await self.process.stdin.drain()

        await self.wait_for_output("Enter packet number to drop")
        self.process.stdin.write(b"-1\n")
        await self.process.stdin.drain()

    async def _read_output(self) -> None:
        """Read output from QEMU process stdout."""
        if not self.process or not self.process.stdout or not self.log_path: return

        with open(self.log_path, "w") as log_f:
            log_f.write(f"=== {self._device_name.upper()} QEMU started at {datetime.now()} ===\n")
            log_f.flush()

            while True:
                try:
                    line = await self.process.stdout.readline()
                    if not line: break
                    decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")

                    if decoded.startswith("UART_SOCKET_DIR="):
                        self._uart_socket_dir = decoded.split("=", 1)[1]
                        self._log_line(log_f, f"[INFO] UART socket dir: {self._uart_socket_dir}")
                        continue

                    if decoded:
                        self._output_history.append(decoded)
                        if len(self._output_history) > 1000:
                            self._output_history = self._output_history[-500:]
                        await self._output_queue.put(decoded)
                        self._log_line(log_f, decoded)
                        elapsed = time.time() - self.start_time
                        self.log_callback(f"[{elapsed:5.2f}s] [target {self._device_name}] {decoded}")
                except Exception:
                    break

    @property
    def uart_socket_path(self) -> Optional[str]:
        """Get the UART socket path."""
        if self._uart_socket_dir:
            return f"{self._uart_socket_dir}/{self._device_name}_uart.sock"
        return None

    async def stop(self) -> None:
        """Terminate the QEMU subprocess."""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                if self.process:
                    try: self.process.kill()
                    except ProcessLookupError: pass
                    await self.process.wait()

        if self._reader_task:
            self._reader_task.cancel()
            try: await self._reader_task
            except asyncio.CancelledError: pass

class UartBridge:
    """Bridges UART sockets between EOT and HOT QEMU instances."""
    def __init__(self, baud_rate: Optional[int] = None, log_callback: Optional[callable] = None):
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._eot_writer: Optional[asyncio.StreamWriter] = None
        self._hot_writer: Optional[asyncio.StreamWriter] = None
        self.baud_rate = baud_rate
        self.log_callback = log_callback or (lambda x: None)

    async def _bridge_loop(self, eot_reader: asyncio.StreamReader, hot_reader: asyncio.StreamReader) -> None:
        """Bidirectional forwarding between sockets."""
        await asyncio.gather(
            self._forward(eot_reader, self._hot_writer, "EOT->HOT"),
            self._forward(hot_reader, self._eot_writer, "HOT->EOT"),
        )

    async def start(self, eot_socket_path: str, hot_socket_path: str) -> None:
        """Start bridging between the two UART sockets."""
        eot_reader, self._eot_writer = await self._connect(eot_socket_path)
        hot_reader, self._hot_writer = await self._connect(hot_socket_path)
        self._running = True
        self._task = asyncio.create_task(self._bridge_loop(eot_reader, hot_reader))

    async def _connect(self, path: str):
        for _ in range(50):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(path)
                return await asyncio.open_unix_connection(sock=sock)
            except (ConnectionRefusedError, FileNotFoundError):
                await asyncio.sleep(0.1)
        raise RuntimeError(f"Could not connect to socket: {path}")

    async def _forward(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, name: str) -> None:
        try:
            while self._running:
                data = await reader.read(4096)
                if not data: break

                
                if self.baud_rate:
                    wire_bits = len(data) * 8
                    delay = wire_bits / self.baud_rate
                    await asyncio.sleep(delay)
                    self.log_callback(f"         [{name}] {wire_bits} bits ({delay:.3f}s) sets")

                writer.write(data)
                await writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_callback(f"[{name}] Bridge error: {e}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        for writer in [self._eot_writer, self._hot_writer]:
            if writer:
                writer.close()
                try: await writer.wait_closed()
                except Exception: pass


class TestOrchestrator:
    """Manages both EOT and HOT devices for testing."""

    def __init__(
        self,
        eot_bin: str = "./eot",
        hot_bin: str = "./hot",
        arm_mode: bool = False,
        seed: Optional[int] = None,
        eot_mode: str = "default",
        hot_mode: str = "default",
        baud_rate: Optional[int] = None,
        quiet: bool = True,
    ):
        self.eot_bin = eot_bin
        self.hot_bin = hot_bin
        self.arm_mode = arm_mode
        self.seed = seed
        self.eot_mode = eot_mode
        self.hot_mode = hot_mode
        self.baud_rate = baud_rate
        self.quiet = quiet
        self._uart_bridge = None
        self._log_messages: List[str] = []

    def log(self, message: str) -> None:
        """Log a message for this test run."""
        self._log_messages.append(message)
        if not self.quiet:
            print(message)
            sys.stdout.flush()

    def get_logs(self) -> List[str]:
        """Return all logged messages."""
        return self._log_messages

    def _prepare_sockets(self) -> List[str]:
        """Create a unique directory and return socket paths within it."""
        import tempfile
        import shutil

        # Create a unique directory within tmp_sockets
        os.makedirs("./tmp_sockets", exist_ok=True)
        self._temp_socket_dir = tempfile.mkdtemp(prefix="test_", dir="./tmp_sockets")
        
        socket_paths = [
            os.path.join(self._temp_socket_dir, "eot_to_hot.sock"),
            os.path.join(self._temp_socket_dir, "hot_to_eot.sock"),
        ]
        return socket_paths

    async def setup(
        self,
        test_name: str,
        eot_drops: Optional[List[int]] = None,
        hot_drops: Optional[List[int]] = None,
        eot_mode: Optional[str] = None,
        hot_mode: Optional[str] = None,
    ) -> None:
        """Initialize test environment and start devices."""
        self._test_name = test_name

        socket_paths = []
        if not self.arm_mode:
            socket_paths = self._prepare_sockets()

        timestamp = datetime.now().strftime("%H%M%S_%f")
        self.log_dir = Path(f"test_logs/{test_name}_{timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        e_mode = eot_mode or self.eot_mode
        h_mode = hot_mode or self.hot_mode

        if self.arm_mode:
            # If a base seed is provided, give EOT and HOT different but deterministic seeds
            eot_seed = self.seed if self.seed is None else self.seed
            hot_seed = self.seed if self.seed is None else self.seed + 1
            self.eot = QemuDeviceRunner("eot", eot_drops, seed=eot_seed, mode=e_mode, log_callback=self.log)
            self.hot = QemuDeviceRunner("hot", hot_drops, seed=hot_seed, mode=h_mode, log_callback=self.log)
        else:
            self.eot = UnixDeviceRunner(self.eot_bin, socket_paths, eot_drops, mode=e_mode, log_callback=self.log)
            self.hot = UnixDeviceRunner(self.hot_bin, socket_paths, hot_drops, mode=h_mode, log_callback=self.log)

        await asyncio.gather(
            self.eot.start(self.log_dir),
            self.hot.start(self.log_dir),
        )

        if self.arm_mode:
            await asyncio.sleep(1.0)
            eot_socket = getattr(self.eot, "uart_socket_path", None)
            hot_socket = getattr(self.hot, "uart_socket_path", None)
            if eot_socket and hot_socket:
                self._uart_bridge = UartBridge(baud_rate=self.baud_rate, log_callback=self.log)
                await self._uart_bridge.start(eot_socket, hot_socket)
                msg = f"UART bridge started: {eot_socket} <-> {hot_socket}"
                if self.baud_rate:
                    msg += f" (baud rate: {self.baud_rate})"
                self.log(msg)

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

        if not self.arm_mode and hasattr(self, "_temp_socket_dir"):
            import shutil
            try:
                shutil.rmtree(self._temp_socket_dir)
            except Exception:
                pass

    def elapsed_time(self) -> float:
        """Get elapsed time since test start."""
        if self.eot:
            return time.time() - self.eot.start_time
        return 0.0

    def print_early_header(self, name):
        """Print brief test header at the start of each test (to log)."""
        mode_str = " (ARM/QEMU)" if self.arm_mode else ""
        self.log(f"\n=== Test: {name}{mode_str} ===")

    def print_header(self) -> None:
        """Print test header at the start of each test (to log)."""
        if self.log_dir:
            self.log(f"Logs: {self.log_dir}")
        if self.eot:
            self.log(f"EOT log: {self.eot.log_path}")
        if self.hot:
            self.log(f"HOT log: {self.hot.log_path}")
