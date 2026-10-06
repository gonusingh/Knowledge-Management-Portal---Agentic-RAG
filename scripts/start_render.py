import os
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

API_STARTUP_TIMEOUT_SECONDS = 180


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_for_api(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + API_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("FastAPI exited before becoming healthy.")
        try:
            with urlopen("http://127.0.0.1:8000/health", timeout=1):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(
        f"FastAPI did not become healthy within {API_STARTUP_TIMEOUT_SECONDS} seconds."
    )


def main() -> int:
    if "PORT" not in os.environ:
        raise RuntimeError("Render did not provide the required PORT variable.")

    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )
    ui_process: subprocess.Popen[bytes] | None = None

    def request_shutdown(_signum: int, _frame: object) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)

    try:
        wait_for_api(api_process)
        ui_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "ui/app.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                os.environ["PORT"],
            ]
        )

        while True:
            for process in (api_process, ui_process):
                return_code = process.poll()
                if return_code is not None:
                    return return_code
            time.sleep(0.5)
    finally:
        stop_process(ui_process)
        stop_process(api_process)


if __name__ == "__main__":
    raise SystemExit(main())
