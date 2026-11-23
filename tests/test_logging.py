"""Whatever ends a CLI command lands in its log file: the traceback, or the stop signal."""

import logging
import signal

import pytest

from evotaste import worker


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker, "_STOP_SIGNAL", None)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    path = worker.setup_logging("r", "test")
    yield path
    for s, h in handlers.items():
        signal.signal(s, h)
    logging.basicConfig(force=True, handlers=[logging.NullHandler()])


def test_exception_traceback_is_logged(log_file):
    def crash():
        raise ValueError("boom in the middle of a step")

    with pytest.raises(ValueError):
        worker.logged(crash)
    text = log_file.read_text()
    assert "terminated by an exception" in text and "Traceback" in text and "ValueError: boom" in text


def test_stop_signal_is_logged_as_such(log_file):
    def stopped():
        worker._stop(signal.SIGTERM, None)

    with pytest.raises(SystemExit) as e:
        worker.logged(stopped)
    assert e.value.code == 128 + signal.SIGTERM
    text = log_file.read_text()
    assert "received SIGTERM" in text and "stopped by SIGTERM" in text and "Traceback" not in text


def test_refusal_is_logged(log_file):
    def refuse():
        raise SystemExit("runs/x is not prepared")

    with pytest.raises(SystemExit):
        worker.logged(refuse)
    assert "exiting: runs/x is not prepared" in log_file.read_text()
