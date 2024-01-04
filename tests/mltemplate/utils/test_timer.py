"""Unit test methods for mltemplate.utils.timer utility module."""
import time

from mltemplate.utils import Timer


def test_timer():
    """Test the Timer class."""
    timer = Timer()

    # Test that the timer can be started
    timer.start()
    assert timer._start_time is not None
    assert timer._stop_time is None

    # Test that the timer can be stopped
    time.sleep(0.1)
    timer.stop()
    assert timer._start_time is not None
    assert timer._stop_time is not None
    assert timer.duration() > 0.1

    # Test that the timer can be reset
    timer.reset()
    assert timer._start_time is None
    assert timer._stop_time is None
    assert timer.duration() == 0.0

    # Test that the timer can be started and stopped multiple times
    timer.start()
    time.sleep(0.1)
    assert timer.duration() > 0.1
    timer.stop()
    time.sleep(1.0)
    assert timer.duration() < 0.2
    timer.start()
    time.sleep(0.1)
    timer.stop()
    assert timer._start_time is not None
    assert timer._stop_time is not None
    assert timer.duration() > 0.2
    assert timer.duration() < 1.0

    # Test that the timer can be converted to a string
    assert isinstance(str(timer), str)
