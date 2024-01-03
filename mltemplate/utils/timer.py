"""Utility class for a simple Timer object."""
import time
from typing import Optional


class Timer:
    """Utility timer class.

    This class can be used to time operations. It can be started, stopped, and reset. The duration of the timer can be
    retrieved at any time.

    Example::

        import time
        from mltemplate.utils import Timer

        timer = Timer()
        timer.start()
        time.sleep(1)
        timer.stop()
        print(f'The timer ran for {timer.duration()} seconds.')  # The timer ran for 1.0000000000000002 seconds.
        timer.reset()
        timer.start()
        time.sleep(2)
        timer.stop()
        print(f'The timer ran for {timer.duration()} seconds.')  # The timer ran for 2.0000000000000004 seconds.
    """
    def __init__(self):
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._duration: float = 0.

    def start(self):
        """Start the timer."""
        self._start_time = time.time()
        self._stop_time = None

    def stop(self):
        """Stop the timer."""
        if self._stop_time is None:
            self._stop_time = time.time()
            self._duration += self._stop_time - self._start_time

    def duration(self) -> float:
        """Get the duration of the timer."""
        if self._start_time is not None and self._stop_time is None:
            return self._duration + (time.time() - self._start_time)
        else:
            return self._duration

    def reset(self):
        """Reset the timer."""
        self._start_time = None
        self._stop_time = None
        self._duration = 0.

    def __str__(self):
        return f'{self.duration():.3f}s'
