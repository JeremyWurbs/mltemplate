"""Utility class for simple Timer and TimerCollection classes."""
import time
from typing import Dict, Optional


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
        self._duration: float = 0.0

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
        self._duration = 0.0

    def __str__(self):
        return f"{self.duration():.3f}s"


class TimerCollection:
    """Utility class for timing multiple operations.

    This class keeps a collection of named timers. Each timer can be started, stopped, and reset. The duration of each
    timer can be retrieved at any time. If a timer is stopped and restarted, the duration will be added to the previous
    duration. The timers can be reset individually, or all at once.

    Example::

        import time
        from mltemplate.utils import TimerCollection

        tc = TimerCollection()
        tc.start('Timer 1')
        tc.start('Timer 2')
        time.sleep(1)
        tc.stop('Timer 1')
        time.sleep(1)
        tc.stop('Timer 2')
        tc.start('Timer 3')
        time.sleep(1)
        tc.reset('Timer 1')
        print(tc)
            # Timer 1: 0.000s
            # Timer 2: 2.000s
            # Timer 3: 1.000s
        tc.reset_all()
        print(tc)
            # Timer 1: 0.000s
            # Timer 2: 0.000s
            # Timer 3: 0.000s

    """

    def __init__(self):
        self._timers: Dict[str, Timer] = {}

    def start(self, name: str):
        """Start the timer with the given name."""
        if name not in self._timers:
            self._timers[name] = Timer()
        self._timers[name].start()

    def stop(self, name: str):
        """Stop the timer with the given name."""
        if name not in self._timers:
            raise KeyError(f"Timer {name} does not exist. Unable to stop.")
        self._timers[name].stop()

    def duration(self, name: str) -> float:
        """Get the duration of the timer with the given name."""
        if name not in self._timers:
            raise KeyError(f"Timer {name} does not exist. Unable to get duration.")
        return self._timers[name].duration()

    def reset(self, name: str):
        """Reset the timer with the given name."""
        if name not in self._timers:
            raise KeyError(f"Timer {name} does not exist. Unable to reset.")
        self._timers[name].reset()

    def reset_all(self):
        """Reset all timers."""
        for timer in self._timers.values():
            timer.reset()

    def names(self):
        """Get the names of all timers."""
        return self._timers.keys()

    def __str__(self):
        """Print each timer to the nearest microsecond."""
        return "\n".join([f"{name}: {timer.duration():.6f}s" for name, timer in self._timers.items()])
