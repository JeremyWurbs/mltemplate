"""Utility class for a TimerCollection object."""
from typing import Dict

from mltemplate.utils import Timer


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
        for name, _ in self._timers.items():
            timer = self._timers[name]
            timer.reset()

    def names(self):
        """Get the names of all timers."""
        return self._timers.keys()

    def __str__(self):
        """Print each timer to the nearest microsecond."""
        return "\n".join([f"{name}: {self._timers[name].duration():.6f}s" for name, _ in self._timers.items()])
