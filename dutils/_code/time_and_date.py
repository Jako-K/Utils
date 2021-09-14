"""
Description:
Make it more convenient to work with time and dates
"""

import time as _time
from datetime import timedelta as _timedelta
import numpy as _np
from . import type_check as _type_check


class StopWatch:
    """
    Function description:
    Keep track of time, much like a stop watch.

    Example:
    >> timer = StopWatch()
    >> timer.start()
    >> time.sleep(2)
    >> timer.stop()
    >> print(timer.get_elapsed_time())
    ~ 2.0
    """

    def __init__(self, time_unit:str="seconds", start_on_create:bool=False, precision_decimals:int=3):
        """
        @param time_unit: Determine which unit format time is displayed in, must be in `legal_units`
        @param start_on_create: If the stop watch should start the instance it's initialized
        @param precision_decimals: The amount of decimals time is displayed with.
        """

        # Checks
        self.legal_units = ["hour/min/sec", "seconds", "minutes", "hours"]
        _type_check.assert_types([time_unit, start_on_create, precision_decimals], [str, bool, int])
        _type_check.assert_comparison_number(precision_decimals, 0, ">=", "precision_decimals")

        self._start_time = None
        self._elapsed_time = None
        self._is_running = False
        self._unit = None; self.set_unit(time_unit)
        self.precision_decimals = precision_decimals
        if start_on_create:
            self.start()


    def __str__(self):
        return "StopWatch"


    def start(self):
        if self._start_time is not None:
            self.reset()

        self._start_time = _time.time()
        self._is_running = True


    def _calculate_elapsed_time(self):
        if self._start_time is None:
            return None
        else:
            return round(_time.time() - self._start_time, self.precision_decimals)


    def stop(self):
        if self._start_time is None:
            raise RuntimeError("Call `start()` before `stop()`")
        self._elapsed_time = self._calculate_elapsed_time()
        self._is_running = False


    def get_elapsed_time(self):
        current_time = self._calculate_elapsed_time() if self._is_running else self._elapsed_time

        if current_time is None:
            return 0
        elif self._unit == "seconds":
            return current_time
        elif self._unit == "minutes":
            return current_time / 60.0
        elif self._unit == "hours":
            return current_time / 3600.0
        elif self._unit == "hour/min/sec":
            return str(_timedelta(seconds=current_time)).split(".")[0] # the ugly part is just to remove milliseconds
        else:
            raise RuntimeError("Should not have gotten this far")


    def set_unit(self, time_unit:str):
        """
        Function description:
        set the unit used to display time

        @param time_unit: Determine which unit format time is displayed in, must be in `legal_units`
        """

        # Checks
        _type_check.assert_type(time_unit, str)
        _type_check.assert_in(time_unit, self.legal_units)

        self._unit = time_unit


    def reset(self):
        self._start_time = None
        self._elapsed_time = None
        self._is_running = False


class FPSTimer:
    """
    Keep track of frames per second.

    Example:
    >> fps_timer = FPSTimer()
    >> fps_timer.start()
    >>
    >> for i in range(3):
    >>    fps_timer.increment()
    >>    print(fps_timer.get_fps())
    >>    time.sleep(0.5)
    >>
    >> print( get_avg_fps() )

    ~ 0
    ~ 1.969
    ~ 1.98
    ~ 1.974
    """

    def __init__(self, precision_decimals:int=3):
        """ @param precision_decimals: The amount of decimals fps is displayed with. """
        # Checks
        _type_check.assert_type(precision_decimals, int)
        _type_check.assert_comparison_number(precision_decimals, 0, ">=", "precision_decimals")

        self._start_time = None
        self._elapsed_time = None
        self.ticks = []
        self.precision_decimals = precision_decimals


    def __str__(self):
        return "FPSTimer"


    def start(self):
        self._start_time = _time.time()


    def increment(self):
        if self._start_time is None:
            raise RuntimeError("Call `start()` before you call `increment()`")
        self.ticks.append( _time.time() - self._start_time )


    def get_frame_count(self):
        return len(self.ticks)


    def get_fps(self):
        if self._start_time is None:
            raise RuntimeError("Call `start()` before you call `get_fps()`")

        if len(self.ticks) < 2:
            fps = 0
        else:
            fps = 1 / (self.ticks[-1] - self.ticks[-2])
        return round(fps, self.precision_decimals)


    def reset(self):
        self._elapsed_time = None
        self.ticks = []


month_names = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
               'august', 'september', 'october', 'november', 'december']
month_names_abb = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


__all__ = [
    "StopWatch",
    "FPSTimer",
    "month_names",
    "month_names_abb"
]


StopWatch(time_unit = "hours", start_on_create= False, precision_decimals = 3)