"""
Description:
Make it more convenient to work with time and dates
"""

import time
from datetime import timedelta
import numpy as np
import type_check


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
        type_check.assert_types([time_unit, start_on_create, precision_decimals], [str, bool, int])
        type_check.assert_comparison_number(precision_decimals, 0, ">=", "precision_decimals")

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

        self._start_time = time.time()
        self._is_running = True


    def _calculate_elapsed_time(self):
        if self._start_time is None:
            return None
        else:
            return round(time.time() - self._start_time, self.precision_decimals)


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
            return str(timedelta(seconds=current_time)).split(".")[0] # the ugly part is just to remove milliseconds
        else:
            raise RuntimeError("Should not have gotten this far")


    def set_unit(self, time_unit:str):
        """
        Function description:
        set the unit used to display time

        @param time_unit: Determine which unit format time is displayed in, must be in `legal_units`
        """

        # Checks
        type_check.assert_type(time_unit, str)
        type_check.assert_in(time_unit, self.legal_units)

        self._unit = time_unit


    def reset(self):
        self._start_time = None
        self._elapsed_time = None
        self._is_running = False


class FPSTimer:
    """
    Function description:
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
        type_check.assert_type(precision_decimals, int)
        type_check.assert_comparison_number(precision_decimals, 0, ">=", "precision_decimals")

        self._start_time = None
        self._elapsed_time = None
        self.ticks = []
        self.precision_decimals = precision_decimals


    def __str__(self):
        return "FPSTimer"


    def start(self):
        self._start_time = time.time()


    def _get_elapsed_time(self):
        return round(time.time() - self._start_time, self.precision_decimals)


    def increment(self):
        if self._start_time is None:
            raise RuntimeError("Call `start()` before you call `increment()`")
        self.ticks.append(self._get_elapsed_time())


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


    def get_avg_fps(self):
        fps = np.mean([1 / (self.ticks[i] - self.ticks[i - 1]) for i in range(1, len(self.ticks))])
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


# TODO Move this to a test folder
# TODO Figure out how to check FPSTimer and  StopWatch in a better way
import unittest
class UnitTests(unittest.TestCase):


    def test_stop_watch(self):
        # Init checks
        with self.assertRaises(TypeError):
            StopWatch(time_unit = "seconds", start_on_create= None, precision_decimals = 3)
        with self.assertRaises(TypeError):
            StopWatch(time_unit = "seconds", start_on_create= False, precision_decimals = 3.0)
        with self.assertRaises(TypeError):
            StopWatch(time_unit = None, start_on_create= False, precision_decimals = 3)
        with self.assertRaises(ValueError):
            StopWatch(time_unit = "hours", start_on_create= False, precision_decimals = 3)
        with self.assertRaises(ValueError):
            StopWatch(time_unit = "seconds", start_on_create= False, precision_decimals = -1)

        for value in StopWatch().legal_units:
            StopWatch(time_unit=value)

        # Check all functions
        timer = StopWatch(precision_decimals=0)
        self.assertRaises(RuntimeError, timer.stop)
        timer.start()
        timer.get_elapsed_time()
        timer.stop()
        timer.reset()
        timer.set_unit("minutes")


    def test_fps_timer(self):
        # Init checks
        with self.assertRaises(TypeError):
            FPSTimer(precision_decimals=None)
        with self.assertRaises(ValueError):
            FPSTimer(precision_decimals=-1)

        # Check all functions
        fps_timer = FPSTimer(precision_decimals=0)
        with self.assertRaises(RuntimeError): fps_timer.get_fps()
        with self.assertRaises(RuntimeError): fps_timer.increment()
        fps_timer.start()
        fps_timer.increment()
        self.assertEqual(fps_timer.get_frame_count(), 1)
        fps_timer.get_fps()
        fps_timer.get_avg_fps()
        fps_timer.reset()


    def test_months(self):
        self.assertEqual(len(month_names) == len(month_names_abb) == 12, True)

if __name__ == "__main__":
    unittest.main()