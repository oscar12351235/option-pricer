from typing import Protocol


class Exercise(Protocol):
    def exercise_time(self, timeindex: int, timegrid) -> bool: ...


class European:
    def exercise_time(self, timeindex, timegrid):
        return timeindex == len(timegrid) - 1


class American:
    def exercise_time(self, timeindex, timegrid):
        return 0 < timeindex < len(timegrid) - 1
