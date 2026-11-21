import time

class Timer:
    """Wall-clock context manager; `seconds` is set on exit, even if the body raises."""

    def __init__( self ) -> None:
        self.seconds: float = 0.0
        self._start: float = 0.0

    def __enter__( self ) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__( self, *exc_info ) -> None:
        self.seconds = time.perf_counter() - self._start
