import time

import pytest

from emotion_classifier.timing import Timer

def test_timer_measures_a_sleep():
    with Timer() as timer:
        time.sleep( 0.05 )
    assert timer.seconds == pytest.approx( 0.05, abs = 0.04 )

def test_timer_is_zero_before_use():
    assert Timer().seconds == 0.0

def test_timer_records_time_even_if_body_raises():
    timer = Timer()
    with pytest.raises( RuntimeError ):
        with timer:
            time.sleep( 0.02 )
            raise RuntimeError( "boom" )
    assert timer.seconds >= 0.02
