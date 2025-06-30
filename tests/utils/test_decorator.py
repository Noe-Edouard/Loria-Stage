from utils.decorator import log_time, log_call
from utils.decorator import format_arg
import numpy as np

def test_timer(caplog):
    @log_time()
    def dummy_func():
        return 42

    with caplog.at_level("DEBUG", logger="timer"):
        result = dummy_func()
        assert result == 42
        
def test_caller(caplog):
    @log_call()
    def dummy_func(a, b=2):
        return a + b

    with caplog.at_level("DEBUG", logger="caller"):
        result = dummy_func(3, b=5)
        assert result == 8
        
def test_format():
    
    arr = np.ones((2,3))
    assert format_arg(arr).startswith("ndarray(shape=(2, 3), dtype=")
    assert format_arg([1]*20) == "list(len=20)"
    assert format_arg({"a":1, "b":2}).startswith("dict(keys=[")

