import time 
from functools import wraps


def timer(function): # decoratuer
    @wraps(function)
    
    def count_time(*args, **kwargs): # unpack tuple/dict
        start = time.perf_counter()
        output = function(*args, **kwargs)
        end = time.perf_counter()
        print(f'[TIMER] {function.__name__} executed in {end-start:.4f} seconds')
        return output
    
    return count_time


