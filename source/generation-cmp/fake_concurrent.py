from concurrent.futures import Executor, Future

class SingleThreadExecutor(Executor):
    '''
    Used to replace multithreading / multiprocessing for debugging purposes.
    Offers a similar interface as ThreadPoolExecutor and ProcessPoolExecutor.
    '''
    def __init__(self, *args, **kwargs): pass

    def submit(self, fn, *args, **kwargs):
        future = Future()

        result = fn(*args, **kwargs)
        future.set_result(result)

        return future
