"""进程间互斥：基于 fcntl 的逻辑 run 锁（自 sequential/launcher.py 迁入，供队列 runner 与各研究线共用）。"""
import fcntl
import os


class LogicalRunLock:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.handle = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, 'a+', encoding='utf-8')
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f'Logical run is already locked: {self.path}') from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f'{os.getpid()}\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def release(self):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
