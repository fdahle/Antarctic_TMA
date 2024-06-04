# Package imports
import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def log_and_print():
    class Tee(object):
        def __init__(self, _log_stream):
            self.log_stream = _log_stream
            self.stdout = sys.stdout

        def write(self, message):
            self.stdout.write(message)
            self.log_stream.write(message)

        def flush(self):
            self.stdout.flush()
            self.log_stream.flush()

    log_stream = StringIO()
    tee = Tee(log_stream)
    sys.stdout = tee
    try:
        yield log_stream
    finally:
        sys.stdout = sys.stdout.stdout
