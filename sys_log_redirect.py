import sys

if not sys.stdout:
    class FakeStdOut:
        def __init__(self, filename="main-backend.log"):
            self.log = open(filename, "a")

        def write(self, message):
            self.log.write(message)

        def flush(self):
            pass

        def isatty(self):
            return True

    sys.stdout = FakeStdOut()
