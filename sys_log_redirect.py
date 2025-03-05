import sys
import codecs

if not sys.stdout:
    class FakeStdOut:
        def __init__(self, filename="main-backend.log"):
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            if isinstance(message, str):
                message = message.encode('utf-8')
            self.log.write(message.decode('utf-8'))

        def flush(self):
            pass

        def isatty(self):
            return True

    sys.stdout = FakeStdOut()
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

