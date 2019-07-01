from enum import IntEnum
from contextlib import contextmanager
from typing import List
import tempfile
import ctypes
import sys
import os

LIB_PATH = '/'.join(os.path.realpath(__file__).split('/')[:-1])
lib = ctypes.cdll.LoadLibrary('{}/libmarabou.so'.format(LIB_PATH))


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

def _create_property_file() -> str:
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('y0 >= 0.01')
        path = f.name
    return path


class MarabouResult(IntEnum):
    UNSAT = 1
    SAT = 2
    TIMEOUT = 3
    ERROR = 4
    UNKNOWN = 5


class Marabou:
    def __init__(self):
        self.LP_c_char = ctypes.POINTER(ctypes.c_char)
        self.LP_LP_c_char = ctypes.POINTER(self.LP_c_char)

        self.obj = lib.Marabou_new()

    def __del__(self):
        lib.Marabou_delete(self.obj)

    def run(self, argv: List[str]) -> MarabouResult:
        argv.append(_create_property_file())
        _argc = len(argv)
        _argv = (self.LP_c_char * (_argc + 1))()
        for i, arg in enumerate(argv):
            enc_arg = arg.encode('ascii')
            _argv[i] = ctypes.create_string_buffer(enc_arg)

        ret = lib.Marabou_run(self.obj, ctypes.c_int(_argc), _argv)

        for result in MarabouResult:
            if ret == result:
                return result

if __name__ == '__main__':
    _ret = Marabou().run(sys.argv)
    print(_ret)
