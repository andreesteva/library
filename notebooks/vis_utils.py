import time

def progress(i, num_total, t):
    """Function for printing progress and time elapsed."""
    dt = time.time() - t
    print '\r', i, '/',num_total, 'Elapsed Time:', dt , 'Time Remaining:',
    print 1.0 * dt / (i+1) * (num_total-i-1),


def tic():
    """Begin timing."""
    return time.time()


def toc(t, text = ''):
    """Pring elapsed time."""
    print 'Elapsed Time:', text, time.time() - t
