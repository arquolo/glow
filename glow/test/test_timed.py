import time
from glow.core import Timed

v = Timed('world', timeout=.1)

for t in range(10):
    print(t, v.get())
    time.sleep(.2)
