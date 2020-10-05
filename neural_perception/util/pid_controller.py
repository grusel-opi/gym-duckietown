import time

class PID:
    def __init__(self, k_p = 1.0, k_i = 0.0, k_d = 0.0, target = 0):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.target = target
        self.min_out, self.max_out = (None, None)
        self.reset()

    def update(self, current_value):
        now = time.monotonic()
        dt = now - self.last_t

        err = self.target - current_value

        if self.last_val is not None:
            de = current_value - self.last_val
        else:
            de = current_value - current_value

        self.p = self.k_p * err
        self.i += self.k_i * err * dt
        self.i = clip(self.i, self.out_lim)
        self.d = -self.k_d * de / dt

        out = self.p + self.i + self.d
        out = clip(out, self.out_lim)

        self.last_out = out
        self.last_val = current_value
        self.last_t = now

        return out

    def out_lim(self, lim):
        if lim is None:
            self.min_out, self.max_out = None, None
            return

        min_output, max_output = lim

        if None not in lim and max_output < min_output:
            raise ValueError('The lower limit must be less than upper limit!')

        self._min_output = min_output
        self._max_output = max_output

        self.i = clip(self.i, self.out_lim)
        self.last_out = clip(self.last_out, self.out_lim)

    def reset(self):
        self.p = 0
        self.i = 0
        self.d = 0

        self.last_t = time.monotonic()
        self.last_out = None
        self.last_val = None

def clip(val, lim):
    lower, upper = lim

    if val is None:
        return None
    elif upper is not None and val > upper:
        return upper
    elif lower is not None and val < lower:
        return lower
    return val