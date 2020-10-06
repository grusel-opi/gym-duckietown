import time
from sympy.solvers.inequalities import reduce_rational_inequalities as r_r_i
from sympy.abc import x


def clip(val, lim):
    lower_bound, upper_bound = lim

    if val is None:
        return None
    elif upper_bound is not None and val > upper_bound:
        return upper_bound
    elif lower_bound is not None and val < lower_bound:
        return lower_bound
    return val

def calculate_out_lim(env, vel):
    baseline = env.unwrapped.wheel_dist
    k_r_inv = (env.gain + env.trim) / env.k
    u = (((vel + 0.5 * x * baseline) / env.radius) * k_r_inv)

    lower_bound = r_r_i([[u >= -1]], x)
    upper_bound = r_r_i([[u <= 1]], x)

    return (float(lower_bound.as_set().start), float(upper_bound.as_set().end))


class PID:
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, target=0, out_lim=(None, None)):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.target = target
        self.min_out, self.max_out = out_lim
        self.reset()

    def update(self, curr_val):
        now = time.monotonic()
        dt = now - self.last_t

        err = self.target - curr_val

        if self.last_in is not None:
            de = curr_val - self.last_in
        else:
            de = curr_val - curr_val

        self.p = self.k_p * err
        self.i += self.k_i * err * dt
        self.i = clip(self.i, self.get_out_lim())
        self.d = -self.k_d * de / dt

        out = self.p + self.i + self.d
        out = clip(out, self.get_out_lim())

        self.last_out = out
        self.last_in = curr_val
        self.last_t = now

        return out

    def get_out_lim(self):
        return self.min_out, self.max_out

    def set_out_lim(self, limits):
        if limits is None:
            self.min_out, self.max_out = None, None
            return

        min_output, max_output = limits

        self.min_out = min_output
        self.max_out = max_output

        self.i = clip(self.i, self.get_out_lim())
        self.last_out = clip(self.last_out, self.get_out_lim())

    def reset(self):
        self.p = 0
        self.i = 0
        self.d = 0

        self.last_t = time.monotonic()
        self.last_out = None
        self.last_in = None