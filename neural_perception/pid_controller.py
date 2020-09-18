from time import time


class PID:

    def __init__(self, kp, ki, kd, target, origin_time=None):
        if origin_time is None:
            origin_time = time()

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.cp = 0.0
        self.ci = 0.0
        self.cd = 0.0

        self.prev_time = origin_time
        self.prev_err = 0.0

        self.target = target

    def update_target(self, new_target):
        self.target = new_target

    def update(self, current_value, curr_time=None):
        if curr_time is None:
            curr_time = time()

        dt = curr_time - self.prev_time
        if dt <= 0:
            return 0

        err = self.target - current_value

        de = err - self.prev_err

        self.cp = err
        self.ci += err * dt
        self.cd = de / dt

        self.prev_time = curr_time
        self.prev_err = err

        return (
                self.kp * self.cp +
                self.ki * self.ci +
                self.kd * self.cd
        )
