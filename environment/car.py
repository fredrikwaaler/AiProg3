import math


class Car:

    def __init__(self, config):
        self.initial_position = config["initial_state"][0]
        self.position = config["initial_state"][0]
        self.velocity = config["initial_state"][1]
        self.maxv = 0.07
        self.minv = -0.07
        self.maxp = 0.6
        self.minp = -1.2

    # lower and upper bounds on velocity: -0.07 and 0.07
    def update_velocity_and_position(self, action):
        # action is defined as either 1, 0 or -1
        new_v = self.velocity + (.001)*action - (.0025) * \
            math.cos(3*self.position)
        self.velocity = lambda new_v, minv, maxv: max(
            min(self.maxv, new_v), self.minv)
        self.update_position()

    # lower and upper bounds for position being -1.2 and 0.6, respectively
    def update_position(self):
        new_p = self.position + self.velocity
        self.position = lambda new_v, minp, maxp: max(
            min(self.maxp, new_p), self.minp)

    def get_state(self):
        return self.position, self.velocity, self.initial_position

    def reached_top(self):
        if math.cos(3*(self.position+math.pi/2)) > 0.97 and self.position > self.initial_position:
            return True
        return False
