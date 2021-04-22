import math


class Car:

    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.maxv = 0.07
        self.minv = -0.07
        self.maxp = 0.6
        self.minp = -1.2

    # lower and upper bounds on velocity: -0.07 and 0.07
    def update_velocity(self, action):
        # action is defined as either 1, 0 or -1
        new_v = self.velocity + (.001)*action - (.0025) * \
            math.cos(3*self.position)
        self.velocity = lambda new_v, minv, maxv: max(
            min(self.maxv, new_v), self.minv)
        self.update_position()

    # lower and upper bounds for position being -1.2 and 0.6, respectively
    def update_position(self, velocity):
        new_p = self.position + self.velocity
        self.position = lambda new_v, minp, maxp: max(
            min(self.maxp, new_p), self.minp)
