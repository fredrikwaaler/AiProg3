import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment.coarsecoder import CoarseCoder
from environment.car import Car


class Environment:

    def __init__(self, config):
        self.config = config
        self.max_steps = config["max_steps"]
        self.final_reward = config["final_reward"]
        self.loser_penalty = config["loser_penalty"]
        self.coarse_code = CoarseCoder(config)
        self.car = Car(config)
        self.steps = 0

    def visualize_landscape(self, position):
        # the relationship between x and height (depth) is given by:
        height = math.cos(3*(position+math.pi/2))
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set(xlim=(self.car.minp, self.car.maxp), ylim=(0, 1.5))

        x = np.linspace(self.car.minp, self.car.maxp, 91)
        t = np.linspace(0, 1.5, 30)
        X2, T2 = np.meshgrid(x, t)

        height = math.cos(3*(T2+math.pi/2))
        F = 0.9*height*np.sinc(X2*(1 + height))

        line = ax.plot(x, F[0, :], color='k', lw=2)[0]

        def animate(i):
            line.set_ydata(F[i, :])

        anim = FuncAnimation(fig, animate, interval=100, frames=len(t)-1)

        plt.draw()
        plt.show()

    def update_steps(self):
        self.steps += 1

    def reached_max_steps(self):
        return self.steps == self.max_steps

    def reached_top(self):
        return self.car.reached_top()

    def get_actions(self):
        """ Returns actions for the environment.
         TODO: if necessary, implement with check that returns only legal actions, i.e. only 0 and -1 if at end of the right hand side of position range
        """
        return [1, 0, -1]

    def perform_action(self, action):
        self.car.update_velocity_and_position(action)
        pos, _, init_pos = self.car.get_state()
        reward = self.loser_penalty if self.steps == self.max_steps - 1 else 0
        reward = self.finalreward * \
            (math.cos(3*(pos+math.pi/2))) if pos > init_pos else reward
        return reward

    def new_simulation(self):
        self.steps = 0
        self.car(self.config)

    def get_state(self):
        pos, vel = self.car.get_state()
        return self.coarse_code.get_coarse_encoding(pos, vel)
