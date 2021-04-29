import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment.coarsecoder import CoarseCoder, TileEncoder
from environment.car import Car
import yaml


class Environment:

    def __init__(self, config):
        self.config = config
        self.max_steps = config["max_steps"]
        self.final_reward = config["final_reward"]
        self.loser_penalty = config["loser_penalty"]
        self.coarse_code = TileEncoder(config)
        self.car = Car(config)
        self.steps = 0

    def visualize_landscape(self, car_positions):
        # the relationship between x and height (depth) is given by:
        car_heights = [math.cos(3 * (pos + math.pi / 2))
                       for pos in car_positions]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set(xlim=(self.car.minp, self.car.maxp), ylim=(-1.5, 1.5))

        x = np.linspace(self.car.minp, self.car.maxp, 91)
        y = [math.cos(3 * (pos + math.pi / 2)) for pos in x]

        ax.plot(x, y, 'k', lw=1)
        car = ax.plot(car_positions[0], car_heights[0], 'ro', lw=4)[0]

        def animate(i):
            car.set_xdata(car_positions[i])
            car.set_ydata(car_heights[i])

        anim = FuncAnimation(fig, animate, interval=100,
                             frames=len(car_positions)-1)
        anim.save('filename.mp4')

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
        old_pos, old_vel, _ = self.car.get_state()
        self.car.update_velocity_and_position(action)
        pos, vel, _ = self.car.get_state()

        if round(pos, 1) == 0.6:
            r = 500
        else:
            r = -1
        # reward = self.loser_penalty if self.steps == self.max_steps - 1 else reward
        return r

    def perform_action2(self, action):
        # Calculate current distance from init pos
        self.car.update_velocity_and_position(action)
        pos, _, _ = self.car.get_state()
        if round(pos, 1) != 0.6:
            return -1
        else:
            return self.final_reward

    def new_simulation(self):
        self.steps = 0
        self.car = Car(self.config)

    def get_state(self):
        pos, vel, _ = self.car.get_state()
        return self.coarse_code.get_coarse_encoding(pos, vel)

    def get_position(self):
        pos, _, _ = self.car.get_state()
        return pos


"""
if __name__ == '__main__':
    config = yaml.full_load(open("/configs/config.yml"))
    env_cfg = config["Environment"]
    env = Environment(env_cfg)

    x = np.linspace(-3, 3, 91)
    t = np.linspace(1, 25, 30)
    X2, T2 = np.meshgrid(x, t)

    sinT2 = np.sin(2 * np.pi * T2 / T2.max())
    F = 0.9 * sinT2 * np.sinc(X2 * (1 + sinT2))


    env.visualize_landscape([-1.0, 0.2, -0.6, 0.3, 0.4])
"""
