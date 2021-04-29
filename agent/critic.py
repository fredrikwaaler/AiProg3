import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from agent.split_gd import SplitGD


class Critic:
    def __init__(self, config, granularity):
        self.learning_rate = config["learning_rate"]
        self.eli_decay = config["eli_decay"]
        self.discount_factor = config["discount_factor"]
        self.dims = Critic.create_dims(
            internal_dims=config["internal_dims"], granularity=granularity)
        self.model = self.gennet(self.dims, learning_rate=self.learning_rate)
        self.splitGD = SplitGD(self.model, self.learning_rate,
                               self.discount_factor, self.eli_decay)
        self.studied = []

    @staticmethod
    def create_dims(internal_dims, granularity):
        return [granularity[0]*granularity[1]] + internal_dims + [1]

    def reset_eli_dict(self):
        """
        Resets eligibilities (done before a new episode)
        """
        self.splitGD.reset_eli_dict()

    def update_eligs(self, *args):
        """
        Decays eligibilities for one step
        """
        self.splitGD.update_eligs()

    def train(self, state, td_error):
        """
        Trains network after a new observation (td_error), eligibilites come into use in splitGD class fit function
        :param state: list(list(int))
        :param td_error: float
        """
        state_tensor = self.convert_state_to_tensor(state)
        td_error_tensor = tf.reshape(td_error, [1, 1])
        self.model = self.splitGD.fit(
            state_tensor=state_tensor, td_error=td_error_tensor)

    def compute_td_err(self, current_state, next_state, reward):
        """
        Computes TD-error after performing an action from current_state leading next_state and reward
        Measures degree of surprise after a state transition
        :param current_state: list(list(int))
        :param next_state: list(list(int))
        :param reward: integer
        """
        # Initialize unseen states as random float between 0 and 1
        if list(current_state) not in [list(a) for a in self.studied]:
            # if current_state not in self.studied:
            self.studied.append(current_state)
            state_value = random.uniform(0, 1)
        else:
            # Predict value of current state
            s = self.convert_state_to_tensor(current_state)
            state_value = self.splitGD.model(s).numpy()[0][0]

        # Initialize unseen "next" states as random float between 0 and 1 as well
        if list(next_state) not in [list(a) for a in self.studied]:
            state_prime_value = random.uniform(0, 1)
        else:
            # Predict value of new state
            s_p = self.convert_state_to_tensor(next_state)
            state_prime_value = self.splitGD.model(s_p).numpy()[0][0]
        # delta = r + V(s') - V(s)
        return reward + self.discount_factor * state_prime_value - state_value

    def convert_state_to_tensor(self, state):
        """
        Converts a list representation of the state to a tensor
        :param state: list(list(int))
        """
        tensor = tf.convert_to_tensor(
            [np.concatenate([np.array(i) for i in state])])
        # dims[0] depends on board size
        return tf.reshape(tensor, [1, self.dims[0]])

    def gennet(self, dims, learning_rate, opt='SGD', loss='MeanSquaredError()', activation="relu", last_activation="sigmoid"):
        """
        Compiles a keras model with dimensions given by dims
        :param dims: list(int)
        :param learning_rate: float 
        """
        model = keras.models.Sequential()
        opt = eval('keras.optimizers.' + opt)
        loss = eval('tf.keras.losses.' + loss)
        model.add(keras.layers.Dense(input_shape=(dims[0],),  # Determines shape after first input of a board state
                                     units=dims[0], activation=activation))
        for layer in range(1, len(dims)-1):
            model.add(keras.layers.Dense(
                units=dims[layer], activation=activation))
        model.add(keras.layers.Dense(
            units=dims[-1], activation=last_activation))
        model.compile(optimizer=opt(lr=learning_rate), loss=loss)
        return model
