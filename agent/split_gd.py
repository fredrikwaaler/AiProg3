import tensorflow as tf
import warnings
import numpy as np


class SplitGD:
    """
    Takes in a keras model and accommodates needs to modify gradients before applying them during backpropagation
    Uses eligibility traces to update params from previously seen states
    """

    def __init__(self, keras_model, learning_rate, discount_factor, eli_decay):
        self.model = keras_model
        self.eligs = np.array([])
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.eli_decay = eli_decay

    def update_eligs(self):
        """
        Decays eligibility of params by one time-step.
        Ensures that weights associated with states are updated with respect to
        how long ago they occurred during the episode.
        Discount by discount factor is also performed here
        """
        self.eligs = np.multiply(
            self.eligs, self.discount_factor * self.eli_decay)

    def reset_eli_dict(self):
        """
        Resets eligibilities (done before a new episode)
        """
        self.eligs = np.array([])

    def modify_gradients(self, gradients, td_error):
        """
        Modifies the gradients before backpropagation is performed.
        Modification depends on td-error and eligibility
        """
        warnings.filterwarnings("ignore")
        # Initializes new eligibilites after a reset (this will be done during the first fit() call in an episode)
        if len(self.eligs) == 0:
            # Gradients are a list of tensors, need to keep shape intact
            self.eligs = tf.zeros(shape=np.shape(gradients), dtype=tf.float32)
        # Eligibilty depends on how active parameter was for input state e_i = e_i + grad
        self.eligs = np.add(self.eligs, gradients, dtype=object)
        # Gradients are changed to equal e_i * delta
        gradients = np.multiply(self.eligs, td_error[0][0])
        return gradients

    def fit(self, state_tensor, td_error):
        """
        Takes in state and td_error computed after moving to state prime
        Updates weights in neural net with respect to how active they were for current prediction and previous states
        """
        params = self.model.trainable_weights
        with tf.GradientTape() as tape:
            prediction = self.model(state_tensor)
            gradients = tape.gradient(prediction, params)
            gradients = self.modify_gradients(gradients, td_error)
            self.model.optimizer.apply_gradients(
                zip(gradients, params))
