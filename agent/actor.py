# actions: 1) apply a small force (-1) that accelerates the cart to the left,
# 2) apply a small force (+1) that accelerates it to the right, and 3) apply no force (0)

import random
from collections import defaultdict


class Actor:
    """
    The actor class keeps track of the policy to be used when deciding next move.
    Uses dictionaries to keep track of the value of each SAP pair and eligibility.
    """

    def __init__(self, learning_rate, discount_factor, eli_decay, epsilon, epsilon_decay):
        """
        Initializes an actor using a default value of 0 to allow accumulated values.
        :param learning_rate: float
        :param discount_factor: float
        :param eli_decay: float
        :param epsilon: float
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.policy_dict = defaultdict(lambda: 0)
        self.eli_dict = {}  # defaultdict(lambda: 0)
        self.discount_factor = discount_factor
        self.eli_decay = eli_decay
        self.learning_rate = learning_rate

    def update_policy_dict(self, state, action, td_err):
        """
        Updates the policy using the formula:
         policy_value+ learning_rate*td_err*eligibility
        :param state: list[list[int]]
        :param action: list[tuple(int,int)]
        :param td_err: float
        """
        if (str(state), str(action)) in self.policy_dict.keys():
            self.policy_dict[(str(state), str(action))] += self.learning_rate * \
                td_err * self.get_elig(str(state), str(action))
        else:
            self.policy_dict[(str(state), str(action))] = self.learning_rate * \
                td_err * self.get_elig(state, action)

    # Updates eligibility using: discount_factor*eli_decay*eli_dict[state,action]

    def update_eli_dict(self, state, action, i):
        """
        Updates eligibility using: discount_factor*eli_decay*eli_dict[state,action]

        :param state: list[list[int]]
        :param action: list[tuple(int,int)]
        :param i: int (0 for current state)

        """
        if i == 0:
            self.eli_dict[(str(state), str(action))] = 1
            return
        else:
            value = self.get_elig(state, action) * \
                self.discount_factor * self.eli_decay
            element = {(str(state), str(action)): value}
            self.eli_dict.update(element)

    def get_elig(self, state, action):
        """
        Return the eligibility for a SAP pair.
        :param state: list[list[int]]
        :param action: list[tuple(int,int)]
        """
        if (str(state), str(action)) in self.eli_dict.keys():
            return self.eli_dict[(str(state), str(action))]
        else:
            return 0

    def get_policy(self, state, action):
        """
        Returns the policy for a given SAP pair.

        :param state: list[list[int]]
        :param action: list[tuple(int,int)]
        """

        if (str(state), str(action)) in self.policy_dict.keys():
            return self.policy_dict[(str(state), str(action))]
        else:
            return 0

    def reset_eli_dict(self):
        """
        Reset eli_dict after episode ends
        """
        self.eli_dict = {}

    def get_action(self, state, legal_actions):
        """
        Returns action recommended by current policy, with the
        exception of random exploration epsilon percent of the time
        :param state: list[list[int]]
        :param legal_actions: list[list[tuple(int,int)]]
        """
        self.epsilon = self.epsilon*self.epsilon_decay
        if random.uniform(0, 1) >= self.epsilon:
            return max(legal_actions, key=lambda action: self.get_policy(state, action))
        return random.choice(legal_actions)
