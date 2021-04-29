# An episode should halt when either a) the cart successfully reaches the rightmost point, where pos = 0.6,
#  or b) 1000 actions have been taken.


# TODO: Implement visualization of 100 episodes of mountain-car simulations, with a maximum of 1000 steps allowed per episode.
# TODO: Illustration of mountain-car simulation status (normally visualized as a movie) with the curved line depicting the landscape and the oval denoting the mountain car.
# TODO: Visualize the reward function

from agent.critic import Critic
from agent.actor import Actor
from environment.environment import Environment
import yaml
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm  # Progressbar

config = yaml.full_load(open("configs/config.yml"))
env_cfg = config["Environment"]
actor_cfg = config["Actor"]
critic_cfg = config["Critic"]
training_cfg = config["Training"]


def main():
    """
    Sets the parameters for the Environment, Critic, and Actor according to the imported config file.
    Creates an environment where a predefined number of episodes can be performed.
    Instantiates an actor to keep track of the policy, and a critic to keep track of the value at each state
    Runs a predefined number of episodes creating a new board for each episode.
    For each episode, the actor and the critic are updated according to the Actor-Critic model.
    Finally, epsilon is set to zero, and the environment plays a game with the updated policy.
    """

    env = Environment(env_cfg)
    granularity = env_cfg["granularity"]
    critic = Critic(critic_cfg, granularity)
    actor = Actor(actor_cfg)

    episodes = training_cfg["number_of_episodes"]
    steps_per_episode = []

    for episode in tqdm(range(episodes), desc=f"Playing {episodes} episodes", colour='#39ff14'):
        env.new_simulation()
        path = []
        critic.reset_eli_dict()
        actor.reset_eli_dict()
        while not env.reached_top() or not env.reached_max_steps():
            print(env.steps)
            env.update_steps()
            current_state = copy(env.get_state())
            legal_actions = env.get_actions()
            action = actor.get_action(
                state=current_state, legal_actions=legal_actions)
            path.append((str(current_state), str(action)))
            reward = env.perform_action(action=action)

            td_err = critic.compute_td_err(
                current_state=current_state, next_state=env.get_state(), reward=reward)

            # Previous states on the path are updated as well during the call to train() by eligibility traces
            critic.train(state=current_state, td_error=td_err)
            critic.update_eligs()

            # Update actor beliefs on SAPs for all pairs seen thus far in the episode
            for i, sap in enumerate(reversed(path)):
                actor.update_eli_dict(
                    state=str(sap[0]), action=str(sap[1]), i=i)
                actor.update_policy_dict(
                    state=str(sap[0]), action=str(sap[1]), td_err=td_err)

        steps.append(env.get_steps())

    plot_learning(steps_per_episode)

    # Enable history tracking to visualize final simulation
    env.new_simulation()  # track_history=True)

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0  # Set exploration to 0
    print("Attempting final simulation to show you how smart I am now")
    while not env.reached_top() or not env.reached_max_steps():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    # env.board.visualize(0.1)


def plot_learning(steps_per_episode):
    """
    Plots remaining pieces after each episode during a full run of training
    Should converge to one if the agent is learning
    """
    episode = [i for i in range(len(steps_per_episode))]
    plt.plot(episode, steps_per_episode)
    plt.xlabel("Episode number")
    plt.ylabel("Steps")
    plt.show()


main()
