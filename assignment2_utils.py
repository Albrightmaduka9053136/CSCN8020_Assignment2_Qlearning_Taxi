#!/usr/bin/python3
import time
import gymnasium as gym

#---------------------------
# Helper functions
#---------------------------

def describe_env(env: gym.Env):
    num_actions = env.action_space.n
    obs = env.observation_space
    num_obs = env.observation_space.n
    reward_range = getattr(env, "reward_range", None)

    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger",
    }
    print("Observation space: ", obs)
    print("Observation space size: ", num_obs)
    print("Reward Range: ", reward_range)
    print("Number of actions: ", num_actions)
    print("Action description: ", action_desc)
    return num_obs, num_actions


def get_action_description(action):
    action_desc = {
        0: "Move south (down)",
        1: "Move north (up)",
        2: "Move east (right)",
        3: "Move west (left)",
        4: "Pickup passenger",
        5: "Drop off passenger",
    }
    return action_desc[action]


def describe_obs(obs):
    obs_desc = {
        0: "Red",
        1: "Green",
        2: "Yellow",
        3: "Blue",
        4: "In taxi"
    }
    obs_dict = breakdown_obs(obs)
    print(
        "Passenger is at: {0}, wants to go to {1}. Taxi currently at ({2}, {3})".format(
            obs_desc[int(obs_dict["passenger_location"])],
            obs_desc[int(obs_dict["destination"])],
            int(obs_dict["taxi_row"]),
            int(obs_dict["taxi_col"]),
        )
    )


def breakdown_obs(obs):
    # ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination = X
    destination = obs % 4
    obs -= destination
    obs //= 4

    passenger_location = obs % 5
    obs -= passenger_location
    obs //= 5

    taxi_col = obs % 5
    obs -= taxi_col
    obs //= 5

    taxi_row = obs

    observation_dict = {
        "destination": destination,
        "passenger_location": passenger_location,
        "taxi_row": taxi_row,
        "taxi_col": taxi_col
    }
    return observation_dict


def simulate_episodes(env, agent, num_episodes=3):
    """
    Visual simulation using agent.select_action(state).
    NOTE: In Gymnasium, step returns (obs, reward, terminated, truncated, info).
    """
    for _ in range(num_episodes):
        terminated = False
        truncated = False
        state, _ = env.reset()
        describe_obs(int(state))
        env.render()
        while not (terminated or truncated):
            action = agent.select_action(int(state))
            env.render()
            time.sleep(0.1)
            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
        time.sleep(1.0)


def main():
    env = gym.make("Taxi-v3")
    describe_env(env)

    env2 = gym.make("Taxi-v3", render_mode="human")
    # simulate_episodes(env2, agent)

if __name__ == "__main__":
    main()