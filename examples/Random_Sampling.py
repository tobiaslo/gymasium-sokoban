import gymnasium as gym
import gym_sokoban
import time
from gymnasium.wrappers import RecordVideo


# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package
env_name = 'Boxoban-v0'
env = gym.make(env_name, render_mode='rgb_array')
env = RecordVideo(env, video_folder='.')

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))

for i_episode in range(1):
    observation, info = env.reset(seed=None, options= None)

    for t in range(10):
        # env.render()
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(1)
        observation, reward, done, truncated, info = env.step(action)

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    env.close()

time.sleep(10)
