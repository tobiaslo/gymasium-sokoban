import gymnasium as gym
import gym_sokoban
import time
from gymnasium.wrappers import RecordVideo


# Before you can make a Sokoban Environment you need to call:
# import gym_sokoban
# This import statement registers all Sokoban environments
# provided by this package
env_name = 'Sokoban-v0'
env = gym.make(env_name, render_mode='human', num_gen_steps=4)
# env = RecordVideo(env, video_folder='.')

ACTION_LOOKUP = env.unwrapped.get_action_lookup()
print("Created environment: {}".format(env_name))


for i_episode in range(10):
    observation, info = env.reset(seed=None)

    for t in range(3):
        env.render()
        action = env.action_space.sample()

        # Sleep makes the actions visible for users
        time.sleep(0.01)
        observation, reward, done, truncated, info = env.step(action)

        print(ACTION_LOOKUP[action], reward, done, info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            env.render()
            break

    env.close()
