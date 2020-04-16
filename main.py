import gym
import cma
from attention_agent import AttentionAgent


env_name = 'CarRacing-v0'

env = gym.make(env_name)
agent = AttentionAgent()

max_episodes = 500
max_steps = 1000

for i_episode in range(max_episodes):
    observation = env.reset()
    for t in range(max_steps):
        env.render()
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break
env.close()