import gym
from attention_agent import AttentionAgent


env_name = 'CarRacing-v0'

env = gym.make(env_name)
env.close()
env = gym.make(env_name)
agent = AttentionAgent()

max_episodes = 500
max_steps = 1000

for i_episode in range(max_episodes):
    observation = env.reset()
    reward_sum = 0
    for t in range(max_steps):
        #env.render()

        action = agent.act(observation)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            print('Episode finished after {} timesteps'.format(t+1))
            break
    agent.episode_end(reward_sum)
    print('Episode reward: ', reward_sum)



# Test
agent.test()
observation = env.reset()
reward_sum = 0
for t in range(max_steps):
    env.render()
    action = agent.act(observation)
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    if done:
        print('Episode finished after {} timesteps'.format(t+1))
        break
print('Episode reward: ', reward_sum)


env.close()
