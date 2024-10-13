import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Q matrix of expected value of taking action a in state s
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Learning rate = how quickly we adapt
lr = .85
# Q(s,a) = r + y * max(Q(s', a'))
y = .99

num_episodes = 2000
max_steps_per_episode = 99
num_steps = []
rewards = []

for episode in range(num_episodes):
    if (episode % 100) == 0:
        print("-----------NEW EPISODE ({})------------".format(episode))
    state = env.reset()
    total = 0
    steps = 0
    for _ in range(max_steps_per_episode):
        action = np.argmax(Q[state, :] +
                           np.random.randn(1, env.action_space.n) * (1.0 / (episode + 1)))
        new_state, reward, done, _ = env.step(action)
        # Update Q-table
        Q[state, action] = (Q[state, action] +
                            lr * (reward +
                                  y * np.max(Q[new_state, :]) - Q[state, action]))

        # Accounting
        steps += 1
        total += reward
        state = new_state
        if done:
            break
    num_steps.append(steps)
    rewards.append(total)

env.render()
plot = plt.figure()
plt.plot(rewards)
plot.show()
input('asss')
print("Final Q table values")
print(Q)
