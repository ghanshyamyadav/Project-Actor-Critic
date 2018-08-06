import numpy as np
import random
import copy
from collections import namedtuple, deque
import matplotlib.pyplot as plt

from agent import Agent
from unityagents import UnityEnvironment

import torch
import torch.nn.functional as F
import torch.optim as optim

env = UnityEnvironment(file_name='./Reacher')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states)

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)



agent = Agent(num_agents, state_size=state_size, action_size=action_size, random_seed=0)
def load_checkpoint():
    try:
        hyperparams = torch.load('./params.pth')
        return hyperparams['episode'], hyperparams['scores'], hyperparams['scores_deque']
    except Exception as e:
        return 1, [], deque(maxlen=100)

def ddpg(n_episodes=310, max_t=1000):
    episode, scores, scores_deque = load_checkpoint()
    #print(episode, scores)
    agent.reinitialise('checkpoint_agent.pth')
    scores_deque = deque(maxlen=100)
    for i_episode in range(episode, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            score += env_info.rewards

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):                                  # exit loop if episode finished
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(score)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.saveStates('checkpoint_agent.pth')
            torch.save({'episode': i_episode, 'scores':scores, 'scores_deque': scores_deque}, 'params.pth')
    return scores

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
env.close()