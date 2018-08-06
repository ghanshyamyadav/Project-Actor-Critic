[//]: # (Image References)

[image1]: ./image.gif "Trained Agent"

# Project : Continuous Control

### Introduction

In this project, an agent has been trained in a environment where a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 

![Trained Agent][image1]

The agent has been trained to maintain its position at the target location for as many time steps as possible.


**Action and State space**

The state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.
 
 Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


**Goal**

The goal of the project is to solve the environment considering the presence of multiple agents.  In particular, twenty different agents must achieve an  average score of +30 (over 100 consecutive episodes, and over all agents).  

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

**Algorithm**

An actor-critic, model-free algorithm based on the deterministic policy gradient has been used to solve the environment.

### Getting Started

1. Follow the [instructions](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) to install Unity ML-Agents. 

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
       - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
       - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
       - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the environment.


### Instructions

Follow the instructions in `Report.ipynb` to get started with training your own agent!  To use a Jupyter notebook, run the following command from the `p2_continuous_control/` folder:
```
jupyter notebook
```
and open `Report.ipynb` from the list of files.  Alternatively, you may prefer to work with the [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) interface.  To do this, run this command instead:
```
jupyter-lab
```
The agent can also be trained from the command line.
From the p2_continuous_control/ folder, run the following command:

```
python3 train_agent.py  
```
