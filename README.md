# rl-examples
Examples of published reinforcement learning algorithms in recent
literature implemented in TensorFlow.
Most of my research is in the continuous domain, and I haven't spent much
time testing these in discrete domains such as Atari etc.

![PPO LSTM solving BipedalWalker-v2](https://github.com/Anjum48/rl-examples/blob/master/ppo/BipedalWalker-v2.gif)
![PPO solving CarRacing-v0](https://github.com/Anjum48/rl-examples/blob/master/ppo/CarRacing-v0.gif)

*BipedalWalker-v2 solved using PPO with a LSTM layer*
*CarRacing-v0 solved using PPO with a joined actor-critic network*

## Algorithms Implemented
Thanks to DeepMind and OpenAI for making their research openly available.
Big thanks also to the TensorFlow community.

| Algorithm | Paper                                                   | 
| --------- | ------------------------------------------------------- |
| DPPG      | [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)     |
| A3C       | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)    |
| PPO       | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)                 |
| DPPO      | [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286) |
| GAE       | [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) |


- GAE was used in all algorithms except for DPPG
- Where possible, I've added an LSTM layer to the policy and value functions.
This sometimes achieved higher scores in some environments, but can have stability issues
- In some environments, having a joint network for the actor & critic performs better (i.e. where CNNs are used).
These scripts are suffixed, e.g. `ppo_joined.py`

## Training
All the Python scripts are written as standalone scripts. Just run them
as you would for a single file or in your IDE. The models
and TensorBoard summaries are saved in the same directory as the script.
DPPO has a helper script to set off the worker threads

## Requirements
- Python 3.5+
- OpenAI Gym 0.10.3+
- TensorFlow 1.6
- Numpy 1.13+

DPPO was tested on a 16 core machine using CPU only, so the helper
script will need to be updated for your particular setup.
For my setup, there was usually no speed advantage training BipedalWalker on the
CPU vs GPU (GTX 1080), but CarRacing did get a performance boost due to the usage of CNNs

## Issues/Todo's
- The LSTM batching in A3C is incorrect. Need to fix this (see PPO_LSTM for the correct implementation)
- Distributed Proximal Policy Optimisation with the LSTM (`dppo_lstm.py`) is sometimes a bit unstable,
but does work at low learning rates
