# rl-examples
Examples of published reinforcement learning algorithms in recent
literature implemented in TensorFlow.
Most of my research is in the continuous domain, and I haven't spent much
time testing these in discrete domains such as Atari etc.

![PPO LSTM solving BipedalWalker-v2](https://github.com/Anjum48/rl-examples/blob/master/ppo/BipedalWalker_PPO-LSTM.gif)

*BipedalWalker-v2 solved using PPO with a LSTM layer*

## Algorithms Implemented
Thanks to DeepMind and OpenAI for making their research openly available.
Big thanks also to the TensorFlow community.

| Algorithm | arXiv Link                       | Paper                                                   | 
| --------- | -------------------------------- | ------------------------------------------------------- |
| DPPG      | https://arxiv.org/abs/1509.02971 | Continuous control with deep reinforcement learning     |
| A3C       | https://arxiv.org/abs/1602.01783 | Asynchronous Methods for Deep Reinforcement Learning    |
| PPO       | https://arxiv.org/abs/1707.06347 | Proximal Policy Optimization Algorithms                 |
| DPPO      | https://arxiv.org/abs/1707.02286 | Emergence of Locomotion Behaviours in Rich Environments |
| GAE       | https://arxiv.org/abs/1506.02438 | High-Dimensional Continuous Control Using Generalized Advantage Estimation |


- GAE was used in all algorithms except for DPPG
- Where possible, I've added an LSTM layer to the policy and value functions.
This usually made the more complex environments more stable
- DPPO is currently a bit unstable. Work in progress

## Training
All the Python scripts are written as standalone scripts. Just run them
as you would for a single file or in your IDE. The models
and TensorBoard summaries are saved in the same directory as the script.
DPPO has a helper script to set off the worker threads

## Requirements
- Python 3.5+
- OpenAI Gym
- TensorFlow 1.4
- Numpy 1.13+

DPPO was tested on a 16 core machine using CPU only, so the helper
script will need to be updated for your particular setup.
For my setup, there was usually no speed advantage training on the 
CPU vs GPU (GTX 1080), but your performance may differ
