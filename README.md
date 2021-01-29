# Meta-Reinforcement Learning for Reliable Communication in THz/VLC Wireless VR Networks


This is the simplified code for the paper ''Meta-Reinforcement Learning for Reliable Communication in THz/VLC Wireless VR Networks''.
This repo is heavily inspired by the fantastic implementation [MoritzTaylor/maml-rl-tf2](https://github.com/MoritzTaylor/maml-rl-tf2).

## Usage
You can use the [`main.py`](main.py) script in order to train the MPG and DMPG algorithms.
```
python main.py --env-name THzVR-v0
```
```
python main.py --env-name DualTHzVR-v0
```
This script was tested with:
Python 3.6
numpy 1.14.0
tensorflow 2.0.0.
