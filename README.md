# Project Description

This project integrates Residual Physics with Deep Reinforcement Learning algorithms for the dynamic thermal management 
of data centers. The tested algorithms include: SAC, PPO, DDPG, SAC-Lagrangian, and PPO-Lagrangian.

# Platforms

The proposed framework has been tested in Ubuntu 18.04 LTS, Python 3.6, EnergyPlus 9.3.0.

# Requirements


## 1. Testbed Preparation
This project utilized the OpenAI Env wrapper developed by Moriyama et al. The user should intall the wrapper according
to the instructions in:
[EnergyPlus Testbed](https://github.com/IBM/rl-testbed-for-energyplus).
The EnergyPlus version is 9.3.0. <br />

Please put the weather data, and the EnergyPlus model named _CW2_4Zone_COSIM_ in the folder _EnergyPlusModels_ to the 
installation path according to the instructions of testbed. Please remember to modify the environment variables in
_$(HOME)/.bashrc_. <br />

Copy the _baselines_energyplus_ and _gym_energyplus_ to the same name folders under the testbed. 

## 2. Spinningup
We refer to the code structure of [Spinningup](https://github.com/openai/spinningup). Please copy the folder
_rpsdrl_ to the installation path: _spinningup/spinup/algos_. 

# Run

Under the path: _rl-testbed-for-energyplus/baselines_energyplus/rp_sdrl_: <br />
`python3 run_energyplus.py` <br />

You can select the following choices according to the instruction in the terminal:
- Train or deploy an algorithm. 
- Choose one vanilla algorithm: SAC, SAC-Lagrangian, PPO, PPO-Lagrangian, DDPG, and PPO.
- Use Residual Physics or not.
- Add state uncertainty or not.
- The type of Post-Post Shielding. <br />

# Other References
- [PPO-Lag](https://github.com/akjayant/PPO_Lagrangian_PyTorch)
- [SAC-Lag](https://github.com/openai/safety-starter-agents)

The project is still ongoing. :-)







