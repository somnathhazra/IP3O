# Incentivizing Safer Actions in Policy Optimization for Constrained Reinforcement Learning

Code for Incrementally Penalized Proximal Policy Optimization (IP3O) algorithm accepted at IJCAI 2025.

This repository is based on [OmniSafe](https://github.com/PKU-Alignment/omnisafe).

# Installation Guide

Create conda env: ```conda create -n ip3o python=3.8``` 

Activate conda env: ```conda activate ip3o```

```
git clone https://github.com/PKU-Alignment/omnisafe.git
cd omnisafe
```

Install Omnisafe

Register the algorithm file `ip3o.py` from `omnisafe/algorithms/on_policy/penalty_function` in omnisafe.
Change the `__init__.py` file in the respective omnisafe library.

Register the configuration file for IP3O provided in `omnisafe/configs/on-policy` in omnisafe.

# Run Experiments

Run the code by:
```
cd examples
python train_policy.py --algo IP3O --env-id SafetyPointGoal1-v0 --parallel 1 --total-steps 2500000 --device cuda:0 --vector-env-nums 1 --torch-threads 16
```

All results will be stored in the `examples/runs` folder.
