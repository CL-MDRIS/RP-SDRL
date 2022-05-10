# Copyright (c) IBM Corp. 2018. All Rights Reserved.
# Project name: Reinforcement Learning Testbed for Power Consumption Optimization
# This project is licensed under the MIT License, see LICENSE

from re import match
import os
from gym_energyplus.envs.CW2_4Zone_COSIM import CW2_4Zone_COSIM

def build_ep_model(model_file, log_dir, verbose = False):
    model_basename = os.path.splitext(os.path.basename(model_file))[0]
    if match('CW2_4Zone_COSIM.*', model_basename):
        model = CW2_4Zone_COSIM(model_file=model_file, log_dir=log_dir, verbose=verbose)
    else:
        raise ValueError('Unsupported EnergyPlus model')
    return model
