#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines_energyplus.common.energyplus_util import make_energyplus_env, energyplus_arg_parser, energyplus_logbase_dir
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import os
import shutil
import datetime
import gym_energyplus
from spinup.algos.rpsdrl.sac import sac
from spinup.algos.rpsdrl.ddpg import ddpg
from spinup.algos.rpsdrl.ppo import ppo
from spinup.algos.rpsdrl.ppo_lag import ppo_lag
from spinup.algos.rpsdrl.sac_lag import sac_lag
from spinup.algos.rpsdrl.deploy import deploy


def train(env_id, choice=None, vani=None, algo=None, uncert=None, store_folder=None, filename=None, predictor=None,
          seed=None):
    # Create a new base directory like /tmp/openai-2018-05-21-12-27-22-552435
    log_dir = os.path.join(energyplus_logbase_dir(), datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    if not os.path.exists(log_dir + '/output'):
        os.makedirs(log_dir + '/output')
    os.environ["ENERGYPLUS_LOG"] = log_dir
    model = os.getenv('ENERGYPLUS_MODEL')
    if model is None:
        print('Environment variable ENERGYPLUS_MODEL is not defined')
        os.exit()
    weather = os.getenv('ENERGYPLUS_WEATHER')
    if weather is None:
        print('Environment variable ENERGYPLUS_WEATHER is not defined')
        os.exit()
    print('train: init logger with dir={}'.format(log_dir))
    logger.configure(log_dir)

    env = make_energyplus_env(env_id)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs("try_safe_0", 0)

    if choice == 'train':
        if vani == "sac":
            sac.sac(env, algo=algo, uncert=uncert, store_folder=store_folder, filename=filename,
                    logger_kwargs=logger_kwargs)
        elif vani == 'ddpg':
            ddpg.ddpg(env, algo=algo, uncert=uncert, store_folder=store_folder, filename=filename,
                      logger_kwargs=logger_kwargs)
        elif vani == 'ppo':
            ppo.ppo(env, algo=algo, uncert=uncert, store_folder=store_folder, filename=filename,
                    logger_kwargs=logger_kwargs)
        elif vani == 'sac-lag':
            sac_lag.sac_lag(env, algo=algo, uncert=uncert, store_folder=store_folder, filename=filename,
                            logger_kwargs=logger_kwargs)
        elif vani == 'ppo-lag':
            ppo_lag.ppo_lag(env, algo=algo, uncert=uncert, store_folder=store_folder, filename=filename,
                            logger_kwargs=logger_kwargs)
    elif choice == 'deploy':
        deploy.deploy(env, vani=vani, algo=algo, uncert=uncert, predictor=predictor, seed=seed,
                      logger_kwargs=logger_kwargs)

    env.close()

def main():
    args = energyplus_arg_parser().parse_args()    # Get the command from the command window and analyze it

    # The number of random training.
    seed_num = 5
    store_folder = '.'

    # Input required settings.
    try:
        choice = input('agent and prediction model should have been trained if deploy (train or deploy): ')
    except ValueError:
        print("please input full name")

    if choice == 'train':
        vanilla = input('please select an vanilla algorithm (sac, ddpg, ppo, sac-lag, ppo-lag): ')
        algo = input('do you want to use residual physics (drl or rp-drl): ')
        uncertainty = input('do you want to add state Uncertainty (yes or no): ')
        if uncertainty == "yes":
            uncert = True
        elif uncertainty == "no":
            uncert = False
        else:
            uncert = None
            assert uncert is None

        for i in range(seed_num):
            train(args.env, choice=choice, vani=vanilla, algo=algo, uncert=uncert, store_folder=store_folder,
                  filename='seed' + str(i + 1) + '-')

    elif choice == 'deploy':
        for i in range(3):
            print('************** Please Check the Agent Path and NN Path in the Deploy File **************')
        vanilla = input('please select an vanilla algorithm (sac, ddpg, ppo, sac-lag, ppo-lag): ')
        algo = input('do you want to use residual physics (drl or rp-drl): ')
        predictor = input('the type of shielding (prenn, rp-prenn, physics, none): ')
        uncertainty = input('do you want to add state Uncertainty (yes or no): ')
        seed = input('the seed number of training (1~5): ')
        if uncertainty == "yes":
            uncert = True
        elif uncertainty == "no":
            uncert = False
        else:
            uncert = None
            assert uncert is None

        train(args.env, choice=choice, vani=vanilla, algo=algo, uncert=uncert, store_folder=store_folder,
              filename='deploy', predictor=predictor, seed=seed)


if __name__ == '__main__':
    main()

