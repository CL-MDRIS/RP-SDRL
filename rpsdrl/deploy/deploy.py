import torch
import os
import tensorflow as tf
from spinup.utils.logx import EpochLogger
from spinup.algos.rpsdrl.processing import TrainDRLProcessing, PreNNProce, Shielding
from spinup.algos.rpsdrl.prediction import core as core_pre
from spinup.algos.rpsdrl.ddpg import core as core_ddpg
from spinup.algos.rpsdrl.ppo import core as core_ppo
from spinup.algos.rpsdrl.ppo_lag import core_lag as core_ppo_lag
from spinup.algos.rpsdrl.sac import core as core_sac
import spinup.algos.rpsdrl.prediction as prediction

"""
This file will load corresponding neural networks according to the requirements. 
"""

def import_net(vani=None, predictor=None, uncert=None, obs_dim=None, act_space=None, algo=None, seed=None):

    # Download prediction neural networks.
    predictor_file = 'pureprenn' if predictor == 'prenn' else predictor
    if predictor == 'prenn' or predictor == 'rp-prenn':
        nn_folder_ori = prediction.__file__
        nn_folder = nn_folder_ori.replace('__init__.py', 'nn_model')
        nn_file_key = predictor_file + str(uncert)
        for f in os.listdir(nn_folder):
            if nn_file_key in f:
                path = nn_folder + '/' + f
                for pth in os.listdir(path):
                    if "step_10000000" in pth:
                        nn_path = path + '/' + pth

        nn_input_dim = 14 if predictor == 'prenn' else 18
        nn_out_dim, nn_hidden_sizes = 4, (128, 128, 128)
        nn_net = core_pre.PreNN(in_dim=nn_input_dim, out_dim=nn_out_dim, hidden_sizes=nn_hidden_sizes)
        nn_net_state_dict = torch.load(nn_path)
        nn_net.load_state_dict(nn_net_state_dict['model'])
    else:
        nn_net = None

    # Download agent policy networks.
    aget_net_state_dict, agent_path_sac_lag, agent_path = None, None, None

    agent_key = vani + algo + str(uncert) + 'seed' + seed
    agents_folers = os.getcwd() + '/drl_store/agent_store'
    for f_agent in os.listdir(agents_folers):
        if agent_key in f_agent:
            agent_foler = agents_folers + '/' + f_agent
            for agent_file in os.listdir(agent_foler):
                if "episode_29" in agent_file:
                    agent_path = agent_foler + '/' + agent_file

    if vani == 'sac-lag':
        agent_path_sac_lag = agent_path
    else:
        agent_path = agent_path
        aget_net_state_dict = torch.load(agent_path)

    if vani == 'sac':
        agent_net = core_sac.MLPActorCritic(obs_dim, act_space[0])
        agent_net.load_state_dict(aget_net_state_dict['model'])
        return nn_net, agent_net, None
    elif vani == 'sac-lag':
        with tf.gfile.FastGFile(agent_path_sac_lag, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            x_ph, mu, pi = tf.import_graph_def(graph_def, return_elements=["x_ph:0", "main/pi/mu:0", "main/pi/pi:0"])
            sess = tf.Session()
        return nn_net, sess, {'x_ph': x_ph, 'mu': mu, 'pi': pi}
    elif vani == 'ppo':
        agent_net = core_ppo.MLPActorCritic(obs_dim, act_space)
        agent_net.load_state_dict(aget_net_state_dict['model'])
        return nn_net, agent_net, None
    elif vani == 'ppo-lag':
        agent_net = core_ppo_lag.MLPActorCritic(obs_dim, act_space)
        agent_net.load_state_dict(aget_net_state_dict['model'])
        return nn_net, agent_net, None
    elif vani == 'ddpg':
        agent_net = core_ddpg.MLPActorCritic(obs_dim, act_space[0], act_space)
        agent_net.load_state_dict(aget_net_state_dict['model'])
        return nn_net, agent_net, None


def deploy(env_fn, vani=None, algo='rp-drl', uncert=False, predictor=None, max_ep_len=35040, seed='1',
           ac_kwargs=dict(), logger_kwargs=dict()):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    assert vani in ['sac', 'sac-lag', 'ppo', 'ppo-lag', 'ddpg']
    assert algo in ['drl', 'rp-drl']
    assert predictor in ['prenn', 'rp-prenn', 'physics', 'none']

    train_drl_processing = TrainDRLProcessing(algo=algo, uncert=uncert, vani=vani)
    prenn_proce = PreNNProce(solution=predictor, uncert=uncert)

    nn_net, agent_net, other = import_net(vani=vani, predictor=predictor, uncert=uncert, obs_dim=train_drl_processing.obs_dim,
                        act_space=train_drl_processing.act_spa, algo=algo, seed=seed)

    shielding = Shielding(vani=vani, solution=predictor, uncert=uncert, train_class=train_drl_processing,
                          prenn_class=prenn_proce, ac_net=agent_net, prenn_net=nn_net, act_spa=env_fn.action_space,
                          other=other)

    # Instantiate environment
    power_rew_rev = -1000
    env = env_fn
    episode = 0
    ep_ret, ep_len = 0, 0
    vio_total = 0
    elec_total = 0
    o, _, _, info = train_drl_processing.env_wrapper(env.reset())

    for i in range(max_ep_len):

        action, vio_num = shielding.correction(o, info=info)

        o2, r, d, info = train_drl_processing.env_wrapper(env.step(action))

        vio_total += vio_num
        elec_total += info['reward_part'][3] * power_rew_rev

        ep_ret += r
        ep_len += 1
        o = o2

        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, _, _, info = train_drl_processing.env_wrapper(env.reset())
            # Log info about epoch
            logger.log_tabular('Epoch', episode)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.dump_tabular()
            print('********************************************************************')
            print("Violation number: " + str(vio_total) + "   Violation percentage: " +
                  str(format(vio_total/((i+1)*4)*100, '.2f')) + '%')
            print('Annual Electricity: ', elec_total/4)

