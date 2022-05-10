import numpy as np
import torch
from torch.optim import Adam
import time
from copy import deepcopy
import spinup.algos.pytorch.ppo_lag.core_lag as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from torch.nn.functional import softplus
from spinup.algos.rpsdrl.processing import TrainDRLProcessing

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.crew_buf = np.zeros(size, dtype=np.float32)

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cval_buf = np.zeros(size, dtype=np.float32)

        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, crew, val, cval, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.crew_buf[self.ptr] = crew

        self.val_buf[self.ptr] = val
        self.cval_buf[self.ptr] = cval

        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        crews = np.append(self.crew_buf[path_slice], last_cval)

        vals = np.append(self.val_buf[path_slice], last_val)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        cdeltas = crews[:-1] + self.gamma * cvals[1:] - cvals[:-1]

        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        self.cadv_buf[path_slice] = core.discount_cumsum(cdeltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.cret_buf[path_slice] = core.discount_cumsum(crews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        cadv_mean, cadv_std = mpi_statistics_scalar(self.cadv_buf)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        self.cadv_buf = (self.cadv_buf - cadv_mean)  # / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, cret=self.cret_buf,
                    adv=self.adv_buf, cadv=self.cadv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo_lag(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=256, episodes=30, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4, penalty_lr=5e-2,
        vf_lr=8e-4, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=35040, target_kl=0.01, logger_kwargs=dict(),
        save_freq=1, algo='rp-drl', uncert=False, store_folder='.', filename='ppo_lag_test'):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    assert algo in ['drl', 'rp-drl']

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # Instantiate environment
    env = env_fn
    train_drl_processing = TrainDRLProcessing(algo=algo, uncert=uncert, vani='ppo-lag', store_folder=store_folder, filename=filename)

    obs_dim = train_drl_processing.obs_dim
    act_dim = train_drl_processing.act_dim

    # Create actor-critic module
    act_space = deepcopy(train_drl_processing.act_spa)
    ac = actor_critic(obs_dim, act_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, cadv, logp_old = data['obs'], data['act'], data['adv'], data['cadv'], data['logp']
        cur_cost = data['cur_cost']
        penalty_param = data['cur_penalty']
        cost_limit = 25
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)

        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_rpi = (torch.min(ratio * adv, clip_adv)).mean()

        # clip_cadv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * cadv
        # loss_cpi = (torch.min(ratio * cadv, clip_cadv)).mean()
        loss_cpi = ratio * cadv
        loss_cpi = loss_cpi.mean()

        p = softplus(penalty_param)
        penalty_item = p.item()

        pi_objective = loss_rpi - penalty_item * loss_cpi
        pi_objective = pi_objective / (1 + penalty_item)
        loss_pi = -pi_objective

        cost_deviation = (cur_cost - cost_limit)

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, cost_deviation, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret, cret = data['obs'], data['ret'], data['cret']
        return ((ac.v(obs) - ret) ** 2).mean(), ((ac.vc(obs) - cret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    penalty_param = torch.tensor(1.0, requires_grad=True).float()
    penalty = softplus(penalty_param)

    penalty_optimizer = Adam([penalty_param], lr=penalty_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    cvf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        cur_cost = logger.get_stats('EpCost')[0]
        data = buf.get()
        data['cur_cost'] = cur_cost
        data['cur_penalty'] = penalty_param
        pi_l_old, cost_dev, pi_info_old = compute_loss_pi(data)
        # print(penalty_param)
        loss_penalty = -penalty_param * cost_dev

        penalty_optimizer.zero_grad()
        loss_penalty.backward()
        mpi_avg_grads(penalty_param)
        penalty_optimizer.step()
        # print(penalty_param)

        # penalty = softplus(penalty_param)

        data['cur_penalty'] = penalty_param

        pi_l_old = pi_l_old.item()
        v_l_old, cv_l_old = compute_loss_v(data)
        v_l_old, cv_l_old = v_l_old.item(), cv_l_old.item()

        # Train policy with multiple steps of gradient descent
        train_pi_iters = 80
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, _, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.2 * target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        train_v_iters = 80
        for i in range(train_v_iters):
            loss_v, loss_vc = compute_loss_v(data)
            vf_optimizer.zero_grad()
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

            cvf_optimizer.zero_grad()
            loss_vc.backward()
            mpi_avg_grads(ac.vc)
            cvf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

        return loss_pi, loss_v, loss_vc, loss_penalty


    def test_agent(episode):
        ep_ret, ep_len = 0, 0
        o, _, _, _ = train_drl_processing.env_wrapper(env.reset())

        for i in range(max_ep_len):
            _, _, _, _, action = train_drl_processing.get_action(ac, o, deterministic=True)
            o2, r, d, info = train_drl_processing.env_wrapper(env.step(action))

            ep_ret += r
            ep_len += 1
            o = o2

            if d:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                train_drl_processing.data_store(ep_ret)
                o, _, _, _ = train_drl_processing.env_wrapper(env.reset())
                # Log info about epoch
                logger.log_tabular('Epoch', episode)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.dump_tabular()
                print('****************** Reset Environment and Start Training ******************')
        return ep_ret

    # Prepare for interaction with environment
    epochs = np.int(episodes * max_ep_len / local_steps_per_epoch) + 1
    start_time = time.time()
    episode = 0
    step = 0
    epi_rews = []
    o, _, _, _ = train_drl_processing.env_wrapper(env.reset())
    ep_ret, ep_len, ep_cret = 0, 0, 0

    print("Start Training    Vanilla: PPO_Lag    " + "algorithm: " + algo + "    Uncertainty: " + str(uncert))
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):

            a_theta, v, vc, logp, action = train_drl_processing.get_action(ac, o, deterministic=False)

            o2, r, d, info = train_drl_processing.env_wrapper(env.step(action))

            c = info['cost']
            ep_ret += r
            ep_cret += c
            ep_len += 1

            # Tensorboard monitor.
            train_drl_processing.tensorboard(o, info, action, step)

            # save and log
            buf.store(o, a_theta, r, c, v, vc, logp)
            logger.store(VVals=v)
            logger.store(CVVals=vc)

            # Update obs (critical!)
            o = o2

            step += 1

            terminal = d
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended:
                    _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                    vc = 0
                buf.finish_path(last_val=v, last_cval=vc)

                if epoch_ended:
                    logger.store(EpCost=ep_cret)
                    ep_cret = 0

                if terminal:
                    print('****************** Reset Environment and Start Testing ******************')
                    epi_rews.append(test_agent(episode=episode))
                    train_drl_processing.save_agent(ac, episode)
                    episode += 1

        loss_pi, loss_v, loss_vc, loss_penalty = update()



