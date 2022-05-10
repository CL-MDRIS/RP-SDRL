from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import spinup.algos.pytorch.sac.core as core
from spinup.utils.logx import EpochLogger
from spinup.algos.rpsdrl.processing import TrainDRLProcessing

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_episode=35040, episodes=10,
        replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=8e-4, alpha=0.1, batch_size=32, start_steps=0, update_after=0,
        update_every=1, max_ep_len=35040, logger_kwargs=dict(), save_freq=1, replay_size_pred=int(1e6), algo='rp-drl',
        uncert=False, store_folder='.',filename='sac_test'):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    assert algo in ['drl', 'rp-drl']

    env = env_fn
    train_drl_processing = TrainDRLProcessing(algo=algo, uncert=uncert, vani='sac', store_folder=store_folder, filename=filename)

    act_dim = train_drl_processing.act_dim
    obs_dim = train_drl_processing.obs_dim

    # Create actor-critic module and target networks

    ac = actor_critic(obs_dim, act_dim, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def test_agent(episode):
        ep_ret, ep_len = 0, 0
        o, _, _, _ = train_drl_processing.env_wrapper(env.reset())

        for i in range(steps_per_episode):

            a_theta, action = train_drl_processing.get_action(ac, o, deterministic=True)
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
    total_steps = steps_per_episode * episodes
    start_time = time.time()
    o, _, _, _ = train_drl_processing.env_wrapper(env.reset())
    ep_ret, ep_len = 0, 0
    episode = 0
    epi_rews = []
    # Main loop: collect experience in env and update/log each epoch
    print("Start Training    Vanilla: SAC    " + "algorithm: " + algo + "    Uncertainty: " + str(uncert))

    for t in range(total_steps):
        a_theta, action = train_drl_processing.get_action(ac, o, deterministic=False)
        # Get new information of the next step.
        o2, r, d, info = train_drl_processing.env_wrapper(env.step(action))
        # Tensorboard monitor.
        train_drl_processing.tensorboard(o, info, action, t)

        replay_buffer.store(o, a_theta, r, o2, d)
        o = o2

        if d:
            print('****************** Reset Environment and Start Testing ******************')
            epi_rews.append(test_agent(episode=episode))
            train_drl_processing.save_agent(ac, episode)
            episode += 1

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)




