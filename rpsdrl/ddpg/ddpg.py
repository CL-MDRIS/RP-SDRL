from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import spinup.algos.pytorch.ddpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.algos.rpsdrl.processing import TrainDRLProcessing

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
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
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

def ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_episode=35040, episodes=10,
         replay_size=int(1e6), gamma=0.99, polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=32, start_steps=0,
         update_after=1, update_every=1, act_noise=1.5, max_ep_len=35040, logger_kwargs=dict(),
         algo='rp-drl', uncert=False, store_folder='.', filename='ddpg_test'):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    assert algo in ['drl', 'rp-drl']

    ## Set random seed.
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    env = env_fn
    train_drl_processing = TrainDRLProcessing(algo=algo, uncert=uncert, vani='ddpg', store_folder=store_folder, filename=filename)

    act_dim = train_drl_processing.act_dim
    obs_dim = train_drl_processing.obs_dim

    # Create actor-critic module and target networks
    act_space = deepcopy(train_drl_processing.act_spa)
    ac = actor_critic(obs_dim, act_dim, act_space, act_noise, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

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
    o,_,_,_ = train_drl_processing.env_wrapper(env.reset())
    ep_ret, ep_len = 0, 0
    episode = 0
    epi_rews = []

    # Main loop: collect experience in env and update/log each epoch
    print("Start Training    Vanilla: DDPG    " + "algorithm: " + algo + "    Uncertainty: " + str(uncert))
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
