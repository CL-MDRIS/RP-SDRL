import numpy as np
import tensorflow as tf
import time
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from spinup.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, mpi_statistics_scalar, num_procs
from spinup.algos.rpsdrl.processing import TrainDRLProcessing
from copy import deepcopy

EPS = 1e-8

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None, dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def get_target_update(main_name, target_name, polyak):
    ''' Get a tensorflow op to update target variables based on main variables '''
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak*targ_vars[v_targ] + (1-polyak)*main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu, name='mu')
    pi = tf.tanh(pi, name='pi')
    return mu, pi, logp_pi

"""
Actors and Critics
"""

def mlp_actor(x, a, name='pi', hidden_sizes=(64, 64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    # policy
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    return mu, pi, logp_pi

def mlp_critic(x, a, pi, name, hidden_sizes=(64, 64), activation=tf.nn.relu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda x: tf.squeeze(mlp(x=x,
                                      hidden_sizes=list(hidden_sizes)+[1],
                                      activation=activation,
                                      output_activation=None),
                                      axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x, a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x, pi], axis=-1))

    return critic, critic_pi


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    costs=self.costs_buf[idxs],
                    done=self.done_buf[idxs])

"""
Soft Actor-Critic
"""
def sac_lag(env_fn, actor_fn=mlp_actor, critic_fn=mlp_critic, ac_kwargs=dict(), seed=0, steps_per_episode=35040, episodes=30,
        replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=8e-4, batch_size=32, local_start_steps=int(0),
        max_ep_len=256, logger_kwargs=dict(), local_update_after=int(0), update_freq=1, render=False,
        fixed_entropy_bonus=None, entropy_constraint=-1.0, fixed_cost_penalty=None, cost_constraint=None, cost_lim=25,
        algo='rp-drl', uncert=False, store_folder='.', filename='sac_lag_test'):

    use_costs = fixed_cost_penalty or cost_constraint or cost_lim
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    assert algo in ['drl', 'rp-drl']

    # Env instantiation
    env = env_fn
    train_drl_processing = TrainDRLProcessing(algo=algo, uncert=uncert, vani='sac-lag', store_folder=store_folder, filename=filename)

    obs_dim = train_drl_processing.obs_dim
    act_dim = train_drl_processing.act_dim

    # Setting seeds
    # tf.set_random_seed(seed)
    # np.random.seed(seed)
    # env.seed(seed)

    # Share information about action space with policy architecture
    act_space = deepcopy(train_drl_processing.act_spa)
    ac_kwargs['action_space'] = act_space

    # Inputs to computation graph

    x_ph = tf.placeholder(name='x_ph', dtype=tf.float32, shape=(None, obs_dim) if obs_dim else (None,))
    a_ph, x2_ph, r_ph, d_ph, c_ph = placeholders(act_dim, obs_dim, None, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs)
        qc, qc_pi = critic_fn(x_ph, a_ph, pi, name='qc', **ac_kwargs)

    with tf.variable_scope('main', reuse=True):
        # Additional policy output from a different observation placeholder
        # This lets us do separate optimization updates (actor, critics, etc)
        # in a single tensorflow op.
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr1', **ac_kwargs)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs)
        _, qc_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qc', **ac_kwargs)

    # Entropy bonus
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha',
                                         initializer=0.0,
                                         trainable=True,
                                         dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    log_alpha = tf.log(alpha)

    # Cost penalty
    if use_costs:
        if fixed_cost_penalty is None:
            with tf.variable_scope('costpen'):
                soft_beta = tf.get_variable('soft_beta',
                                             initializer=0.0,
                                             trainable=True,
                                             dtype=tf.float32)
            beta = tf.nn.softplus(soft_beta)
            log_beta = tf.log(beta)
        else:
            beta = tf.constant(fixed_cost_penalty)
            log_beta = tf.log(beta)
    else:
        beta = 0.0  # costs do not contribute to policy optimization
        print('Not using costs')

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    if proc_id()==0:
        var_counts = tuple(count_vars(scope) for scope in
                           ['main/pi', 'main/qr1', 'main/qr2', 'main/qc', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t qc: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_pi_targ - alpha * logp_pi2))
    qc_backup = tf.stop_gradient(c_ph + gamma*(1-d_ph)*qc_pi_targ)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + beta * qc_pi)
    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1)**2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2)**2)
    qc_loss = 0.5 * tf.reduce_mean((qc_backup - qc)**2)
    q_loss = qr1_loss + qr2_loss + qc_loss

    # Loss for alpha
    entropy_constraint *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    # alpha_loss = - soft_alpha * (entropy_constraint - pi_entropy)
    alpha_loss = - alpha * (entropy_constraint - pi_entropy)
    print('using entropy constraint', entropy_constraint)

    # Loss for beta
    if use_costs:
        if cost_constraint is None:
            # Convert assuming equal cost accumulated each step
            # Note this isn't the case, since the early in episode doesn't usually have cost,
            # but since our algorithm optimizes the discounted infinite horizon from each entry
            # in the replay buffer, we should be approximately correct here.
            # It's worth checking empirical total undiscounted costs to see if they match.
            cost_constraint = cost_lim * (1 - gamma ** max_ep_len) / (1 - gamma) / max_ep_len
        print('using cost constraint', cost_constraint)
        beta_loss = beta * (cost_constraint - qc)

    # Policy train op
    # (has to be separate from value train op, because qr1_pi appears in pi_loss)
    train_pi_op = MpiAdamOptimizer(learning_rate=lr).minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi')

    # Value train op
    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=lr).minimize(q_loss, var_list=get_vars('main/q'), name='train_q')

    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    if use_costs and fixed_cost_penalty is None:
        costpen_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_entreg_op]):
            train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))

    # Polyak averaging for target variables
    target_update = get_target_update('main', 'target', polyak)

    # Single monolithic update with explicit control dependencies
    with tf.control_dependencies([train_pi_op]):
        with tf.control_dependencies([train_q_op]):
            grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])
    if use_costs and fixed_cost_penalty is None:
        grouped_update = tf.group([grouped_update, train_costpen_op])

    # Initializing targets to match main variables
    # As a shortcut, use our exponential moving average update w/ coefficient zero
    target_init = get_target_update('main', 'target', 0.0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc': qc})

    def test_agent(episode):
        o, _, _, _ = train_drl_processing.env_wrapper(env.reset())

        r, d, ep_ret, ep_len = 0, False, 0, 0

        for i in range(steps_per_episode):
            _, action = train_drl_processing.get_action(sess, o, deterministic=True, other=dict(mu=mu, x_ph=x_ph, pi=pi))

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
                logger.log_tabular('EpCost', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('QR1Vals', with_min_and_max=True)
                logger.log_tabular('QR2Vals', with_min_and_max=True)
                logger.log_tabular('QCVals', with_min_and_max=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQR1', average_only=True)
                logger.log_tabular('LossQR2', average_only=True)
                logger.log_tabular('LossQC', average_only=True)
                logger.log_tabular('LossAlpha', average_only=True)
                logger.log_tabular('LogAlpha', average_only=True)
                logger.log_tabular('Alpha', average_only=True)
                if use_costs:
                    logger.log_tabular('LossBeta', average_only=True)
                    logger.log_tabular('LogBeta', average_only=True)
                    logger.log_tabular('Beta', average_only=True)
                logger.dump_tabular()
                print('****************** Reset Environment and Start Training ******************')
        return ep_ret

    # variables to measure in an update
    vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, LossQC=qc_loss,
                       QR1Vals=qr1, QR2Vals=qr2, QCVals=qc, LogPi=logp_pi, PiEntropy=pi_entropy,
                       Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss)
    if use_costs:
        vars_to_get.update(dict(Beta=beta, LogBeta=log_beta, LossBeta=beta_loss))

    print('starting training', proc_id())

    # Main loop: collect experience in env and update/log each epoch
    start_time = time.time()
    saver = tf.train.Saver()
    episode = 0

    epi_rews = []

    o, _, _, _ = train_drl_processing.env_wrapper(env.reset())

    r, d, ep_ret, ep_cost, ep_len = 0, False, 0, 0, 0

    total_steps = steps_per_episode * episodes
    local_steps = 0
    local_steps_per_epoch = steps_per_episode // num_procs()
    local_batch_size = batch_size // num_procs()

    print("Start Training    Vanilla: SAC_Lag    " + "algorithm: " + algo + "    Uncertainty: " + str(uncert))

    for t in range(total_steps // num_procs()):

        a_theta, action = train_drl_processing.get_action(sess, o, deterministic=False, other=dict(mu=mu, x_ph=x_ph, pi=pi))

        # Step the env
        o2, r, d, info = train_drl_processing.env_wrapper(env.step(action))

        c = info['cost']

        train_drl_processing.tensorboard(o, info, action, local_steps)

        ep_cost += c
        local_steps += 1
        # Store experience to replay buffer
        replay_buffer.store(o, a_theta, r, o2, d, c)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if t > 0 and t % update_freq == 0:
            for j in range(update_freq):
                batch = replay_buffer.sample_batch(local_batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             c_ph: batch['costs'],
                             d_ph: batch['done'],
                            }
                if t < local_update_after:
                    logger.store(**sess.run(vars_to_get, feed_dict))
                else:
                    values, _ = sess.run([vars_to_get, grouped_update], feed_dict)
                    logger.store(**values)

        if t % max_ep_len == 0:
            logger.store(EpCost=ep_cost)
            ep_cost = 0

        if d:
            epi_rews.append(test_agent(episode=episode))
            train_drl_processing.save_agent(sess, episode)
            episode += 1

    tf.reset_default_graph()
