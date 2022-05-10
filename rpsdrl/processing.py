import numpy as np
import torch
from gym import spaces
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import csv
import datetime
import tensorflow as tf
from copy import deepcopy


def data_time():
    now = datetime.datetime.now()
    return str(now.strftime("%m%d%H%M"))


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
    else:
        print("---  There is this folder!  ---")


class TrainDRLProcessing:
    def __init__(self, algo=None, uncert=False, vani=None, store_folder=None, filename=None):
        """
        This class is used to pre-process the data from the environment, output action, store and display results/agents.

        Terms in state vector:[Environment temperature, East zone temperature HVAC1, West zone temperature HVAC1,
        East zone temperature HVAC2, West zone temperature HVAC2, HVAC power, East zone HVAC1 ITE power, West zone HVAC1
        ITE power, East zone HVAC2 ITE power, West zone HVAC2 ITE power]

        Terms in action vector:[HVAC1 supply air temperature, HVAC1 supply air mass flow rate, HVAC2 supply air
        temperature, HVAC2 supply air mass flow rate]
        """

        self.T_p = 18 # Fixed supply air temperature.
        self.T_target = 25 # Target zone temperature.
        self.Tz_lb, self.Tz_ub = 24, 26 # The lower and upper temperature constraints.
        self.c_p = 1.005 # Constant pressure specific heat capacity of air. Units kj/(kg*k)
        self.act_dim = 4
        self.Ts_lb, self.Ts_ub, self.flow_lb, self.flow_ub = 10, 25, 64*0.25, 64
        self.algo, self.vani, self.uncert = algo, vani, uncert

        if self.algo == 'drl' or self.algo == 'physics':
            lo = self.Ts_lb    # Limited by the capacity of the chiller.
            hi = self.Ts_ub    # T_target is 25℃. Set-points higher than 25℃ is meaningless.
            flow_lo = self.flow_lb  # A lower limitation is necessary for safety.
            flow_hi = self.flow_ub    # Limited by the maximum pump flow rate.
            self.obs_dim = 10
        elif self.algo == 'rp-drl':
            lo = self.Ts_lb - self.T_p   # To reach the lower bound of the original action space.
            hi = self.Ts_ub - self.T_p     # To reach the upper bound of the original action space.
            flow_lo = -5  # To compensate the calculated flow rate.
            flow_hi = 5   # To compensate the calculated flow rate.
            self.obs_dim = 10 + int(self.act_dim/2) # T_p is a fixed value. No need to feed into NN.

        self.act_spa = spaces.Box(low=np.array([lo, flow_lo, lo, flow_lo]),
                                  high=np.array([hi, flow_hi, hi, flow_hi]),
                                  dtype=np.float32)

        self.data_time = data_time()

        if store_folder is not None and filename is not None:
            # File path to store trained agents.
            self.agent_path = store_folder + '/drl_store/agent_store/' + self.vani + self.algo + str(self.uncert) + \
                              filename + self.data_time
            mkdir(self.agent_path)

            # File path of tensorboard runs.
            self.runs_path = store_folder+'/drl_store/runs/' + self.vani + self.algo + str(self.uncert) + filename + \
                             self.data_time
            mkdir(self.runs_path)
            self.writer = SummaryWriter(self.runs_path)

            # File path to store data.
            self.file_path = store_folder+'/drl_store/data_store/' + self.vani + self.algo + str(self.uncert) + \
                             filename + self.data_time
            mkdir(self.file_path)
            f = open(self.file_path + '/' + self.vani + self.algo + filename + 'step' + '.csv', 'w')
            self.csv_write_step = csv.writer(f)
            self.csv_write_step.writerow(['Reward'])

    def T_upboud(self, Q_1, Q_2):

        # Provide initial supply air temperature range (optional function).
        Ts_up_1 = self.T_target - Q_1 / (self.c_p * self.flow_ub)
        Ts_low_1 = self.T_target - Q_1 / (self.c_p * self.flow_lb)
        Ts_up_2 = self.T_target - Q_2 / (self.c_p * self.flow_ub)
        Ts_low_2 = self.T_target - Q_2 / (self.c_p * self.flow_lb)
        return np.array([Ts_low_1, Ts_low_2]), np.array([Ts_up_1, Ts_up_2])

    def a_theta(self, ac_net, o, deterministic=False, other=None):

        # To provide the output of policy networks.
        if self.vani == 'sac':
            a_theta = ac_net.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
            a_theta_scale = self.act_spa.low + (a_theta + 1.) * 0.5 * (self.act_spa.high - self.act_spa.low)
            return a_theta, a_theta_scale

        elif self.vani == 'ddpg':
            a_theta_scale = ac_net.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
            return a_theta_scale, a_theta_scale

        elif self.vani == 'ppo':
            a_theta, v, logp = ac_net.step(torch.as_tensor(o, dtype=torch.float32))
            a_theta_scale = np.clip(a_theta, [-1, -1, -1, -1], [1, 1, 1, 1])
            a_theta_scale = self.act_spa.low + (a_theta_scale + 1.) * 0.5 * (self.act_spa.high - self.act_spa.low)
            return a_theta, v, logp, a_theta_scale

        elif self.vani == 'ppo-lag':
            a_theta, v, vc, logp = ac_net.step(torch.as_tensor(o, dtype=torch.float32))
            a_theta_scale = np.clip(a_theta, [-1, -1, -1, -1], [1, 1, 1, 1])
            a_theta_scale = self.act_spa.low + (a_theta_scale + 1.) * 0.5 * (self.act_spa.high - self.act_spa.low)
            return a_theta, v, vc, logp, a_theta_scale

        elif self.vani == 'sac-lag':
            act_op = other['mu'] if deterministic else other['pi']
            a_theta = ac_net.run(act_op, feed_dict={other['x_ph']: o.reshape(1, -1)})[0]
            a_theta_scale = self.act_spa.low + (a_theta + 1.) * 0.5 * (self.act_spa.high - self.act_spa.low)
            return a_theta, a_theta_scale

        else:
            a_theta, a_theta_scale = None, None
            assert a_theta is not None

    def a_p(self, o):

        # Calculate the cooling load of a specific HVAC, units kw.
        Q_1 = o[6] + o[7]
        Q_2 = o[8] + o[9]

        # Calculate the required flow rate.
        m_1 = np.clip(Q_1 / (self.c_p * (self.T_target - self.T_p + 1e-8)), self.flow_lb, self.flow_ub)
        m_2 = np.clip(Q_2 / (self.c_p * (self.T_target - self.T_p + 1e-8)), self.flow_lb, self.flow_ub)
        a_p = np.array([self.T_p, m_1, self.T_p, m_2])

        return a_p

    def a_p_theta_core(self, o, a_theta_scale):
        """
        The supply temperature of the baseline controller is T_p. Then T_p + T_theta enables a full exploration of the
        supply temperature range. Thanks to this, Physics will calculate an initial prediction of flow rate and DRL
        will learn the compensation.
        """
        # Calculate the cooling load of a specific HVAC, units kw.
        Q_1 = o[6] + o[7]
        Q_2 = o[8] + o[9]

        # Compensate the fixed supply temperature.
        T_p_theta_1 = self.T_p + a_theta_scale[0]
        T_p_theta_2 = self.T_p + a_theta_scale[2]

        # Compensate the calculated flow rate.
        m_p_theta_1 = np.clip(Q_1 / (self.c_p * (self.T_target - T_p_theta_1 + 1e-8)), self.flow_lb, self.flow_ub) + \
                      a_theta_scale[1]
        m_p_theta_2 = np.clip(Q_2 / (self.c_p * (self.T_target - T_p_theta_2 + 1e-8)), self.flow_lb, self.flow_ub) + \
                      a_theta_scale[3]

        a_p_theta = np.array([T_p_theta_1, m_p_theta_1, T_p_theta_2, m_p_theta_2])

        return a_p_theta

    def get_action(self, ac_net, o, deterministic=False, other=None):

        # To ouptput actions.
        if self.vani == 'sac' or self.vani == 'ddpg':
            a_theta, a_theta_scale = self.a_theta(ac_net, o, deterministic=deterministic)
            if self.algo == 'drl':
                return a_theta, a_theta_scale
            elif self.algo == 'rp-drl':
                a_p_theta = self.a_p_theta_core(o, a_theta_scale)
                return a_theta, a_p_theta

        elif self.vani == 'ppo':
            a_theta, v, logp, a_theta_scale = self.a_theta(ac_net, o, deterministic=deterministic)
            if self.algo == 'drl':
                return a_theta, v, logp, a_theta_scale
            elif self.algo == 'rp-drl':
                a_p_theta = self.a_p_theta_core(o, a_theta_scale)
                return a_theta, v, logp, a_p_theta

        elif self.vani == 'ppo-lag':
            a_theta, v, vc, logp, a_theta_scale = self.a_theta(ac_net, o, deterministic=deterministic)
            if self.algo == 'drl':
                return a_theta, v, vc, logp, a_theta_scale
            elif self.algo == 'rp-drl':
                a_p_theta = self.a_p_theta_core(o, a_theta_scale)
                return a_theta, v, vc, logp, a_p_theta

        elif self.vani == 'sac-lag':

            a_theta, a_theta_scale = self.a_theta(ac_net, o, deterministic=deterministic, other=other)
            if self.algo == 'drl':
                return a_theta, a_theta_scale
            elif self.algo == 'rp-drl':
                a_p_theta = self.a_p_theta_core(o, a_theta_scale)
                return a_theta, a_p_theta

        else:
            a_theta = None
            assert a_theta is None

    def st_uncert(self, o):

        # Add state uncertainties if required.
        state_uncert = np.array([np.random.normal(0, 0.5), np.random.normal(0, 0.5), np.random.normal(0, 0.5),
                                np.random.normal(0, 0.5), np.random.normal(0, 0.5),
                                np.random.normal(0, o[5]/10), np.random.normal(0, o[6]/10), np.random.normal(0, o[7]/10),
                                np.random.normal(0, o[8]/10), np.random.normal(0, o[9]/10)])
        return state_uncert

    def cost_function(self, o):
        # Calculate violation cost for Lagrangian-based solutions.
        T_zone = o[1:5]
        c_1 = np.where(T_zone >= self.Tz_lb, 0, 1)
        c_2 = np.where(T_zone <= self.Tz_ub, 0, 1)
        c = 1 if np.sum(c_1 + c_2) > 0 else 0
        return c

    def env_wrapper(self, tuple):
        # Pre-processing the feedbacks from E+, before sending to the agent.
        o, r, d, info = tuple[0], tuple[1], tuple[2], tuple[3]

        if self.vani == 'ppo-lag' or self.vani == 'sac-lag':
            c = self.cost_function(o)
            info['cost'] = c

        # Add uncertainty if required.
        if self.uncert:
            st_uncert = self.st_uncert(o)
            o += st_uncert
            info['st_uncert'] = st_uncert

        # Add two more states if rp-drl is required.
        if self.algo == 'rp-drl':
            a_p = self.a_p(o)
            # T_p is a fixed value. No need to feed into NN.
            a_p = np.array([a_p[1], a_p[3]])
            o = np.concatenate((o, a_p))

        return o, r, d, info

    def save_agent(self, ac_net, episode):
        if self.vani == 'sac-lag':
            graph_def = tf.get_default_graph().as_graph_def()
            var_list = ["x_ph", "main/pi/mu", "main/pi/pi"]
            constant_graph = tf.graph_util.convert_variables_to_constants(ac_net, graph_def, var_list)
            with tf.gfile.FastGFile(self.agent_path + '/episode_' + str(episode) + ".pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())
        else:
            torch.save({'model': ac_net.state_dict()}, self.agent_path+'/episode_' + str(episode) + '.pth')

    def tensorboard(self, o, info, action, step):
        st = o[:10] - info['st_uncert'] if self.uncert else o
        self.writer.add_scalar('data/T_env', st[0], step)
        self.writer.add_scalar('data/Tz_west1', st[1], step)
        self.writer.add_scalar('data/Tz_east1', st[2], step)
        self.writer.add_scalar('data/Tz_west2', st[3], step)
        self.writer.add_scalar('data/Tz_east2', st[4], step)
        self.writer.add_scalar('data/Total_power', st[5]+st[6]+st[7]+st[8]+st[9], step)
        self.writer.add_scalar('data/T_1', action[0], step)
        self.writer.add_scalar('data/m_1', action[1], step)
        self.writer.add_scalar('data/T_2', action[2], step)
        self.writer.add_scalar('data/m_2', action[3], step)
        if self.vani == 'ppo-lag' or self.vani == 'sac-lag':
            self.writer.add_scalar('data/cost', info['cost'], step)

    def data_store(self, reward):
        self.csv_write_step.writerow([reward])

    def plot(self, rewards):
        plt.figure()  # Create a figure
        plt.title("Episode Reward", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=10, direction='in')
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Episode Rewards", fontsize=12)
        plt.plot(np.arange(len(rewards)), rewards, color='b')
        plt.legend(loc='best')
        plt.show()


class PreNNProce:
    """
    This class is to process the inputs and utilize physics for prediction models.
    """
    def __init__(self, solution=None, uncert=False, store_folder=None, filename=None):
        self.solution = solution
        self.uncert = uncert
        self.delta_t = 15*60 # Time step, units second.
        self.c_p = 1.005 # Constant pressure specific heat capacity of air. Units kj/(kg*k)
        self.air_den = 1.293 # Air density. Units kg/m^3.
        self.v_zone = 15*15*4.5 # Zone volume. Units m^3.
        self.c_z = self.air_den * self.c_p * self.v_zone

        self.data_time = data_time()

        if store_folder is not None and filename is not None:
            # File path to store neural networks.
            self.folder = 'pure' + self.solution if self.solution == 'prenn' else self.solution

            self.nn_path = store_folder + '/nn_model/' + self.folder + str(self.uncert) + filename + self.data_time
            mkdir(self.nn_path)

            # File path to store tensorboard runs.
            self.runs_path = store_folder+'/runs/' + self.folder + str(self.uncert) + filename + self.data_time
            mkdir(self.runs_path)
            self.writer = SummaryWriter(self.runs_path)

            # File path to store data.
            self.file_path = store_folder+'/data_store/' + self.folder + str(self.uncert) + filename + self.data_time
            mkdir(self.file_path)
            f = open(self.file_path + '/' + self.solution + str(self.uncert) + filename + 'step' + '.csv', 'w')
            self.csv_write_step = csv.writer(f)
            self.csv_write_step.writerow(['Loss'])

    def pred_phy(self, o, a):
        # Calculate the zone temperature in the next time step by using physics.
        num = o[:, 6:10] + np.concatenate((a[:, [1, 1]], a[:, [3, 3]]), axis=1) * 0.5 * self.c_p * np.concatenate(
            (a[:, [0, 0]], a[:, [2, 2]]), axis=1) + self.c_z / self.delta_t * o[:, 1:5]

        dem = self.c_z / self.delta_t + np.concatenate((a[:, [1, 1]], a[:, [3, 3]]), axis=1) * 0.5 * self.c_p

        T_pred = num / dem
        o_com = np.concatenate((o, a), axis=1)

        return o_com, T_pred

    def input_proc(self, data):

        # Return the processed inputs and lables.
        o, o_uncert, a, o2 = data['o'], data['o_uncert'], data['a'], data['o2']

        if self.solution == 'prenn':
            o += o_uncert if self.uncert else 0
            x = np.concatenate((o, a), axis=1)
            lable = o2[:, 1:5]
            batch = dict(x=x, lable=lable)
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

        elif self.solution == 'rp-prenn':
            o += o_uncert if self.uncert else 0
            o_com, T_pred = self.pred_phy(o, a)
            x = np.concatenate((o_com, T_pred), axis=1)
            lable = o2[:, 1:5]
            batch = dict(x=x, lable=lable, predict=T_pred)
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
        else:
            pass

    def pre_dep(self, o, a):

        # Called during deployment stage.
        x_prenn = np.concatenate((o, a), axis=1)
        o_com, T_pred = self.pred_phy(o, a)
        x_rpprenn = np.concatenate((o_com, T_pred), axis=1)
        batch = dict(x_prenn=x_prenn, x_rpprenn=x_rpprenn, T_phy=T_pred)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def tensorboard(self, loss, step):
        self.writer.add_scalar('data/loss', loss, step)

    def save_nn(self, nn_net, step):
        torch.save({'model': nn_net.state_dict()}, self.nn_path + '/step_' + str(step) + '.pth')

    def data_store(self, loss):
        self.csv_write_step.writerow([loss])


class Shielding:
    """
    This class will be activated during deployment stage. The function is to avoid violations.
    """
    def __init__(self, vani=None, solution=None, uncert=False, train_class=None, prenn_class=None, ac_net=None,
                 prenn_net=None, act_spa=None, other=None):

        self.vani, self.solution, self.uncert, self.other, self.act_spa = vani, solution, uncert, other, act_spa
        self.train_class, self.prenn_class = train_class, prenn_class
        self.ac_net, self.prenn_net = ac_net, prenn_net

        # The maximum iteration steps
        self.max_iter = 10000
        # The lower and upper temperature constraints.
        self.Tz_lb, self.Tz_ub = 24, 26

        self.act_high = act_spa.high
        self.act_low = act_spa.low
        self.act_range = self.act_high - self.act_low

        # Step size of the correction model.
        self.n = 100
        self.step_size = self.act_range/self.n

        # The safety tolerance to consider the imperfection of the prediction model.
        self.toler = 0.6 if self.uncert else 0.3

    def T_cal(self, data):
        # To calculate the predicted actions.
        if self.solution == 'physics':
            return data['T_phy'].detach().numpy()
        elif self.solution == 'prenn':
            return self.prenn_net.prenn(data['x_prenn']).detach().numpy()
        elif self.solution == 'rp-prenn':
            return (self.prenn_net.prenn(data['x_rpprenn']) + data['T_phy']).detach().numpy()

    def flag_checker(self, T_HVAC):

        # To judge the flag.
        if T_HVAC < self.Tz_lb + self.toler:
            flag_HVAC = 1

        elif T_HVAC > self.Tz_ub - self.toler:
            flag_HVAC = -1
        else:
            flag_HVAC = 0

        return int(flag_HVAC)

    def T_check(self, T_pred):

        # To return the flag.
        T_HVAC_1, T_HVAC_2 = np.mean(T_pred[:2]), np.mean(T_pred[2:4])
        flag_HVAC_1, flag_HVAC_2 = self.flag_checker(T_HVAC_1), self.flag_checker(T_HVAC_2)

        return np.array([flag_HVAC_1, -flag_HVAC_1, flag_HVAC_2, -flag_HVAC_2])

    def vio_count(self, obs, info):
        o = deepcopy(obs)
        if self.uncert:
            o[1:5] -= info['st_uncert'][1:5] # Remove the uncertainty to count actual violations.
        vio = 0
        for item in o[1:5]:
            if item < self.Tz_lb or item > self.Tz_ub:
                vio += 1
        return vio

    def correction(self, obs, info=None):

        vio_num = self.vio_count(obs, info)

        # To return unsafe actions.
        if self.vani == 'sac' or self.vani == 'ddpg':
            _, action = self.train_class.get_action(self.ac_net, obs, deterministic=True)
        elif self.vani == 'sac-lag':
            _, action = self.train_class.get_action(self.ac_net, obs, deterministic=True, other=self.other)
        elif self.vani == 'ppo':
            _, _, _, action = self.train_class.get_action(self.ac_net, obs, deterministic=True)
        elif self.vani == 'ppo-lag':
            _, _, _, _, action = self.train_class.get_action(self.ac_net, obs, deterministic=True)
        else:
            action = None
            assert action is None

        if self.solution == 'none':
            pass
        else:
            for i in range(self.max_iter):
                data = self.prenn_class.pre_dep(np.expand_dims(obs[:10], axis=0), np.expand_dims(action, axis=0))
                T_pred = self.T_cal(data)
                flag = self.T_check(T_pred.squeeze())
                flag_sum = np.sum(np.abs(flag))

                if flag_sum == 0:
                    break
                else:
                    action = np.clip(action + self.step_size * flag, self.act_low, self.act_high)

        return action, vio_num






