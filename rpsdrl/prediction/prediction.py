import numpy as np
import torch
import torch.nn as nn
from spinup.algos.rpsdrl.processing import PreNNProce
import core

class Dataset:
    def __init__(self,):
        # Import train data.
        self.o_train = np.load("./dataset/train/o_data_train.npy")
        self.o_uncert_train = np.load("./dataset/train/o_data_uncert_train.npy")
        self.a_train = np.load("./dataset/train/a_data_train.npy")
        self.o2_train = np.load("./dataset/train/o2_data_train.npy")

        # Import test data
        self.o_test = np.load("./dataset/test/o_data_test.npy")
        self.o_uncert_test = np.load("./dataset/test/o_data_uncert_test.npy")
        self.a_test = np.load("./dataset/test/a_data_test.npy")
        self.o2_test = np.load("./dataset/test/o2_data_test.npy")

        # Import validate data
        self.o_validation = np.load("./dataset/validation/o_data_validation.npy")
        self.o_uncert_validation = np.load("./dataset/validation/o_data_uncert_validation.npy")
        self.a_validation = np.load("./dataset/validation/a_data_validation.npy")
        self.o2_validation = np.load("./dataset/validation/o2_data_validation.npy")

        assert len(self.a_train) == len(self.o_uncert_train) == len(self.a_train) == len(self.o2_train)
        assert len(self.o_test) == len(self.o_uncert_test) == len(self.a_test) == len(self.o2_test)
        assert len(self.o_validation) == len(self.o_uncert_validation) == len(self.a_validation) == len(self.o2_validation)

        self.train_len, self.test_len, self.vali_len = len(self.a_train), len(self.a_test), len(self.a_validation)

    def sample_train(self, batch_size):
        index = np.random.choice(self.train_len, size=batch_size, replace=False)
        data = dict(o=self.o_train[index],
                    o_uncert=self.o_uncert_train[index],
                    a=self.a_train[index],
                    o2=self.o2_train[index]
                    )
        return data

    def sample_test(self, batch_size):
        index = np.random.choice(self.test_len, size=batch_size, replace=False)
        data = dict(o=self.o_test[index],
                    o_uncert=self.o_uncert_test[index],
                    a=self.a_test[index],
                    o2=self.o2_test[index]
                    )
        return data

    def sample_validation(self, batch_size):
        index = np.random.choice(self.test_len, size=batch_size, replace=False)
        data = dict(o=self.o_validation[index],
                    o_uncert=self.o_uncert_validation[index],
                    a=self.a_validation[index],
                    o2=self.o2_validation[index]
                    )
        return data

def prediction(lr=0.0001, total_steps=int(1e7), batch_train=8, solution='rp-prenn', disp_fre=200, hidden_sizes=(128, 128, 128),
               save_nn_fre=int(1e6), uncert=True, store_folder='.', file_name='train'):

    input_dim = 18 if solution == 'rp-prenn' else 14
    out_dim = 4

    net = core.PreNN(in_dim=input_dim, out_dim=out_dim, hidden_sizes=hidden_sizes)
    prenn_proce = PreNNProce(solution=solution, uncert=uncert, store_folder=store_folder, filename=file_name)
    data_set = Dataset()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.prenn.parameters(), lr=lr)

    def update(data):

        inputs = data['x']
        T_label = data['lable']

        T_out = net.prenn(inputs)

        if solution == 'prenn':
            loss = criterion(T_out, T_label)
        elif solution == 'rp-prenn':
            loss = criterion(T_out+data['predict'], T_label)
        else:
            loss = None
            assert loss is None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    step_print = int(total_steps/500)
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t % disp_fre == 0:

            data_test = data_set.sample_test(batch_size=data_set.test_len)
            data_test = prenn_proce.input_proc(data=data_test)

            if solution == 'prenn':
                T_out = net.prenn(data_test['x'])
                loss = criterion(T_out, data_test['lable'])
            elif solution == 'rp-prenn':
                T_out = net.prenn(data_test['x'])
                loss = criterion(T_out + data_test['predict'], data_test['lable'])
            else:
                loss = None
                assert loss is None

            prenn_proce.tensorboard(loss, t)
            prenn_proce.data_store(loss.detach().numpy())

        if t % save_nn_fre == 0:
            prenn_proce.save_nn(net, t)
            print('**** Agent Saved ****')

        if t % step_print == 0:
            print('Training: ' + str(format(t/total_steps*100, '.1f')) + '%')

        data_train = data_set.sample_train(batch_size=batch_train)
        data_train = prenn_proce.input_proc(data=data_train)
        update(data=data_train)

    prenn_proce.save_nn(net, total_steps)
    print('**** Agent Saved ****')

if __name__ == '__main__':
    prediction()








