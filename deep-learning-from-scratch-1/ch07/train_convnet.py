# coding: utf-8
import os
import sys
sys.path.append(os.path.pardir)
import numpy as np
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.my_trainer import Trainer
import matplotlib.pyplot as plt


# load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

max_ephocs = 20

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_ephocs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# save parameters TODO


# draw result graph
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_ephocs)
plt.plot(x, trainer.train_acc_list, label='train acc')
plt.plot(x, trainer.test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
