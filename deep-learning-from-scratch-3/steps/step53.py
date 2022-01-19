if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt

import time
import dezero
from dezero import optimizers
from dezero.dataloaders import DataLoader
from dezero.models import MLP
import dezero.functions as F

max_epochs = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

#load param
file_name = 'my_mlp.npz'
if os.path.exists(file_name):
    model.load_weights(file_name)

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()


for epoch in range(max_epochs):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print('epoch: {}, loss: {:.4f}, time: {:.4f}'.format(
        epoch + 1,
        sum_loss / len(train_set),
        elapsed_time))

model.save_weights(file_name)