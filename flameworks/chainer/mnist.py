from chainer import training
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from chainer import datasets, iterators, optimizers
from chainer import training
from chainer.training import extensions

train, test = datasets.mnist.get_mnist()

batchsize = 128

train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, False, False)


class MLP(Chain):

    def __init__(self, n_mid_units=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


model = MLP()
model.to_cpu()

max_epoch = 10
model = L.Classifier(model)
optimizer = optimizers.Adam()

optimizer.setup(model)

updater = training.updaters.StandardUpdater(train_iter, optimizer)

trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(test_iter, model))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))

trainer.run()