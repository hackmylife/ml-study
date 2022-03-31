import sys
sys.path.append('..')
import numpy as np


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    w2i = {}
    i2w = {}

    for word in words:
        if word not in w2i:
            id = len(w2i)
            w2i[word] = id
            i2w[id] = word

    corpus = np.array([w2i[w] for w in words])

    return corpus, w2i, i2w

