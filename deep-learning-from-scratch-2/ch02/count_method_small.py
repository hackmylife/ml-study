import sys

import numpy as np

sys.path.append('..')
from common.util import preprocess, create_co_matrix, ppmi
import matplotlib.pyplot as plt


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vcab_size = len(word_to_id)
C = create_co_matrix(corpus, vcab_size)
W = ppmi(C, verbose=True)

# SVD(Singular Value Decomposition)
U, S, V = np.linalg.svd(W)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()