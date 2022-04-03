import sys

import numpy as np

sys.path.append('..')
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vcab_size = len(word_to_id)
C = create_co_matrix(corpus, vcab_size)
W = ppmi(C, verbose=True)

np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)