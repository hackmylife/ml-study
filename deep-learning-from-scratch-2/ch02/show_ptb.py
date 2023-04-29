import sys
sys.path.append('..')
from dataset import ptb

print(sys.path)
corpus, word_to_id, id_to_word = ptb.load_data('train')

print('corpus size:', len(corpus))
print('corpus[:30]:', corpus[:30])
print()
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[10]:', id_to_word[10])
print('id_to_word[20]:', id_to_word[20])
print()
print('word_to_id[0]:', word_to_id['car'])
print('word_to_id[10]:', word_to_id['happy'])
print('word_to_id[20]:', word_to_id['lexus'])
