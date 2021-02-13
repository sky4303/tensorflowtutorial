import tensorflow as tf
from tensorflow import keras
import numpy as np

dataset = keras.datasets.imdb
# Only show the 10000 most frequent words:
(training_data, training_labels), (test_data, test_label) = dataset.load_data(num_words=10000)

_word_index = dataset.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
