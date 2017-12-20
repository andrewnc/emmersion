import os
import soundfile as sf
from tqdm import tqdm
import numpy as np
import re
import random


class DataLoader(object):
    def __init__(self, path="data", dtype=np.float64, truncate_secs=None, sample_rate=44100, train_split=0.7):
        self._path = path
        self._dtype = dtype
        self._truncate_secs = truncate_secs
        self._data = None
        self._labels = None
        self._sample_rate = sample_rate
        self._num_data = 0

        # For train and test
        self._train_split = train_split
        self._train_data = None
        self._train_labels = None
        self._test_data = None
        self._test_labels = None
        self._num_train_data = 0
        self._num_test_data = 0

        self.index_to_lang = dict()
        self._lang_to_index = dict()

        print("Loading data")
        if self._truncate_secs is not None:
            self._load_array()
            self.get_batch = self._get_batch_array
        else:
            self._load_list()
            self.get_batch = self._get_batch_list

    def _load_array(self):
        languages = set()
        for file in os.listdir(self._path):
            if file.endswith(".wav"):
                languages.add(re.search('^[a-zA-Z]*', file).group(0))
                self._num_data += 1

        length = self._truncate_secs * self._sample_rate
        num_languages = len(languages)
        self._data = np.zeros([self._num_data, length], dtype=self._dtype)
        self._labels = np.zeros([self._num_data, num_languages], dtype=np.bool)
        for i, x in enumerate(languages):
            self._lang_to_index[x] = i
            self.index_to_lang[i] = x

        index = 0
        for file in tqdm(os.listdir(self._path)):
            if file.endswith(".wav"):
                sig, _ = sf.read(os.path.join(self._path, file))
                sig_length = sig.shape[0]
                self._data[index, :min(length, sig_length)] = sig[:min(length, sig_length)]
                self._labels[index, self._lang_to_index[re.search('^[a-zA-Z]*', file).group(0)]] = 1
                index += 1

        indices = np.random.permutation(self._data.shape[0])
        split_idx = int(self._num_data * self._train_split)
        training_idx, test_idx = indices[:split_idx], indices[split_idx:]
        self._train_data, self._train_labels = self._data[training_idx, :], self._labels[training_idx, :]
        self._test_data, self._test_labels = self._data[test_idx, :], self._labels[test_idx, :]
        self._num_train_data = self._train_data.shape[0]
        self._num_test_data = self._test_data.shape[0]

    def _get_batch_array(self, batch_size=32, test=False):
        if test:
            indices = np.random.randint(0, self._num_test_data, batch_size)
            return self._test_data[indices, :], self._test_labels[indices, :]

        indices = np.random.randint(0, self._num_train_data, batch_size)
        return self._train_data[indices, :], self._train_labels[indices, :]

    def _load_list(self):
        self._data = []
        for file in tqdm(os.listdir(self._path)):
            if file.endswith(".wav"):
                sig, _ = sf.read(os.path.join(self._path, file))
                self._data.append((sig.astype(self._dtype), re.search('^[a-zA-Z]*', file).group(0)))
        self._num_data = len(self._data)

    def _get_batch_list(self, batch_size=32):
        xs, ys = [], []
        for i in range(batch_size):
            data_point = self._data[random.randint(0, self._num_data)]
            xs.append(data_point[0])
            ys.append(data_point[1])
        return xs, ys
