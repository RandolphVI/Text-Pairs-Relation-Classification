# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import multiprocessing
import gensim
import logging
import json
import numpy as np

from collections import OrderedDict
from pylab import *
from gensim.models import word2vec
from tflearn.data_utils import to_categorical, pad_sequences

TEXT_DIR = '../data/content.txt'
METADATA_DIR = '../data/metadata.tsv'


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_prediction_file(output_file, front_data_id, behind_data_id,
                           all_labels, all_predict_labels, all_predict_values):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted scores provided by network
        front_data_id: The front data record id info provided by class Data
        behind_data_id: The behind data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_values: The all predict values by threshold
    Raises:
        IOError: If the prediction file is not a .json file
    """
    # TODO
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        all_predict_labels = all_predict_labels.tolist()
        all_predict_values = all_predict_values.tolist()

        data_size = len(all_predict_labels)

        for i in range(data_size):
            labels = int(all_labels[i])
            predict_labels = int(all_predict_labels[i])
            predict_values = round(all_predict_values[i], 4)
            data_record = OrderedDict([
                ('front_testid', front_data_id[i]),
                ('behind_testid', behind_data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_values', predict_values)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=True) + '\n')


def create_metadata_file(embedding_size, output_file=METADATA_DIR):
    """
    Create the metadata file based on the corpus file(Use for the Embedding Visualization later).

    Args:
        embedding_size: The embedding size
        output_file: The metadata file
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist."
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = gensim.models.Word2Vec.load(word2vec_file)
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    """
    Create the word2vec model based on the given embedding size and the corpus file.

    Args:
        embedding_size: The embedding size
        input_file: The corpus file
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    sentences = word2vec.LineSentence(input_file)
    # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
    model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                   sg=0, workers=multiprocessing.cpu_count())
    model.save(word2vec_file)


def load_word2vec_matrix(embedding_size):
    """
    Return the word2vec model matrix.

    Args:
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist. "
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = gensim.models.Word2Vec.load(word2vec_file)
    vocab_size = len(model.wv.vocab.items())
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            embedding_matrix[value] = model[key]
    return vocab_size, embedding_matrix


def data_word2vec(input_file, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data (includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        word2vec_model: The word2vec model file
    Returns:
        The Class Data (includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    if not input_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        labels = []
        front_testid = []
        behind_testid = []
        front_content_indexlist = []
        behind_content_indexlist = []
        total_line = 0
        for eachline in fin:
            data = json.loads(eachline)
            front_testid.append(data['front_testid'])
            behind_testid.append(data['behind_testid'])
            labels.append(data['label'])
            front_content_indexlist.append(_token_to_index(data['front_features']))
            behind_content_indexlist.append(_token_to_index(data['behind_features']))
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def front_testid(self):
            return front_testid

        @property
        def behind_testid(self):
            return behind_testid

        @property
        def front_tokenindex(self):
            return front_content_indexlist

        @property
        def behind_tokenindex(self):
            return behind_content_indexlist

        @property
        def labels(self):
            return labels

    return _Data()


def load_data_and_labels(data_file, embedding_size):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        embedding_size: The embedding size
    Returns:
        The class Data
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if not os.path.isfile(word2vec_file):
        create_word2vec_model(embedding_size, TEXT_DIR)

    model = word2vec.Word2Vec.load(word2vec_file)

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, word2vec_model=model)

    # plot_seq_len(data_file, data)

    # print('Found {0} texts.'.format(data.number))

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        data_front: The padded front data
        data_behind: The padded behind data
        onehot_labels: The one-hot labels
    """
    data_front = pad_sequences(data.front_tokenindex, maxlen=pad_seq_len, value=0.)
    data_behind = pad_sequences(data.behind_tokenindex, maxlen=pad_seq_len, value=0.)
    onehot_labels = to_categorical(data.labels, nb_classes=2)
    return data_front, data_behind, onehot_labels


def plot_seq_len(data_file, data, percentage=0.98):
    """
    Visualizing the sentence length of each data sentence.

    Args:
        data_file: The data_file
        data: The class Data (includes the data tokenindex and data labels)
        percentage: The percentage of the total data you want to show
    """
    data_analysis_dir = '../data/data_analysis/'
    if 'train' in data_file.lower():
        output_file = data_analysis_dir + 'Train Sequence Length Distribution Histogram.png'
    if 'validation' in data_file.lower():
        output_file = data_analysis_dir + 'Validation Sequence Length Distribution Histogram.png'
    if 'test' in data_file.lower():
        output_file = data_analysis_dir + 'Test Sequence Length Distribution Histogram.png'
    result = dict()
    for x in (data.front_tokenindex + data.behind_tokenindex):
        if len(x) not in result.keys():
            result[len(x)] = 1
        else:
            result[len(x)] += 1
    freq_seq = [(key, result[key]) for key in sorted(result.keys())]
    x = []
    y = []
    avg = 0
    count = 0
    border_index = []
    for item in freq_seq:
        x.append(item[0])
        y.append(item[1])
        avg += item[0] * item[1]
        count += item[1]
        if (count / 2) > data.number * percentage:
            border_index.append(item[0])
    avg = avg / data.number
    print('The average of the data sequence length is {0}'.format(avg))
    print('The recommend of padding sequence length should more than {0}'.format(border_index[0]))
    xlim(0, 200)
    plt.bar(x, y)
    plt.savefig(output_file)
    plt.close()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
