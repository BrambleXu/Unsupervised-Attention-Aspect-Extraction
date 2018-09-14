#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import codecs
import operator
import re
from pathlib import Path
import pandas as pd

num_regex = re.compile(r'^[+-]?[0-9]+\.?[0-9]*$')


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab(domain, maxlen=0, vocab_size=0):
    assert domain in {'restaurant', 'beer'}
    source = '../preprocessed_data/' + domain + '/train.txt'

    total_words, unique_words = 0, 0
    word_freqs = {}

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)

    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('  keep the top %i words' % vocab_size)

    # Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('../preprocessed_data/%s/vocab' % domain, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()

    return vocab


def read_dataset(domain, phase, vocab, maxlen):
    # assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}

    source = '../preprocessed_data/' + domain + '/' + phase + '.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            continue
        if not len(words):
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x


def get_data(domain, vocab_size=0, maxlen=0):
    print('Reading data from', domain)
    print(' Creating vocab ...')
    vocab = create_vocab(domain, maxlen, vocab_size)
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset(domain, 'train', vocab, maxlen)
    print('  test set')
    test_x, test_maxlen = read_dataset(domain, 'test', vocab, maxlen)
    maxlen = max(train_maxlen, test_maxlen)
    return vocab, train_x, test_x, maxlen

def create_vocab3(domain, maxlen=0, vocab_size=0):
    assert domain in {'restaurant', 'beer'}
    source = '../preprocessed_data/' + domain + '/train.txt'

    total_words, unique_words = 0, 0
    word_freqs = {}

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.split()
        if maxlen > 0 and len(words) > maxlen:
            continue

        for w in words:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)

    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('  keep the top %i words' % vocab_size)

    # Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('../preprocessed_data/%s/vocab' % domain, mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()

    return vocab


def read_dataset3(domain, phase, vocab, maxlen):
    # assert domain in {'restaurant', 'beer'}
    assert phase in {'train', 'test'}

    source = '../preprocessed_data/' + domain + '/' + phase + '.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    fin = codecs.open(source, 'r', 'utf-8')
    for line in fin:
        words = line.strip().split()
        if maxlen > 0 and len(words) > maxlen:
            continue
        if not len(words):
            continue

        indices = []
        for word in words:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x


def get_data3(train_path, test_path, vocab, vocab_size=0, maxlen=0):
    """
    We already have vocab, so there is no need to create new vocab
    And the maxlen should be same with the data used in attention model
    """
    print('Reading data from', train_path)
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    x_train, y_train = df2data(data_train)
    x_test, y_test = df2data(data_test)

    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset2(x_train, vocab, maxlen)
    print('  test set')
    test_x, test_maxlen = read_dataset2(x_test, vocab, maxlen)
    return vocab, train_x, test_x


def read_dataset2(data, vocab, maxlen):
    """
    :param data:  [['judging', 'from', 'previous', 'posts', 'this', 'used']
                   ['to', 'be', 'a', 'good', 'place', ',', 'but', 'not', 'any', 'longer', '.']]
    :param vocab: a dictionary contain word and index
    :param maxlen: the max length of sentence
    :return: sentence represented as index
    """
    # assert domain in {'restaurant', 'beer'}
    # assert phase in {'train', 'test'}

    # source = '../preprocessed_data/' + domain + '/' + phase + '.txt'
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = 0
    data_x = []

    for s in data:
        indices = []
        for word in s:
            if is_number(word):
                indices.append(vocab['<num>'])
                num_hit += 1
            elif word in vocab:
                indices.append(vocab[word])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1
        data_x.append(indices)
        if maxlen_x < len(indices):
            maxlen_x = len(indices)

    print('   <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100 * num_hit / total, 100 * unk_hit / total))
    return data_x, maxlen_x

def get_data2(train_path, test_path, vocab_size=0, maxlen=0):
    print('Reading data from', train_path)
    print(' Creating vocab ...')
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)

    x_train, y_train = df2data(data_train)
    x_test, y_test = df2data(data_test)
    vocab = create_vocab2(x_train, maxlen, vocab_size)
    print(' Reading dataset ...')
    print('  train set')
    train_x, train_maxlen = read_dataset2(x_train, vocab, maxlen)
    print('  test set')
    test_x, test_maxlen = read_dataset2(x_test, vocab, maxlen)
    maxlen = max(train_maxlen, test_maxlen)
    return vocab, train_x, test_x, maxlen


def is_number(token):
    return bool(num_regex.match(token))


def create_vocab2(x_all, maxlen=0, vocab_size=0):
    # assert domain in {'restaurant', 'beer'}
    # source = 'preprocessed_data/' + domain + '/train.txt'
    # source = Path.cwd().parent.joinpath(source)

    total_words, unique_words = 0, 0
    word_freqs = {}

    for s in x_all:
        if maxlen > 0 and len(s) > maxlen:
            continue

        for w in s:
            if not is_number(w):
                try:
                    word_freqs[w] += 1
                except KeyError:
                    unique_words += 1
                    word_freqs[w] = 1
                total_words += 1

    print('   %i total words, %i unique words' % (total_words, unique_words))
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)

    vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
    index = len(vocab)

    for word, _ in sorted_word_freqs:
        vocab[word] = index
        index += 1
        if vocab_size > 0 and index > vocab_size + 2:
            break
    if vocab_size > 0:
        print('  keep the top %i words' % vocab_size)

    # Write (vocab, frequence) to a txt file
    vocab_file = codecs.open('../preprocessed_data/semeval-2016/vocab', mode='w', encoding='utf8')
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    for word, index in sorted_vocab:
        if index < 3:
            vocab_file.write(word + '\t' + str(0) + '\n')
            continue
        vocab_file.write(word + '\t' + str(word_freqs[word]) + '\n')
    vocab_file.close()

    return vocab


def df2data(df):
    """Read data and labels from dataframe
    Input:
        df: three columns, ['Sentence #', 'Tag', 'Word']
    Output:
        data: datasize * ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
        label: datasize * ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
    """
    agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),
                                                 s["Tag"].values.tolist())]
    grouped = df.groupby("Sentence #").apply(agg_func)
    data = [[w[0] for w in s] for s in grouped]
    label = [[w[1] for w in s] for s in grouped]

    return data, label


# if __name__ == "__main__":
#     # vocab, train_x, test_x, maxlen = get_data('restaurant')
#     # print(len(train_x))
#     # print(len(test_x))
#     # print(maxlen)
