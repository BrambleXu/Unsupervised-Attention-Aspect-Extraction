#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import argparse
import codecs

import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

import utils as U

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True,
                    help="The path to the output directory")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=200,
                    help="Embeddings dimension (default=200)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=50,
                    help="Batch size (default=50)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000,
                    help="Vocab size. '0' means no limit (default=9000)")
parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15,
                    help="Number of epochs (default=15)")
parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                    help="Number of negative instances (default=20)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0,
                    help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
parser.add_argument("--domain", dest="domain", type=str, metavar='<str>', default='restaurant',
                    help="domain of the corpus {restaurant, beer}")
parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                    help="The weight of orthogonol regularizaiton (default=0.1)")

# args = parser.parse_args()
args = parser.parse_args(["--emb", "../preprocessed_data/restaurant/w2v_embedding",
                          "--domain", "restaurant",
                          "-o", "output_dir"])

# out_dir = args.out_dir_path + '/' + args.domain
# U.mkdir_p(out_dir)
U.print_args(args)

assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}

from keras.preprocessing import sequence
import reader as dataset

###### Get test data #############
# vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
# test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

train_path = Path.cwd().parent.joinpath('datasets/semeval-2016/train.csv')
test_path = Path.cwd().parent.joinpath('datasets/semeval-2016/test.csv')

# read data from csv file
vocab, train_x, test_x, overall_maxlen = dataset.get_data2(train_path, test_path, vocab_size=args.vocab_size, maxlen=args.maxlen)
train_x = sequence.pad_sequences(train_x, maxlen=overall_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)


############# Build model architecture, same as the model used for training #########
from model import create_model
import keras.backend as K
from optimizers import get_optimizer

optimizer = get_optimizer(args)


def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)


model = create_model(args, overall_maxlen, vocab)

## Load the save model parameters
model.load_weights('../pre_trained_model/restaurant' + '/model_param')
model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])


# Create a dictionary that map word index to word
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w

test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('test_att_weights').output,
                      model.get_layer('p_t').output,
                      model.get_layer('att_sentence').output])
train_att_weights, _, train_att_sentences = test_fn([train_x, 0])
test_att_weights, _, test_att_sentences = test_fn([test_x, 0])

np.save('output_dir/semeval-2016/train_att_sentence.npy', train_att_sentences)
np.save('output_dir/semeval-2016/test_att_sentences.npy', test_att_sentences)


# Save attention weights on test sentences into a file
att_out = codecs.open('output_dir/semeval-2016' + '/test_att_weights', 'w', 'utf-8')
print('Saving attention weights on test sentences...')

for c in range(len(test_x)):

    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i != 0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen - line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')