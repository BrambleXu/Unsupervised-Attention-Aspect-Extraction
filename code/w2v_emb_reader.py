#!/usr/bin/env python
#  -*- coding: utf-8  -*-

import logging

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans
# from anago.utils import load_glove
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class W2VEmbReader:

    def __init__(self, emb_path, emb_dim=None):

        logger.info('Loading embeddings from: ' + emb_path)
        self.embeddings = {}
        emb_matrix = []

        # loading pretrained vectors
        # model = gensim.models.Word2Vec.load(emb_path)
        model = KeyedVectors.load_word2vec_format(emb_path, binary=False)

        self.emb_dim = emb_dim

        for word in model.wv.vocab:
            self.embeddings[word] = list(model.wv[word])
            emb_matrix.append(list(model.wv[word]))

        if emb_dim is not None:
            assert self.emb_dim == len(self.embeddings['nice'])

        self.vector_size = len(self.embeddings)
        self.emb_matrix = np.asarray(emb_matrix)

        logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))

    def get_emb_given_word(self, word):

        try:
            return self.embeddings[word]
        except KeyError:
            return None

    def get_emb_matrix_given_vocab(self, vocab, emb_matrix):

        counter = 0.
        for word, index in vocab.items():
            try:
                emb_matrix[index] = self.embeddings[word]
                counter += 1
            except KeyError:
                pass

        logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
        # L2 normalization
        norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)

        return norm_emb_matrix

    def get_aspect_matrix(self, n_clusters):
        """
            We need it for initialization: KMeans-clustered word embeddings
        """

        km = KMeans(n_clusters=n_clusters)
        km.fit(self.emb_matrix)
        clusters = km.cluster_centers_

        # L2 normalization
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)

        return norm_aspect_matrix

    def get_emb_dim(self):
        return self.emb_dim



#
# class W2VEmbReader:
#
#     def __init__(self, emb_path, emb_dim, vocab, n_clusters):
#
#         logger.info('Loading embeddings from: ' + emb_path)
#         # loading pretrained vectors, model is glove dictionary
#         self.embeddings = load_glove(args.emb_path)
#
#         self.vector_size = len(self.embeddings)
#         self.emb_dim = emb_dim
#         if emb_dim is not None:
#             assert emb_dim == len(self.embeddings['nice'])
#         logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))
#
#         # get_emb_matrix_given_vocab
#         emb_matrix = np.zeros((len(vocab), self.emb_dim))
#
#         counter = 0.
#         for word, index in vocab.items():
#             embedding_vecter = self.embeddings.get(word)
#             if embedding_vecter is not None:
#                 emb_matrix[index] = embedding_vecter
#                 counter += 1
#
#         logger.info(
#             '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
#         # L2 normalization
#         self.norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
#
#         # get_aspect_matrix
#         """
#             We need it for initialization: KMeans-clustered word embeddings
#         """
#
#         km = KMeans(n_clusters=n_clusters)
#         km.fit(self.norm_emb_matrix)
#         clusters = km.cluster_centers_
#
#         # L2 normalization
#         self.norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
#
#
#     def get_emb_given_word(self, word):
#
#         try:
#             return self.embeddings[word]
#         except KeyError:
#             return None


#
# class W2VEmbReader:
#
#     def __init__(self, emb_path, emb_dim=None):
#
#         logger.info('Loading embeddings from: ' + emb_path)
#         self.embeddings = {}
#         emb_matrix = []
#
#         # loading pretrained vectors
#         model = gensim.models.Word2Vec.load(emb_path)
#         self.emb_dim = emb_dim
#
#         for word in model.wv.vocab:
#             self.embeddings[word] = list(model.wv[word])
#             emb_matrix.append(list(model.wv[word]))
#
#         if emb_dim is not None:
#             assert self.emb_dim == len(self.embeddings['nice'])
#
#         self.vector_size = len(self.embeddings)
#         self.emb_matrix = np.asarray(emb_matrix)
#
#         logger.info('  #vectors: %i, #dimensions: %i' % (self.vector_size, self.emb_dim))
#
#     def get_emb_given_word(self, word):
#
#         try:
#             return self.embeddings[word]
#         except KeyError:
#             return None
#
#     def get_emb_matrix_given_vocab(self, vocab, emb_matrix):
#
#         counter = 0.
#         for word, index in vocab.items():
#             try:
#                 emb_matrix[index] = self.embeddings[word]
#                 counter += 1
#             except KeyError:
#                 pass
#
#         logger.info(
#             '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(vocab), 100 * counter / len(vocab)))
#         # L2 normalization
#         norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
#
#         return norm_emb_matrix
#
#     def get_aspect_matrix(self, n_clusters):
#         """
#             We need it for initialization: KMeans-clustered word embeddings
#         """
#
#         km = KMeans(n_clusters=n_clusters)
#         km.fit(self.emb_matrix)
#         clusters = km.cluster_centers_
#
#         # L2 normalization
#         norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
#
#         return norm_aspect_matrix
#
#     def get_emb_dim(self):
#         return self.emb_dim

# def load_glove(file):
#     """Loads GloVe vectors in numpy array.
#     Args:
#         file (str): a path to a glove file.
#     Return:
#         dict: a dict of numpy arrays.
#     """
#     embeddings_index = {}
#     with open(file, encoding='utf8') as f:
#         for i, line in tqdm(enumerate(f)):
#             values = line.split()
#             word = ''.join(values[:-300])
#             coefs = np.asarray(values[-300:], dtype='float32')
#             embeddings_index[word] = coefs
#
#     return embeddings_index
#
# glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
