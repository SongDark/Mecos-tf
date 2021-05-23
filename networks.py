#encoding:utf-8

import tensorflow as tf
from tensorflow.python.keras import initializers, regularizers
from tensorflow.python.keras.layers import Layer, LSTM, RNN
from utils import SequenceEncoder, Aggregator
from lstmcell import TestLSTMCell
from recurrent import TestRNN

class Mecos(Layer):

    def __init__(self,
                 n_ways, k_shots,
                 vocabulary_size,
                 embedding_size,
                 embeddings_initializer="glorot_normal",
                 embeddings_regularizer=None,
                 **kwargs):

        self.n_ways = n_ways
        self.k_shots = k_shots 

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)

        super(Mecos, self).__init__(**kwargs)

    def build(self, input_shape):
    
        self.item_embeddings = self.add_weight(
            name="",
            shape=(self.vocabulary_size, self.embedding_size),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            dtype=tf.float32, trainable=True
        )

        self.sequence_encoder = SequenceEncoder(feedforword_layers=2, name="seq_enc")

        cell = TestLSTMCell(input_shape[0][-1])
        self.lstm = TestRNN(cell=cell, return_state=True)

        super(Mecos, self).build(input_shape)

    def call(self, inputs):
        '''
        inputs:
            support seq: bs x (N x K) x maxlen
            support len: bs x (N x K) x 1
            support labels: bs x (N x K)
            support seq: bs x N x maxlen
            support len: bs x N x 1
            support labels: bs x N
        outputs
        '''

        support_seqs, support_lens, support_labels, query_seqs, query_lens = inputs 
        
        support_seqs = tf.nn.embedding_lookup(self.item_embeddings, support_seqs) # bs x (N x K) x maxlen x dim
        support_labels = tf.nn.embedding_lookup(self.item_embeddings, support_labels) # bs x (N x K) x dim
        query_seqs = tf.nn.embedding_lookup(self.item_embeddings, query_seqs) # bs x N x maxlen x dim
        
        support_embs = self.sequence_encoder([support_seqs, support_lens, support_labels]) # bs x (N x K) x (2xdim)
        support_embs = tf.reshape(support_embs, (-1, self.n_ways, self.k_shots, self.embedding_size * 2)) # bs x N x K x (2*dim)
    
        # aggregation for S
        support_embs = tf.reduce_mean(support_embs, axis=-2) # bs x N x (2xdim)

        # Q
        query_embs = self.sequence_encoder([query_seqs, query_lens]) # bs x N x (2xdim)
        
        # matching
        # support_embs = lstm_encoder([support_embs, query_embs])

        support_embs = tf.tile(support_embs, [1, self.n_ways, 1])
        query_embs = tf.reshape(tf.tile(query_embs, [1, 1, self.n_ways]), (-1, self.n_ways**2, self.embedding_size*2))

        support_embs = tf.nn.l2_normalize(support_embs, axis=-1)
        query_embs = tf.nn.l2_normalize(query_embs, axis=-1)

        cos_similarity = tf.reduce_sum(tf.multiply(query_embs, support_embs), axis=-1)
        cos_similarity = tf.reshape(cos_similarity, (-1, self.n_ways, self.n_ways))

        outputs = tf.nn.softmax(cos_similarity, axis=-1)

        return outputs

    def get_config(self):
        config = {}
        base_config = super(Mecos, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# bs = 3
# n_ways = 10
# k_shots = 3
# vocabulary_size = 10000
# embedding_size = 32
# meta_batchsize = n_ways * k_shots

# # seqs = tf.ones([meta_batchsize, 32, 1])
# # maxlen = tf.multiply(tf.ones((meta_batchsize, 1)), 32)
# # labels = tf.ones([meta_batchsize, 1])

# maxlen = None
# support_seqs = tf.keras.layers.Input(shape=(meta_batchsize, maxlen,), batch_size=bs, dtype=tf.int32, name="support_seqs")
# support_lens = tf.keras.layers.Input(shape=(meta_batchsize, 1,), batch_size=bs, dtype=tf.int32, name="support_lens")
# support_labels = tf.keras.layers.Input(shape=(meta_batchsize,), batch_size=bs, dtype=tf.int32, name="support_labels")
# query_seqs = tf.keras.layers.Input(shape=(n_ways, maxlen,), dtype=tf.int32, batch_size=bs)
# query_lens = tf.keras.layers.Input(shape=(n_ways, 1,), dtype=tf.int32, batch_size=bs)
# query_labels = tf.keras.layers.Input(shape=(n_ways,), dtype=tf.int32, batch_size=bs)

# # encoder = SequenceEncoder(feedforword_layers=2, name="my")
# mecos = Mecos(n_ways=n_ways, k_shots=k_shots, vocabulary_size=vocabulary_size, embedding_size=embedding_size)

# cos = mecos([support_seqs, support_lens, support_labels, query_seqs, query_lens])
# # print(cos.shape)
# # model = tf.keras.Model(inputs=[support_seqs, support_lens, support_labels, query_seqs, query_lens], 
# #                        outputs=[emb1, emb2])
# # model.summary() 


# a = tf.constant([[1,1,1],[2,2,2]], tf.int32)
# b = tf.constant([[3,3,3],[4,4,4]], tf.int32)

# a = tf.tile(a, [2,1])
# b = tf.reshape(tf.tile(b, [1,2]), a.shape)
# print(a)
# print(b)

# support_embs = tf.constant([[-1,1],[2,2]], dtype=tf.float32)
# query_embs = tf.constant([[3,3],[4,4]], dtype=tf.float32)

# support_embs = tf.tile(support_embs, [2, 1])
# query_embs = tf.reshape(tf.tile(query_embs, [1, 2]), support_embs.shape)

# support_embs = tf.nn.l2_normalize(support_embs, axis=1)
# query_embs = tf.nn.l2_normalize(query_embs, axis=1)
# cos_similarity = tf.reduce_sum(tf.multiply(query_embs, support_embs), axis=1)

# cos_similarity = tf.reshape(cos_similarity, (2,2))
# print(cos_similarity)