#encoding:utf-8

import tensorflow as tf 

from tensorflow.python.keras.layers import Layer 
from tensorflow.python.keras import initializers, regularizers

class SequenceEncoder(Layer):

    def __init__(self, 
                 feedforword_layers=2,
                 embeddings_initializer=None,
                 embeddings_regularizer=None,
                 **kwargs):
        
        super(SequenceEncoder, self).__init__(**kwargs) 

        self.feedforword_layers = feedforword_layers

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)

    def build(self, input_shape):
        emb_size = input_shape[0][-1]

        self.kernel_weights = {}
        for key in ["W1", "W2", "W3"]:
            self.kernel_weights[key] = \
                self.add_weight(
                    name=key, shape=(emb_size, emb_size), 
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    trainable=True, dtype=tf.float32
                )
        self.kernel_weights["b"] = \
            self.add_weight(
                name="b", shape=(emb_size, ),
                initializer="Zeros",
                trainable=True, dtype=tf.float32
            )
        self.kernel_weights["p"] = \
            self.add_weight(
                name="p", shape=(emb_size, ),
                initializer=self.embeddings_initializer,
                trainable=True, dtype=tf.float32
            )
        for l in range(self.feedforword_layers):
            self.kernel_weights["Wl_%d" % l] = \
                self.add_weight(
                    name="Wl_%d" % l, shape=(emb_size * 2, emb_size * 2), 
                    initializer=self.embeddings_initializer,
                    regularizer=self.embeddings_regularizer,
                    trainable=True, dtype=tf.float32
                )
            self.kernel_weights["bl_%d" % l] = \
                self.add_weight(
                    name="bl_%d" % l, shape=(emb_size * 2, ),
                    initializer="Zeros",
                    trainable=True, dtype=tf.float32
                )
        self.kernel_weights["Wq"] = \
            self.add_weight(
                name="Wq", shape=(emb_size, emb_size * 2), 
                initializer=self.embeddings_initializer,
                regularizer=self.embeddings_regularizer,
                trainable=True, dtype=tf.float32
            )
        super(SequenceEncoder, self).build(input_shape)

    def call(self, inputs):
        '''
        input:
            seqs: bs x (N x K) x maxlen x dim
            lens: bs x (N x K) x 1
            labels: bs x (N x K) x dim
        output:
            representation: bs x (N x K) x (dimx2)
        '''
        if len(inputs) == 2:
            seqs, lens = inputs
            labels = None
        elif len(inputs) == 3:
            seqs, lens, labels = inputs
        else:
            raise ValueError("wrong size inputs=%d" % len(inputs))

        # bs x (N x K) x 1 x dim
        V_last = tf.expand_dims(tf.tensordot(
            seqs[:, :, -1, :], 
            self.kernel_weights["W1"], axes=(-1, 0)), -2)
        # bs x (N x K) x maxlen x dim
        V_seq = tf.tensordot(
            seqs, 
            self.kernel_weights["W2"], axes=(-1, 0))
        # bs x (N x K) x 1 x dim
        V_avg = tf.expand_dims(tf.tensordot(
            tf.divide(tf.reduce_sum(seqs, axis=-2), tf.cast(lens, tf.float32) ), 
            self.kernel_weights["W3"], axes=(-1, 0)), -2)
        
        # bs x (N x K) x maxlen
        emb = tf.tensordot(
            tf.nn.bias_add(V_last + V_seq + V_avg, self.kernel_weights["b"]),
            self.kernel_weights["p"], axes=(-1, 0)
        )

        # attention bs x (N x K) x maxlen
        attn = tf.nn.softmax(emb, axis=-1)

        # weighted bs x (N x K) x dim
        weighted_seqs = tf.reduce_sum(tf.multiply(tf.expand_dims(attn, -1), seqs), axis=-2)
        
        # feedforward bs x (N x K) x (2 * dim)
        if labels is None:
            hidden_proj_init = tf.tensordot(weighted_seqs, self.kernel_weights["Wq"], axes=(-1,0))
        else:
            hidden_proj_init = tf.concat([weighted_seqs, labels], axis=-1)
        
        hidden_proj = hidden_proj_init
        for l in range(self.feedforword_layers):
            hidden_proj = tf.nn.bias_add(tf.tensordot(
                hidden_proj, self.kernel_weights["Wl_%d" % l], axes=(-1,0)), self.kernel_weights["bl_%d" % l])
            hidden_proj = tf.nn.relu(hidden_proj)
        
        # N x 2d
        seq_representation = hidden_proj_init + hidden_proj 

        return seq_representation

    def get_config(self):
        config = {
            "embeddings_initializer": self.embeddings_initializer,
            "embeddings_regularizer": self.embeddings_regularizer,
            "feedforword_layers": self.feedforword_layers
        }
        base_config = super(SequenceEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Aggregator(Layer):

    def __init__(self, mode="mean", axis=1, **kwargs):
        super(Aggregator, self).__init__(**kwargs)
        self.mode = mode
        self.axis = axis 
        self.support_modes = ["max", "mean", "last"]
    
    def call(self, inputs):
        
        if self.mode == "mean": 
            return tf.reduce_mean(inputs, axis=self.axis)
        elif self.mode == "max":
            return tf.reduce_max(inputs, axis=self.axis)
        elif self.mode == "last":
            return inputs[:, -1, :]
        else:
            raise ValueError("fatal aggregator mode=%s" % self.mode)

    def get_config(self):
        config = {
            "mode": self.mode,
            "support_modes": self.support_modes
        }
        base_config = super(Aggregator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    # def compute_output_shape(self, input_shape):
    #     if self.mode == "mean":
    #         return (input_shape[0][0], input_shape[0][-1])



# seqs = tf.ones([2, 10, 32, 7])
# maxlen = tf.multiply(tf.ones((2, 10, 1)), 32)
# labels = tf.ones([2, 10, 7])

# mecos = SequenceEncoder()
# out = mecos([seqs, maxlen, labels])
# print(out.shape)