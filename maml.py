# encoding:utf-8

import tensorflow as tf 
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.layers import Input
from networks import Mecos
import numpy as np 


class MAML:

    def __init__(self,
                 n_ways, k_shots,
                 vocabulary_size,
                 embedding_size
                 ):
        
        self.n_ways = n_ways 
        self.k_shots = k_shots
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.meta_model = self.build_model()
    
        self.inner_writer_step = 0
        self.outer_writer_step = 0
    
    def build_model(self):
        
        maxlen = None
        meta_batchsize = self.n_ways * self.k_shots

        support_seqs = Input(shape=(meta_batchsize, maxlen,), dtype=tf.int32, name="support_seqs")
        support_lens = Input(shape=(meta_batchsize,1,), dtype=tf.int32, name="support_lens")
        support_labels = Input(shape=(meta_batchsize,), dtype=tf.int32, name="support_labels")
        query_seqs = Input(shape=(self.n_ways, maxlen,), dtype=tf.int32, name="query_seqs")
        query_lens = Input(shape=(self.n_ways,1,), dtype=tf.int32, name="query_lens")
        # query_labels = Input(shape=(), dtype=tf.int32, batch_size=self.n_ways, name="query_labels")

        mecos = Mecos(n_ways=self.n_ways, k_shots=self.k_shots, vocabulary_size=self.vocabulary_size, embedding_size=self.embedding_size)
        logits = mecos([support_seqs, support_lens, support_labels, query_seqs, query_lens])

        model = tf.keras.Model(inputs=[support_seqs, support_lens, support_labels, query_seqs, query_lens],
                               outputs=[logits])
        return model 

    def train_on_meta_batch(self, train_tasks_iterator, inner_optimizer=None, inner_step=1, outer_optimizer=None, writer=None):

        meta_support_seqs, meta_support_seqlens, meta_support_labels, \
            meta_query_seqs, meta_query_seqlens, meta_query_labels = next(train_tasks_iterator)

        for support_seqs, support_seqlens, support_labels, \
            query_seqs, query_seqlens, query_labels in zip(meta_support_seqs, meta_support_seqlens, meta_support_labels, meta_query_seqs, meta_query_seqlens, meta_query_labels):

            support_seqs = np.expand_dims(support_seqs, 0)
            support_seqlens = np.expand_dims(support_seqlens, 0)
            support_labels = np.expand_dims(support_labels, 0)
            query_seqs = np.expand_dims(query_seqs, 0)
            query_seqlens = np.expand_dims(query_seqlens, 0)
            query_labels = np.expand_dims(query_labels, 0)

            '''
            Single Task:
                support_seqs: N x K x seqlen
                support_seqlens: N x K x 1
                support_labels: N x K x 1
                query_seqs: N x seqlen
                query_seqlens: N x 1
                query_labels: N x 1
            '''
            task_tape = tf.GradientTape()

            losses = []
            accs = []
            for _ in range(inner_step):
                with task_tape as tape:
                    logits = self.meta_model([support_seqs, support_seqlens, support_labels, query_seqs, query_seqlens])
                    loss = tf.reduce_mean(sparse_categorical_crossentropy(query_labels, logits))
                    acc = (np.argmax(logits, -1) == query_labels).astype(np.int32).mean()

                    losses.append(loss)
                    accs.append(acc)

                    if writer is not None:
                        with writer.as_default():
                            tf.summary.scalar("loss", loss, step=self.inner_writer_step)
                            tf.summary.scalar("acc", acc, step=self.inner_writer_step)
                            self.inner_writer_step += 1
            
            # Update
            with task_tape as tape:
                if outer_optimizer is not None:
                    grads = tape.gradient(tf.reduce_sum(losses), self.meta_model.trainable_variables)
                    outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return np.array(losses), np.array(accs)
