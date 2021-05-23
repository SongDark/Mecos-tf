

import tensorflow as tf
import numpy as np 
from datamanager import RandomSample, Tmall 
from maml import MAML 

maml = MAML(n_ways=128, k_shots=3, vocabulary_size=1090391, embedding_size=100)

train_data = Tmall("./data/tmall/data_format1/meta_sequence_train.txt", batch_size=1, n_ways=128, k_shots=3, q_query=1)
val_data = Tmall("./data/tmall/data_format1/meta_sequence_val.txt", batch_size=1, n_ways=128, k_shots=3, q_query=1)
# train_data.steps = 100
val_data.steps = 100

writer = tf.summary.create_file_writer("./logs/")

optimizer = tf.keras.optimizers.Adam(0.0001)

for epoch in range(10):

    train_progbar = tf.keras.utils.Progbar(train_data.steps)
    val_progbar = tf.keras.utils.Progbar(val_data.steps)

    train_meta_loss, train_meta_acc = [], []
    val_meta_loss, val_meta_acc = [], []

    for i in range(train_data.steps):
        loss, acc = maml.train_on_meta_batch(train_data.get_one_meta_batch(), 
                                             outer_optimizer=optimizer, 
                                             writer=writer)
        train_meta_loss.append(loss)
        train_meta_acc.append(acc)

        train_progbar.update(i + 1, [("loss", np.mean(train_meta_loss)), ("accuracy", np.mean(train_meta_acc))] )
    
    for i in range(val_data.steps):
        loss, acc = maml.train_on_meta_batch(val_data.get_one_meta_batch(), 
                                             outer_optimizer=None, 
                                             writer=None)
        val_meta_loss.append(loss)
        val_meta_acc.append(acc)
        val_progbar.update(i + 1, [("loss", np.mean(val_meta_loss)), ("accuracy", np.mean(val_meta_acc))] )
    
    maml.meta_model.save_weights("./models/maml_epoch%d.h5" % epoch)
