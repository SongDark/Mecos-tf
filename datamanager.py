import numpy as np 
import pandas as pd 
import random
from tqdm import tqdm 
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

class RandomSample:
    '''fot test'''
    def __init__(self, batch_size=1, n_ways=128, k_shots=3, q_query=1) -> None:
        
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.q_query = q_query
        self.batch_size = batch_size

    def get_one_meta_batch(self):
        
        meta_batchsize = self.n_ways * self.k_shots
        maxlen = 10

        support_seqs = np.random.randint(0, 10000, (self.batch_size,meta_batchsize, maxlen), dtype=np.int32) 
        support_lens = np.ones((self.batch_size,meta_batchsize,1)) * maxlen
        support_labels = np.random.randint(0, self.n_ways, (self.batch_size,meta_batchsize,))

        query_seqs = np.random.randint(0, 10000, (self.batch_size,self.n_ways, maxlen), dtype=np.int32) 
        query_lens = np.ones((self.batch_size,self.n_ways,1)) * maxlen
        query_labels = np.random.randint(0, self.n_ways, (self.batch_size,self.n_ways,))

        yield support_seqs, support_lens, support_labels, \
            query_seqs, query_lens, query_labels

class Tmall:
    
    def __init__(self, data_path, batch_size=1, n_ways=128, k_shots=3, q_query=1):

        self.n_ways = n_ways
        self.k_shots = k_shots
        self.q_query = q_query
        self.batch_size = batch_size

        self.dataset = {}

        df = pd.read_csv(data_path, sep="\t", header=None, usecols=[1,2], nrows=None)
        df.columns = ["label", "seq"]
        for label, seq in tqdm(df.values):
            if label not in self.dataset:
                self.dataset[label] = []
            self.dataset[label].append([int(v) for v in seq.split(",")])     
        self.steps = int(len(df) // (batch_size * n_ways * k_shots) )   
        del df

        self.items = list(self.dataset.keys())
    
    def get_one_meta_task(self):
        
        chosen_items = random.sample(self.items, self.n_ways)
        
        support_seqs, support_lens, support_labels = [], [], []
        query_seqs, query_lens, query_labels = [], [], []

        for label, chosen_item in enumerate(chosen_items):
            while len(self.dataset[chosen_item]) < self.k_shots + self.q_query:
                chosen_item = random.sample(self.items, 1)[0]
            
            seqs = random.sample(self.dataset[chosen_item], self.k_shots + self.q_query)
            for i in range(len(seqs)):
                if len(seqs[i]) > 64:
                    seqs[i] = seqs[i][-64:]

            for i in range(self.k_shots):
                support_seqs.append(seqs[i])
                support_lens.append(len(seqs[i]))
                support_labels.append(label)
            
            for i in range(self.k_shots, self.k_shots + self.q_query):
                query_seqs.append(seqs[i][:-1])
                query_lens.append(len(seqs[i]) - 1)
                query_labels.append(label)
        
        support_index = list(range(len(support_seqs)))
        random.shuffle(support_index)
        support_seqs = [support_seqs[i] for i in support_index]
        support_lens = [support_lens[i] for i in support_index]
        support_labels = [support_labels[i] for i in support_index]

        query_index = list(range(len(query_seqs)))
        random.shuffle(query_index)
        query_seqs = [query_seqs[i] for i in query_index]
        query_lens = [query_lens[i] for i in query_index]
        query_labels = [query_labels[i] for i in query_index]

        support_seqs = pad_sequences(support_seqs, padding="post")
        support_lens = np.expand_dims(np.array(support_lens), -1)
        support_labels = np.array(support_labels)

        query_seqs = pad_sequences(query_seqs, padding="post")
        query_lens = np.expand_dims(np.array(query_lens), -1)
        query_labels = np.array(query_labels)

        return support_seqs, support_lens, support_labels,\
            query_seqs, query_lens, query_labels

    def get_one_meta_batch(self):

        meta_support_seqs, meta_support_lens, meta_support_labels = [], [], []
        meta_query_seqs, meta_query_lens, meta_query_labels = [], [], [] 
        
        for _ in range(self.batch_size):
            support_seqs, support_lens, support_labels,\
            query_seqs, query_lens, query_labels = self.get_one_meta_task()

            # print(support_seqs.shape)

            meta_support_seqs.append(support_seqs)
            meta_support_lens.append(support_lens)
            meta_support_labels.append(support_labels)

            meta_query_seqs.append(query_seqs)
            meta_query_lens.append(query_lens)
            meta_query_labels.append(query_labels)

        yield np.array(meta_support_seqs), np.array(meta_support_lens), np.array(meta_support_labels), \
            np.array(meta_query_seqs), np.array(meta_query_lens), np.array(meta_query_labels)

# data = RandomSample()
# x1,x2,x3,x4,x5,x6 = next(data.get_one_meta_batch())
# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# print(x4.shape)
# print(x5.shape)
# print(x6.shape)

# data = Tmall(batch_size=2, n_ways=10, k_shots=1, q_query=1)

# # x1,x2,x3,x4,x5,x6 = data.get_one_meta_task()
# x1,x2,x3,x4,x5,x6 = next(data.get_one_meta_batch())

# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# print(x4.shape)
# print(x5.shape)
# print(x6.shape)

# print(list(x1[0]))
# print(x2[0])