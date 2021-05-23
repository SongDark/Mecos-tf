import pandas as pd 
import random
import numpy as np 
from tqdm import tqdm 

'''1. count item'''
# log = pd.read_csv("./data_format1/user_log_format1.csv", nrows=None, usecols=["user_id", "item_id", "time_stamp"])
# item_gp = log.drop_duplicates().groupby("item_id").size().reset_index().rename(columns={0:"cnt"}).sort_values("cnt", ascending=True)
# item_gp.to_csv("./data_format1/item_cnt_duplicated.csv", index=False)
# item_gp = log.groupby("item_id").size().reset_index().rename(columns={0:"cnt"}).sort_values("cnt", ascending=True)
# item_gp.to_csv("./data_format1/item_cnt.csv", index=False)
 
'''2. meta-sequence'''
# log = pd.read_csv("./data_format1/user_log_format1.csv", nrows=None, usecols=["user_id", "item_id", "time_stamp", "action_type"])
# log["order"] = log.index.values # origin order

# log = log.sort_values(["user_id", "time_stamp", "order"]).drop(["action_type", "order"], axis=1)
# log = log.drop_duplicates()

# item_gp = pd.read_csv("./data_format1/item_cnt_duplicated.csv")
# item_gp = item_gp[item_gp.cnt >= 10]

# cold_items = item_gp.sort_values("cnt").head(int(0.2 * len(item_gp)))
# log = pd.merge(log, cold_items, "left", ["item_id"]).rename(columns={"cnt": "flag"})
# log["flag"] = log["flag"].fillna(0)

# fout = open("./data_format1/meta_sequence.txt", "w")
# for user_id, gp in tqdm( log.groupby("user_id") ):
#     user_log = gp.reset_index(drop=True)
#     index = user_log[user_log.flag != 0].index 
#     for i in index:
#         seq = list(user_log.item_id[:i + 1].values)
#         if len(seq) <= 1:
#             continue
#         label = str(seq[len(seq) - 1])
#         fout.write(str(user_id) + "\t" + label + "\t" + ",".join([str(v) for v in seq]) + "\n")
# fout.close()

# log = pd.read_csv("./data_format1/user_log_format1.csv", nrows=None, usecols=["user_id", "item_id", "time_stamp", "action_type"])
# log["order"] = log.index.values # origin order

# log = log.sort_values(["user_id", "time_stamp", "order"]).drop(["action_type", "order"], axis=1)
# log = log.drop_duplicates()

# item_gp = pd.read_csv("./data_format1/item_cnt_duplicated.csv")
# log = pd.merge(log, item_gp, "left", ["item_id"]).rename(columns={"cnt": "item_cnt"})
# print(log)

# item_gp = item_gp[item_gp.cnt >= 10]
# cold_items = item_gp.sort_values("cnt").head(int(0.2 * len(item_gp)))
# log = pd.merge(log, cold_items, "left", ["item_id"]).rename(columns={"cnt": "flag"})

# log["flag"] = log["flag"].fillna(0)
# log["flag1"] = log.apply(lambda x: 1 if x["flag"] > 0 or x["item_cnt"] <= 10 else 0, axis=1)

# # print(len(log[log.flag > 0]), len(log[(log.flag > 0) | (log.item_cnt <= 10)]), len(log) )

# gp = log.groupby("user_id")["flag1"].sum().reset_index()


'''3. train-test split'''

meta_sequence = pd.read_csv("./data_format1/meta_sequence.txt", sep="\t", header=None, nrows=None)
meta_sequence.columns = ["user_id", "item_id", "seq"]

cold_items = meta_sequence.item_id.unique()

random.seed(2021)
random.shuffle(cold_items)

cold_items_train = pd.DataFrame({"item_id": cold_items[:int(0.7 * len(cold_items))]})
cold_items_val = pd.DataFrame({"item_id": cold_items[int(0.7 * len(cold_items)):int(0.8 * len(cold_items))]})
cold_items_test = pd.DataFrame({"item_id": cold_items[int(0.8 * len(cold_items)):]})
print(len(cold_items_train), len(cold_items_val), len(cold_items_test))

pd.merge(meta_sequence, cold_items_train, "inner", "item_id").to_csv("./data_format1/meta_sequence_train.txt", sep="\t", header=None, index=False)
pd.merge(meta_sequence, cold_items_val, "inner", "item_id").to_csv("./data_format1/meta_sequence_val.txt", sep="\t", header=None, index=False)
pd.merge(meta_sequence, cold_items_test, "inner", "item_id").to_csv("./data_format1/meta_sequence_test.txt", sep="\t", header=None, index=False)