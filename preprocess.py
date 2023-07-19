import collections
import random
from random import shuffle
import pandas as pd
import os
import time
import csv
from utils import traj_to_slot


def gen_all_traj(file_path, file_name):
    # 从原始记录中获得轨迹
    seqs = []
    abs_file = os.path.join(file_path, file_name)
    data = pd.read_csv(abs_file)
    format = '%Y-%m-%d %H:%M:%S'

    data['timestamp'] = data["datetime"]
    data['date'] = data["datetime"].apply(lambda x: time.strftime(format, time.localtime(x)))
    data['day'] = pd.to_datetime(data['date'], errors='coerce').dt.day
    for (user_index, day), group in data.groupby(["user_index", 'day']):
        if group.shape[0] < 24:
            continue
        group.sort_values(by="timestamp")
        ts = group['timestamp'].astype(int).tolist()
        seq = group['loc_index'].tolist()
        seq = traj_to_slot(seq, ts, pad='[PAD]')
        if seq.count('[PAD]') > 24:
            continue
        seq = [str(x) for x in seq]
        seqs.append([" ".join(seq), user_index, day])
    print("Length of all trajectory data is " + str(len(seqs)))
    seqs = pd.DataFrame(seqs)
    indexes = ['trajectory', 'user_index', 'day']
    seqs.columns = indexes
    h5 = pd.HDFStore('data/Dataset Filtered 2 h5/%s.h5' % "all_traj", 'w')
    h5['data'] = seqs
    h5.close()


class ConstructDataSet:
    # 构造训练集测试集
    def __init__(self, data_df, n_pred):
        # ['trajectory', 'user_index', 'day']
        self.data = data_df
        self.train_seq = []
        self.test_seq = []
        self.token_set = set()
        self.train_token_set = set()
        self.train_user_set = set()
        self.n_pred = n_pred
        for index, row in self.data.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            seq = list(seq.split())
            for loc in seq:
                self.token_set.add(loc)
            if day > 25:
                self.test_seq.append([seq, user_index, day])
            else:
                self.train_seq.append([seq, user_index, day])
                self.train_user_set.add(user_index)
                for loc in seq:
                    self.train_token_set.add(loc)

    def store_train_data(self):
        print("All train length is " + str(len(self.train_seq)))
        train_data = pd.DataFrame(self.train_seq)
        indexes = ['trajectory', 'user_index', 'day']
        train_data.columns = indexes
        h5 = pd.HDFStore('data/Dataset Filtered 2 h5/%s.h5' % "train_traj_%s" % self.n_pred, 'w')
        h5['data'] = train_data
        h5.close()

    def store_test_data(self, masked_test_data, name):
        masked_test_data = pd.DataFrame(masked_test_data)
        indexes = ['trajectory', 'masked_pos', 'masked_tokens', 'user_index', 'day']
        masked_test_data.columns = indexes
        h5 = pd.HDFStore('data/Dataset Filtered 2 h5/%s.h5' % name, 'w')
        h5['data'] = masked_test_data
        h5.close()

    def masked_test_seqs(self, test_seq):
        test_records = []
        for record in test_seq:
            seq, user_index, day = record
            cand_maked_pos = [i for i, token in enumerate(seq) if seq[i] != '[PAD]']  # candidate masked position
            shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:self.n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(seq[pos])
                seq[pos] = '[MASK]'  # make mask
            seq, masked_pos, masked_tokens = [str(x) for x in seq], \
                                             [str(x) for x in masked_pos], [str(x) for x in masked_tokens]
            test_records.append([" ".join(seq), " ".join(masked_pos), " ".join(masked_tokens), user_index, day])
        return test_records

    def gen_train_test_data_1(self):
        # 数据集1：测试集中的基站和用户在训练中都出现过
        test_record = []
        for record in self.test_seq:
            seq, user_index, day = record
            if user_index not in self.train_user_set:
                continue
            test_record.append(record)

        masked_test_record = self.masked_test_seqs(test_record)
        print("All test length is " + str(len(masked_test_record)))
        self.store_test_data(masked_test_data=masked_test_record, name="test_traj_%s" % self.n_pred)
        return self.train_seq, masked_test_record


class DataSet:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def gen_all_data(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for index, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            records.append(seq)
        for index, row in self.test_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            seq = list(seq.split())
            records.append(seq)
        print("All data length is " + str(len(records)))
        return records

    def gen_train_data(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for index, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            records.append([seq, user_index, day])
        print("All train length is " + str(len(records)))
        return records

    def gen_train_data_td(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for index, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            seq = eval(seq)
            seq = [str(each) for each in seq]
            records.append([seq, user_index, day])
        print("All train length is " + str(len(records)))
        return records

    def gen_test_data(self):
        # ['trajectory', 'masked_pos', 'masked_tokens']
        test_df = self.test_df
        records = []
        for index, row in test_df.iterrows():
            seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
            user_index, day = row['user_index'], row['day']
            try:  
                eval(seq)
                seq = eval(seq)
            except:pass
            try:
                seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                                list(map(int, masked_tokens.split()))
            except:
                seq, masked_pos, masked_tokens = list(seq), list(map(int, masked_pos)), \
                                             list(map(int, masked_tokens))
            records.append([seq, masked_pos, masked_tokens, user_index, day])
        print("All test length is " + str(len(records)))
        return records
    
    def gen_test_data_td(self):
        # ['trajectory', 'masked_pos', 'masked_tokens']
        test_df = self.test_df
        records = []
        for index, row in test_df.iterrows():
            seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
            user_index, day = row['user_index'], row['day']
            eval(seq)
            seq = eval(seq)
            seq = [str(each) for each in seq]
            seq, masked_pos, masked_tokens = seq, list(map(int, masked_pos.split())), \
                                            list(map(int, masked_tokens.split()))
            
            records.append([seq, masked_pos, masked_tokens, user_index, day])
        print("All test length is " + str(len(records)))
        return records

    def gen_train_data_and_user(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for index, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            records.append([seq, user_index])
        print("All train length is " + str(len(records)))
        return records

    def gen_test_data_and_user(self):
        # ['trajectory', 'masked_pos', 'masked_tokens']
        test_df = self.test_df
        records = []
        for index, row in test_df.iterrows():
            seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
            user_index, day = row['user_index'], row['day']
            seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                             list(map(int, masked_tokens.split()))
            records.append([seq, masked_pos, masked_tokens, user_index])
        print("All test length is " + str(len(records)))
        return records


if __name__ == '__main__':


    train_df = pd.read_hdf(os.path.join('data/Dataset Filtered 2 h5', "train_traj_24" + ".h5"), key='data')
    test_df = pd.read_hdf(os.path.join('data/Dataset Filtered 2 h5', "test_traj_24" + ".h5"), key='data')
    dataset = DataSet(train_df, test_df)
   
