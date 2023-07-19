

from dataclasses import dataclass
import math
import torch.utils.data as Data
from preprocess import DataSet
from utils import  make_exchange_matrix

import torch
from random import *
import os
import pandas as pd
import pickle
from trainword2vec import run_word2vec
from gensim.models import KeyedVectors
class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos, user_ids, day_ids, input_prior, input_next,
                 input_prior_dis, input_next_dis):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.user_ids = user_ids
        self.day_ids = day_ids
        self.input_prior = input_prior
        self.input_next = input_next
        self.input_prior_dis = input_prior_dis
        self.input_next_dis = input_next_dis

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.user_ids[idx], self.day_ids[
            idx], self.input_prior[idx], self.input_next[idx], self.input_prior_dis[idx], self.input_next_dis[idx]

def data_prerocess(args):
    device = 'cuda:%s' % str(args.gpu)
    if args.remask in [0,1]:
        if args.data_type == 'cdr' :
            train_data_name = "cdr/train_traj_%s.h5" % args.pre_len
            if args.is_training:
                test_data_name ="cdr/test_traj_%s.h5" % args.pre_len
            else:
                test_data_name = args.infer_data_path

        elif args.data_type == 'geolife':

            train_data_name = "geo/train_5.h5" 
            if args.is_training:
                test_data_name ="geo/format_test_5.h5" 
            else:
                test_data_name = args.infer_data_path
            
        elif args.data_type == 'tdrive':

            train_data_name = "tdrive/train2_traj_5.h5" 
            if args.is_training:
                test_data_name ="tdrive/test2_traj_5.h5" 
            else:
                test_data_name = args.infer_data_path

        else:
            raise Exception('please check data type', args.data_type)
        
        print('success load ',train_data_name, test_data_name)
        train_df = pd.read_hdf(os.path.join(args.root_path,args.data_path, train_data_name))
        test_df = pd.read_hdf(os.path.join(args.root_path,args.data_path, test_data_name))

        dataset = DataSet(train_df, test_df)
       
        train_data = dataset.gen_train_data()  # [seq, user_index, day]
        test_data = dataset.gen_test_data()  # [seq, masked_pos, masked_tokens, user_index, day]
        train_word_list = list(
            set(str(train_data[i][0][j]) for i in range(len(train_data)) for j in range(len(train_data[i][0]))))
        test_word_list = list(
            set(str(test_data[i][0][j]) for i in range(len(test_data)) for j in range(len(test_data[i][0]))))
        test_masked_list = list(
            set(str(test_data[i][2][j]) for i in range(len(test_data)) for j in range(len(test_data[i][2]))))
        train_word_list.remove('[PAD]')
        test_word_list.remove('[PAD]')
        try:
            test_word_list.remove('[MASK]')
        except:pass
        train_word_list.extend(test_word_list)
        train_word_list.extend(test_masked_list)
        train_word_list = list(set(train_word_list))
        train_word_list.sort()

        train_word_list_int = [int(train_word_list[i]) for i in range(len(train_word_list))]
        train_word_list_int.sort()

        word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i, w in enumerate(train_word_list_int):
            if w == '[PAD]' or w == '[MASK]':
                print("error")
            word2idx[str(w)] = i + 4

        vocab_size = len(word2idx)

        train_token_list = list()
        train_user_list = list()
        train_day_list = list()
        max_value = 0
        for sentence in train_data:
            seq, user_index, day = sentence
            for s in seq:
                try:
                    max_value = max(max_value, word2idx[str(s)])
                except:
                    print(s)
                    
            arr = [word2idx[s] for s in seq]
            train_token_list.append(arr)
            train_user_list.append(user_index)
            train_day_list.append(day)

        exchange_map = make_exchange_matrix(token_list=train_token_list, token_size=vocab_size)
        exchange_map = torch.Tensor(exchange_map).to(device)
        print('vocab size: ',vocab_size)

        # self.idx2word = { str(v):k for k,v in word2idx.items()}

    return vocab_size, exchange_map, train_user_list, train_day_list, train_token_list, word2idx, test_data


class data_provider():
    def __init__(self,args):
        self.args = args
        device = 'cuda:%s' % str(args.gpu)
        if args.remask in [0,1]:
            self.vocab_size, self.exchange_map, self.train_user_list, self.train_day_list, self.train_token_list,self.word2idx, self.test_data = data_prerocess(args)
            if args.is_training == 2  :
                if args.data_type == 'cdr' :
                    train_data_name = "cdr/train_traj_%s.h5" % args.pre_len
                elif args.data_type == 'geolife':
                    train_data_name = "geo/train_5.h5" 
                elif args.data_type == 'tdrive':
                    train_data_name = "tdrive/train2_traj_5.h5" 
                # 训练word2vec 模型
                outputFile = os.path.join('middata','word2vec_'+args.data_type+'_'+str(args.d_model))
                if not os.path.exists(outputFile):
                    train_df = pd.read_hdf(os.path.join(args.root_path,args.data_path, train_data_name))
                    run_word2vec(outputFile,train_df,args.d_model)
                self.wrod2vecPath =  outputFile.split('.')[0]
                print('word2vec model path:' ,self.wrod2vecPath )

        elif args.remask == 2:
            print('load data from pkl')
            if args.data_type == 'cdr':
                self.vocab_size = pickle.load(open( self.args.root_path + 'middata/vocab_size_' + str(args.pre_len) + '.pkl', 'rb'))
                print('load size')
                self.train_user_list = pickle.load(open( self.args.root_path + 'middata/train_user_list_' + str(args.pre_len) + '.pkl', 'rb'))
                print('load train_user_list')
                self.train_day_list = pickle.load(open( self.args.root_path + 'middata/train_day_list_' + str(args.pre_len) + '.pkl', 'rb'))
                print('load train_day_list')
                self.test_total_data = pickle.load(open( self.args.root_path + 'middata/test_total_data_' + str(args.pre_len) + '.pkl', 'rb'))
                print('load test_total_data')
                self.exchange_map = pickle.load(open(self.args.root_path + 'middata/exchange_map_' + str(args.pre_len) + '.pkl', 'rb'))
                self.exchange_map = self.exchange_map.to(device)
                print('load exchange_map')
                self.total_data = pickle.load(open( self.args.root_path + 'middata/total_data_' + str(args.pre_len) + '.pkl', 'rb'))
                print('load total_data')
                

            elif args.data_type == 'geolife':
                self.vocab_size = pickle.load(open(self.args.root_path + 'middata/237_vocab_size_'+str(args.pre_len)+'.pkl','rb'))
                self.total_data = pickle.load(open(self.args.root_path + 'middata/237_total_data_'+str(args.pre_len)+'.pkl','rb'))
                self.exchange_map = pickle.load(open(self.args.root_path + 'middata/237_exchange_map_'+str(args.pre_len)+'.pkl','rb'))
                self.exchange_map = self.exchange_map.to(device)
                self.train_user_list = pickle.load(open(self.args.root_path + 'middata/237_train_user_list_'+str(args.pre_len)+'.pkl','rb'))
                self.train_day_list = pickle.load(open(self.args.root_path + 'middata/237_train_day_list_'+str(args.pre_len)+'.pkl','rb'))
                self.test_total_data = pickle.load(open(self.args.root_path + 'middata/237_test_total_data_'+str(args.pre_len)+'_debug.pkl','rb'))
                self.wrod2vecPath = 'middata/word2vec_train_trajectory_geoPkl_512'
                self.word2idx = {str(i):i for i in range(self.vocab_size)}
            else:
                raise Exception('please check data type', args.data_type)

    
    def get_loader(self,flag,args):
        device = 'cuda:%s' % str(args.gpu)

        exchange_map_copy = self.exchange_map.clone()
        for i in range(len(exchange_map_copy)):
            exchange_map_copy[i][i] = 0

        if flag == 'train':
            if self.args.remask in [0,1]:
                self.total_data = make_train_data(self.train_token_list,exchange_map_copy,self.word2idx,int(self.args.pre_len)) 

            input_ids, masked_tokens, masked_pos = zip(*self.total_data)
            input_prior,input_next,input_prior_dis,input_next_dis=get_id_pn(input_ids,masked_pos,self.args.use_his)

            input_prior = torch.LongTensor(input_prior).to(device)
            input_next = torch.LongTensor(input_next).to(device)
            input_prior_dis = torch.FloatTensor(input_prior_dis).to(device)
            input_next_dis = torch.FloatTensor(input_next_dis).to(device)
            user_ids, day_ids = torch.LongTensor(self.train_user_list).to(device), torch.LongTensor(self.train_day_list).to(device)
            input_ids, masked_tokens, masked_pos, = torch.LongTensor(input_ids).to(device),  torch.LongTensor(masked_tokens).to(device), torch.LongTensor(masked_pos).to(device)
            
            loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos, user_ids, day_ids,input_prior,input_next,input_prior_dis,input_next_dis), self.args.bs, True)
            return loader
        
        elif flag == 'test' or flag == 'infer':
            if self.args.remask in [0,1]:
                self.test_total_data = make_test_data(self.test_data, self.word2idx)
            
            test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids = zip(*self.test_total_data)
            test_input_prior,test_input_next,test_input_prior_dis,test_input_next_dis=get_id_pn(test_input_ids,test_masked_pos,self.args.use_his)
            
            test_input_ids = torch.LongTensor(test_input_ids).to(device)
            test_masked_tokens = torch.LongTensor(test_masked_tokens).to(device)
            test_masked_pos = torch.LongTensor(test_masked_pos).to(device)
            test_user_ids = torch.LongTensor(test_user_ids).to(device) 
            test_day_ids = torch.LongTensor(test_day_ids).to(device)
            test_input_prior = torch.LongTensor(test_input_prior).to(device)
            test_input_next = torch.LongTensor(test_input_next).to(device)
            test_input_prior_dis = torch.FloatTensor(test_input_prior_dis).to(device)
            test_input_next_dis = torch.FloatTensor(test_input_next_dis).to(device)

            loader = Data.DataLoader(MyDataSet(test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids,test_input_prior,test_input_next,test_input_prior_dis,test_input_next_dis), 64, True)

            return loader
    
    def get_vocabsize(self):
        return self.vocab_size
    
    def get_word2vecModel(self):
        word2vec_model = KeyedVectors.load_word2vec_format(self.wrod2vecPath, binary=True)
        return word2vec_model

    def get_exchange_map(self):
        return self.exchange_map
    
    def get_idx2word(self):
        return { str(v) : k for k,v in self.word2idx.items()}

def make_train_data(token_list,exchange_map_copy,word2idx,max_pred):
    total_data = []
    vocab_size = len(word2idx)
    for i in range(len(token_list)):
        tokens_a_index = i  # sample random index in sentences
        tokens_a = token_list[tokens_a_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                        if token != word2idx['[CLS]'] and token != word2idx['[SEP]'] and token != word2idx[
                            '[PAD]']]  # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace


        total_data.append([input_ids, masked_tokens, masked_pos])
    return total_data

def make_test_data(test_data,word2idx):
    # [seq, masked_pos, masked_tokens, user_index, day]
    total_test_data = []
    for sentence in test_data:
        arr = [word2idx[s] for s in sentence[0]]
        user = sentence[3]
        arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
        masked_pos = [pos + 1 for pos in sentence[1]]
        masked_tokens = [word2idx[str(s)] for s in sentence[2]]
        day = sentence[4]
        total_test_data.append([arr, masked_tokens, masked_pos, user, day])
    return total_test_data

def get_dis_score(dis):
    return 1 / math.log(1 + dis, 2)

def get_id_pn(input_ids, masked_pos,use_his):
    input_prior = []
    input_next = []
    input_prior_dis = []  # xy
    input_next_dis = []
    for i in range(len(input_ids)):
        seq = input_ids[i]
        ids_prior = []
        ids_next = []
        ids_prior_dis = []  # xy
        ids_next_dis = []

        for pos in masked_pos[i]:
            if use_his in [3, 4]:
                j = pos - 1
                while seq[j] in [0, 1, 2, 3]:
                    j -= 1
                ids_prior.append(seq[j])
                ids_prior_dis.append(get_dis_score(abs(pos - j)))

                j, dis = (pos + 1) % 50, 1
                while seq[j] in [0, 1, 2, 3]:
                    j = (j + 1) % 50
                    dis += 1
                ids_next.append(seq[j])
                ids_next_dis.append(get_dis_score(abs(dis)))
            elif use_his in [1, 2]:
                for j in range(pos - 1, -1, -1):
                    if seq[j] != 0 and seq[j] != 3:
                        ids_prior.append(seq[j])
                        ids_prior_dis.append(get_dis_score(abs(pos - j)))
                        break
                    # else:
                    #     ids_prior_dis.append(1e8)
                for j in range(pos + 1, 50):
                    if seq[j] != 0 and seq[j] != 3:
                        ids_next.append(seq[j])
                        ids_next_dis.append(get_dis_score(abs(pos - j)))
                        break

        input_prior.append(ids_prior)
        input_next.append(ids_next)
        input_prior_dis.append(ids_prior_dis)
        input_next_dis.append(ids_next_dis)

    return input_prior, input_next, input_prior_dis, input_next_dis