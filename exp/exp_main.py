import numpy as np
import torch
import torch.nn as nn
from torch import optim
from downstream.bert_baseline import BERT
from exp.exp_basic import Exp_Basic
from utils import next_batch, get_evalution, make_exchange_matrix, Loss_Function, adjust_learning_rate,get_vec2word_acc
import time
import os
from tqdm import tqdm
import pickle
from data_factory import data_provider
from gensim.models import KeyedVectors

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.data_provider = data_provider(args)
        self.vocab_size = self.data_provider.get_vocabsize()
        if self.args.is_training == 2 or self.args.is_training == 3 or self.args.is_training == 4:
            self.idx2word = self.data_provider.get_idx2word()
            self.word2vec_model = self.data_provider.get_word2vecModel()
        self.exchange_map = self.data_provider.get_exchange_map()
        self.model = self._build_model(self.vocab_size).to(self.device)
    def _build_model(self,vocab_size):
        model_dict = {
            'trajbert':BERT
        }
        model = model_dict[self.args.model](args = self.args,vocab_size = vocab_size).float()

        return model

    def _select_optimizer(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def _select_criterion(self):
        if self.args.loss == 'loss' :
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = Loss_Function()
        return criterion

    def test(self, test_loader, criterion):
        total_loss = []
        self.model.eval()
        predict_prob = torch.Tensor([]).to(self.device)
        total_masked_tokens = np.array([])
        with torch.no_grad():
            for i, (input_ids, masked_tokens, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis) in enumerate(tqdm(test_loader,ncols=100)):
                total_masked_tokens = np.append(total_masked_tokens,np.array(masked_tokens.cpu()).reshape(-1)).astype(int)
                if self.args.is_training ==2 or self.args.is_training ==4 or self.args.is_training ==3:
                    # h_masked  batch , pre_num, d model
                    logits_lm ,h_masked = self.model(input_ids, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis)
                    h_masked = h_masked.cpu().view(-1,self.args.d_model).clone().numpy()
                    if i == 0:
                        total_h_masked = h_masked
                    else:
                        total_h_masked = np.append(total_h_masked,h_masked, axis=0)
                    
                else:
                    logits_lm = self.model(input_ids, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis)

                logits_lm_ = torch.topk(logits_lm, 100, dim=2)[1]
                predict_prob = torch.cat([predict_prob, logits_lm_], dim=0)
                if self.args.is_training !=2 :
                    if self.args.loss == "spatial_loss":
                        loss_lm = criterion.Spatial_Loss(self.exchange_map, logits_lm.view(-1, self.vocab_size),
                                                        masked_tokens.view(-1))  # for masked LM
                    elif self.args.loss == "top_loss":
                        loss_lm = criterion.top_loss( logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))
                    else:
                        loss_lm = criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
                    loss = (loss_lm.float()).mean()

                    total_loss.append(loss)
        self.model.train()
        if self.args.is_training !=2 : 
            total_loss = torch.mean(torch.stack(total_loss))

            accuracy_score, fuzzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score, map_score, wrong_pre= get_evalution(
            ground_truth=total_masked_tokens, logits_lm=predict_prob, exchange_matrix=self.exchange_map)

            return 'test accuracy score =' + '{:.6f}'.format(accuracy_score) + '\n' + \
            'fuzzzy score =' +  '{:.6f}'.format(fuzzzy_score) + '\n'\
                + 'test top3 score ='+ '{:.6f}'.format(top3_score) + '\n'\
                + 'test top5 score ='+ '{:.6f}'.format(top5_score) + '\n'\
                + 'test top10 score ='+ '{:.6f}'.format(top10_score) + '\n'\
                + 'test top30 score ='+ '{:.6f}'.format(top30_score) + '\n'\
                + 'test top50 score ='+ '{:.6f}'.format(top50_score) + '\n'\
                + 'test top100 score ='+ '{:.6f}'.format(top100_score) + '\n' \
                + 'test MAP score ='+ '{:.6f}'.format(map_score) + '\n'  ,total_loss,accuracy_score,wrong_pre
        else:
            acc = get_vec2word_acc(total_h_masked,total_masked_tokens,self.word2vec_model,self.idx2word)
            print('test accuracy score =' + '{:.6f}'.format(acc) + '\n')
            return acc #'test accuracy score =' + '{:.6f}'.format(acc) + '\n'


    def train(self, setting):
        train_loader = self.data_provider.get_loader(flag='train',args = self.args)

        test_loader = self.data_provider.get_loader(flag='test',args = self.args)

        if self.args.load_checkpoint  and os.path.exists(self.args.root_path+'checkpoint/'+setting+'.pth'): #load checkpoint
            state_dict = torch.load(self.args.root_path+'checkpoint/'+setting+'.pth',map_location=self.device)
            self.model.load_state_dict(state_dict['model'])
            print('load model ' + self.args.root_path+'checkpoint/'+setting+'.pth' +' success')
        
        print('start train')
        
        best_score,score_data = self.get_best_score(setting)
        tmp_best_score = 0

        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer() 
        criterion = self._select_criterion()

        for epoch in range(self.args.epoch):
            train_loss = []
            self.model.train()

            if ((epoch+1) % 10 == 0 and self.args.remask==1 ) :
                train_loader = self.data_provider.get_loader(flag='train',args = self.args)
            
            for i, (input_ids, masked_tokens, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis) in enumerate(tqdm(train_loader,ncols=100)):
                model_optim.zero_grad()
                logits_lm = self.model(input_ids, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis)
               
                if self.args.loss == "spatial_loss":
                    loss_lm = criterion.Spatial_Loss(self.exchange_map, logits_lm.view(-1, self.vocab_size),
                                                    masked_tokens.view(-1))  # for masked LM
                elif self.args.loss == "top_loss":
                    loss_lm = criterion.top_loss( logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))
                else:
                    loss_lm = criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
                
                    
                loss = (loss_lm.float()).mean()

                train_loss.append(loss)
                
                loss.backward()
                model_optim.step()

            train_loss = torch.mean(torch.stack(train_loss))
            print("Epoch: {} | Train Loss: {}  ".format(epoch + 1, train_loss))
            if (epoch+1)%1 ==0 or (epoch+ 1 > 100 and (epoch +1)%2 ==0) or epoch == 0 :
                
                torch.save({'model': self.model.state_dict()},self.args.root_path+'checkpoint/'+setting+'.pth')
                result, test_loss ,accuracy_score,wrong_pre= self.test(test_loader, criterion)
                tmp_best_score = max(tmp_best_score,accuracy_score)
                if accuracy_score >= best_score:
                    torch.save({'model': self.model.state_dict()},self.args.root_path+'result/'+setting+'.pth')
                    print('update best score from ',best_score,' to ',accuracy_score)
                    best_score = accuracy_score
                    score_data[setting] = best_score
                    pickle.dump(score_data,open(self.args.root_path+'middata/best_score.pkl','wb+')) #更新分数
                f = open(self.args.root_path+'result/'+setting+'.txt','a+')
                f.write("epoch: %d \n" %(epoch+1))
                f.write("train loss: %.6f | test loss: %.6f \n" % (train_loss, test_loss))
                f.write(result)
                f.close()
                print("Epoch: {} | Train Loss: {}  Test Loss: {}".format(epoch + 1, train_loss, test_loss))

            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        print('best accuracy:',tmp_best_score)
        return tmp_best_score

    def train_with_word2vec(self, setting):
        train_loader = self.data_provider.get_loader(flag='train',args = self.args)

        test_loader = self.data_provider.get_loader(flag='test',args = self.args)

        if self.args.load_checkpoint  and os.path.exists(self.args.root_path+'checkpoint/'+setting+'.pth'): #load checkpoint
            state_dict = torch.load(self.args.root_path+'checkpoint/'+setting+'.pth',map_location=self.device)
            self.model.load_state_dict(state_dict['model'])
            print('load model ' + self.args.root_path+'checkpoint/'+setting+'.pth' +' success')
        
        print('start train')
        
        best_score,score_data = 0,{}#self.get_best_score(setting)

        model_optim = self._select_optimizer() 
        criterion = self._select_criterion()

        vectors = []
        for token in self.word2vec_model.key_to_index:
            vector = self.word2vec_model[token]
            vectors.append(vector)
        mean_vector = np.mean(vectors, axis=0)

        word2vec_weight = self.model.embedding.tok_embed.weight.cpu().clone()
        for i in range(4,self.vocab_size):
            # vev = torch.zeros(self.args.d_model)
            vev = torch.from_numpy(mean_vector) # 如果没有该token 那么取所有token的均值

            try:
                vev = torch.from_numpy(np.copy(self.word2vec_model[self.idx2word[str(i)]]))
            except:
                print('word2vec model have not token',i,self.idx2word[str(i)])

            word2vec_weight[i] = vev
        self.model.embedding.tok_embed.data = word2vec_weight
        if self.args.word2vec_grad:
            for param in self.model.embedding.tok_embed.parameters():
                    param.requires_grad = True
        print('check token fc2 weight ')
        print(torch.equal(self.model.embedding.tok_embed.weight , self.model.fc2.weight))

        if self.args.is_training ==2:
            best_score = 0
        for epoch in range(self.args.epoch):
            train_loss = []
            self.model.train()
            if ((epoch+1) % 10 == 0 and self.args.remask==1 ) :
                train_loader = self.data_provider.get_loader(flag='train',args = self.args)
            
            for i, (input_ids, masked_tokens, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis) in enumerate(tqdm(train_loader,ncols=100)):
                model_optim.zero_grad()
                
                logits_lm ,h_masked= self.model(input_ids, masked_pos, user_ids, day_ids,input_next,input_prior,input_prior_dis,input_next_dis)

                if self.args.loss == "spatial_loss":
                    loss_lm = criterion.Spatial_Loss(self.exchange_map, logits_lm.view(-1, self.vocab_size),
                                                    masked_tokens.view(-1))  # for masked LM
                    loss = (loss_lm.float()).mean()
                elif self.args.loss == "top_loss":
                    loss_lm = criterion.top_loss( logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))
                    loss = (loss_lm.float()).mean()
                elif self.args.is_training != 2 and self.args.is_training == 3 : 
                    loss_lm = criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
                    loss = (loss_lm.float()).mean()
                elif self.args.is_training == 2 or self.args.is_training == 4:  # 2 4 使用 mse loss
                    # mse loss
                    masked_token_vec = torch.zeros(masked_tokens.view(-1).size(0),self.args.d_model)
                    for i,each in  enumerate(masked_tokens.view(-1).cpu()): #将每个mask token 转成word2vec 向量
                        masked_token_vec[i] = torch.from_numpy(np.copy(self.word2vec_model[self.idx2word[str(each.item())]]))
                    mse = nn.MSELoss()
                    loss = mse(h_masked.view(-1,self.args.d_model), masked_token_vec.to(self.device)) #计算 模型输出向量与token word2vec向量的 mse
                    
                train_loss.append(loss)
                
                loss.backward()
                model_optim.step()
            
            train_loss = torch.mean(torch.stack(train_loss))
            print("Epoch: {} | Train Loss: {}  ".format(epoch + 1, train_loss))
            if (epoch+1)%5 ==0 or (epoch+ 1 > 100 and (epoch +1)%2 ==0) or epoch == 0 :
                
                torch.save({'model': self.model.state_dict()},self.args.root_path+'checkpoint/'+setting+'.pth')
                if self.args.is_training !=2 :
                    result, test_loss ,accuracy_score,wrong_pre= self.test(test_loader, criterion)
                    if accuracy_score >= best_score:
                        torch.save({'model': self.model.state_dict()},self.args.root_path+'result/'+setting+'.pth')
                        print('update best score from ',best_score,' to ',accuracy_score)
                        best_score = accuracy_score
                    f = open(self.args.root_path+'result/'+setting+'.txt','a+')
                    f.write("epoch: %d \n" %(epoch+1))
                    f.write("train loss: %.6f | test loss: %.6f \n" % (train_loss, test_loss))
                    f.write(result)
                    f.close()
                    print("Epoch: {} | Train Loss: {}  Test Loss: {}".format(epoch + 1, train_loss, test_loss))
                else:
                    score = self.test(test_loader, criterion)
                    best_score = max(best_score,score)
        if self.args.is_training ==2:
            print('best acc ',best_score)


            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return 


    def infer(self,setting):
        infer_result_path = self.args.root_path+'infer_result'
        if not os.path.exists(infer_result_path):
            os.mkdir(infer_result_path)

        if not os.path.exists(self.args.root_path+'result/'+setting+'.pth'):
            print('no such model, check args settings')
            return

        if self.args.infer_model_path == '':
            self.load_weight(self.args.root_path+'result/'+setting+'.pth')
            print('load model ' + self.args.root_path+'result/'+setting+'.pth' +' success')

        else:
            self.load_weight(self.args.root_path+'result/'+self.args.infer_model_path)
            print('load model ' + self.args.root_path+'result/'+self.args.infer_model_path +' success')
        
        test_loader = self.data_provider.get_loader(flag='infer',args = self.args)
        criterion = self._select_criterion()
        result, test_loss ,accuracy_score,wrong_pre= self.test(test_loader, criterion)

        f = open(self.args.root_path+'infer_result/'+setting+'.txt','a+')
        f.write(" test loss: %.6f \n" %  test_loss)
        f.write(result)
        f.write('\n'.join(wrong_pre))
        f.close()


        return

    def load_weight(self,model_path):
        state_dict = torch.load(model_path,map_location=self.device)['model']
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def get_best_score(self,setting):
        best_score_file = self.args.root_path+'middata/best_score.pkl' #储存对应模型最高分
        best_model_path = self.args.root_path+'result'
        checkpoint_path = self.args.root_path+'checkpoint'
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        if not os.path.exists(best_model_path):
            os.mkdir(best_model_path)
        if not os.path.exists(best_score_file):
            score_data = {setting:0}
            pickle.dump(score_data,open(best_score_file,'wb+'))
        score_data = pickle.load(open(best_score_file,'rb'))
        if setting not in  score_data:
            score_data[setting] = 0
        best_score = score_data[setting]
        print(setting,' history best socre ',best_score)

        return best_score,score_data
    
    
    


    

