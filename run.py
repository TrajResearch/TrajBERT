import argparse
import torch
from exp.exp_main import Exp_Main as Exp
import time
import os
def main(parser = None):
    if parser == None:
        parser = argparse.ArgumentParser()
    
        # data loader
        parser.add_argument('--root_path', type=str, default='', help='root path') #colab path -> pub/ or ''
        parser.add_argument('--data_path', type=str, default='./data/', help='data path ') # 
        
        parser.add_argument('--pre_len', type=str, default='5', help='predict len') # 
        parser.add_argument('--data_type', type=str, default='cdr', help='geolife or cdr')
        parser.add_argument('--infer_data_path', type=str, default='', help='infer data path ') # 
        parser.add_argument('--infer_model_path', type=str, default='', help='infer model path ')
        # model define
        parser.add_argument('--d_model', default=512, type=int, help='embed size')
        parser.add_argument('--model',type=str, default='trajbert', help='trajbert')
        parser.add_argument('--head', default=2, type=int, help='multi head num')
        parser.add_argument('--layer', default=2, type=int, help='layer')
        parser.add_argument('--max_k', default=-1, type=int, help='relative embedding')
        parser.add_argument('--seq_len', default=50, type=int, help='sequence lenght')
        parser.add_argument('--remask', default=0, type=int, help='remask')  # 1 代表是 0代表不是 2代表load pickle
        parser.add_argument('--if_posiemb', default=0, type=int, help='position embedding')  # 1 代表是 0代表不是 2代表可训练
        parser.add_argument('--relative_v', default=0, type=int,
                            help='padding version')  # 1 代表是padding的，2代表padding clip的 0代表最原始relative
        parser.add_argument('--use_his', default=0, type=int,
                            help='use temporal reference')  # 0代表不用，1:初版  2:初版+权重 3:代表padding版本 4:padding+权重
        parser.add_argument('--earlystop', default=0, type=int, help='early stop')  # earyly stop 
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
        parser.add_argument('--lradj', type=str, default='type0', help='adjust learning rate')

        # train settings
        parser.add_argument('--is_training', type=int, default=1, help='status,2 ,3,4 means use word2vec,2 : mse loss, get_vec2word_acc , 3: cel loss get_evalution, 4: mse loss get_evalution')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--bs', default=256, type=int, help='batch size')
        parser.add_argument('--epoch', default=50, type=int, help='epoch size')
        parser.add_argument('--loss', default='loss', type=str, help='loss fun')
        parser.add_argument('--load_checkpoint', default=0, type=int, help='if continue train') #  0:no 

        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=1, help='gpu')

        parser.add_argument('--word2vec_grad', type=int, default=1, help='1 means true')

    args = parser.parse_args()


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)


    # Exp = Exp_Main
    if args.is_training ==1:
        for ii in range(args.itr):
            # setting record of experiments
            setting = 'data_{}_{}_dmodel_{}_head_{}_layer_{}_loss_{}_bs_{}_epoch_{}_posiemb_{}_max_k_{}_relembV_{}_remask_{}_use_temporal_{}_lr_{}'.format(
                args.data_type,
                args.pre_len,
                args.d_model,
                args.head,
                args.layer,
                args.loss,
                args.bs,
                args.epoch,
                args.if_posiemb,
                args.max_k,
                args.relative_v,
                args.remask,
                args.use_his,
                str(args.lr).split('.')[1],
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            acc = exp.train(setting)

            torch.cuda.empty_cache()
        return acc
    elif args.is_training == 2 or args.is_training == 3 or args.is_training == 4:

        for ii in range(args.itr):
            setting = 'data_{}_{}_dmodel_{}_head_{}_layer_{}_loss_{}_bs_{}_epoch_{}_posiemb_{}_max_k_{}_relembV_{}_remask_{}_use_temporal_{}_lr_{}'.format(
                args.data_type,
                args.pre_len,
                args.d_model,
                args.head,
                args.layer,
                args.loss,
                args.bs,
                args.epoch,
                args.if_posiemb,
                args.max_k,
                args.relative_v,
                args.remask,
                args.use_his,
                str(args.lr).split('.')[1],
                ii)

            exp = Exp(args)  
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train_with_word2vec(setting)

            torch.cuda.empty_cache()
    else:
        for ii in range(args.itr):
            setting = 'data_{}_{}_dmodel_{}_head_{}_layer_{}_loss_{}_bs_{}_epoch_{}_posiemb_{}_max_k_{}_relembV_{}_remask_{}_use_temporal_{}_lr_{}'.format(
                args.data_type,
                args.pre_len,
                args.d_model,
                args.head,
                args.layer,
                args.loss,
                args.bs,
                args.epoch,
                args.if_posiemb,
                args.max_k,
                args.relative_v,
                args.remask,
                args.use_his,
                str(args.lr).split('.')[1],
                ii)

            exp = Exp(args)  
            print('>>>>>>>start infer : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.infer(setting)

            torch.cuda.empty_cache()
    
if __name__ == "__main__":
    st = time.time()
    main()
    print('spent ',round(time.time()-st,4),'s')


