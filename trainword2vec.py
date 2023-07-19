
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors


def word2vec2txt(outputfile,trainlist):
    with open(outputfile,'w') as f:
        seqlist = []
        for traj in trainlist:
            while '[PAD]' in traj:
                  traj.remove('[PAD]')
                    
        for traj in trainlist:
            seq = ' '.join(traj) 
            seqlist.append(seq)
        for i,each in enumerate(seqlist):
            if i == 0:
                print(each)
                f.write(each)
            else:
                f.write('\n'+each)
    

def run_word2vec(outputfile, df, d_model, window=5, min_count=1, workers=40, epochs=100):
    train_trajectory = df['trajectory'].tolist()

    word2vec2txt(outputfile,train_trajectory)
    sentences = LineSentence(outputfile)
    model = Word2Vec(sentences, vector_size=d_model, window=5, min_count=1, workers=40, epochs=100)
    model.wv.save_word2vec_format(outputfile.split('.')[0], binary=True)
    return outputfile.split('.')[0] #model path