#!/usr/bin/env python
# coding: utf-8


import os
import sys
import re
import math
import pandas as pd
import numpy as np
import jieba 
import time
import shutil
import warnings
warnings.filterwarnings("ignore")

#start_time = time.time()
'''清空非空目录'''
def CleanDir( Dir ):
    if os.path.isdir( Dir ):
        paths = os.listdir( Dir )
        for path in paths:
            filePath = os.path.join( Dir, path )
            if os.path.isfile( filePath ):
                try:
                    os.remove( filePath )
                except os.error:
                    print( "remove %s error." %filePath )#引入logging
            elif os.path.isdir( filePath ):
#                 if filePath[-4:].lower() == ".svn".lower():
#                     continue
                shutil.rmtree(filePath,True)

'''创建新的保存目录'''
def add_new_path(path):
    if os.path.exists(path):
        pass    
    else:
        os.mkdir(path)
    return path

'''判断是否为数字字符'''
def not_number(s):
    try:
        float(s)
        return False
    except:
        return True

print('当前执行路径：',os.getcwd())

# In[17]:

'''读取nz_data'''
def read_nz_data(path, key_list, top_new_grams=0):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    f = open(path,'r',encoding='utf-8')
    words_nz = {}
    if top_new_grams == 0:
        top_new_grams = -1  
    for line in f.readlines()[0:top_new_grams]: #2500万条数据
        if len(line):
            line1 = line.strip('\n') 
            linel = line1.split('\t') #[word, dop, l_en, r_en, score]
            word = linel[0]
            try:
                float(linel[1]) and float(linel[2]) and float(linel[3]) and \
                float(linel[4]) 
#                if float(linel[2])==0 or float(linel[3])==0:
#                    print(linel)
                string = cop.sub("", word)
                if len(string) and (not_number(string)):
                    word_nz = { key_list[index]: float(value) for (index, value) \
                               in enumerate(linel[1:5]) }
                    words_nz[string] = word_nz
            except:
                print('丢弃的错误词汇：',linel)
    return words_nz


# In[12]:


'''为新提取的特征词匹配左右熵'''
def find_dop_en(words_cos_tf, words_nz, key_list):
    words_cos_tf = words_cos_tf.set_index(['words'])
    for col in key_list:
        words_cos_tf[col] = np.nan
    words_add = list(words_cos_tf.index)
    for word in words_add:
        word_nz = words_nz.get(word, None)
        if word_nz:
            for col in key_list:
                value = word_nz.get(col, np.nan)
                words_cos_tf.loc[word,col] = value
    words_cos_tf['diff_en'] = words_cos_tf['l_en'] - words_cos_tf['r_en']
    return words_cos_tf

# In[20]:
'''删除左右熵差大的词，减少匹配遍历'''
def del_wsplit_words_nz(words_nz):
    words = list(words_nz.keys())
    #print(words[0:10])
    for w in words:
        diff_en = words_nz[w]['l_en'] - words_nz[w]['r_en']
        if abs(diff_en) > 0.5:
            del words_nz[w]
    return words_nz

#%%
'''进行错词匹配'''
def match_add_reco(add_diff_en, words_nz):
    #add_diff_en中words作为index,diff_en in columns
    words_reco = list(words_nz.keys())#识别新词列表 按score值降序排列的前提
    add_replace_map = {}
    add_nomatch = {}
    words_wsplit = list(add_diff_en.index.values) 
    
    i = 1 #用于计数
    for ww in words_wsplit:
        #try:
        if len(ww):
            diff_en_ww = add_diff_en.loc[ww,'diff_en']
            l_ww = len(ww)
            
            for wr in words_reco:
                if ww in wr:   
                    
                    if diff_en_ww > 0:  #diff_en_ww的绝对值大于阈值 右熵小往右扩充                            
                        if ww in wr[0:l_ww]: #包含ww==wr的情况
                            add_replace_map[ww] = wr
                            break
                    if diff_en_ww < 0:  #diff_en_ww的绝对值大于阈值 左熵小往左扩充                
                        if ww in wr[-l_ww:]: #包含ww==wr的情况
                            add_replace_map[ww] = wr
                            break
#        except:
#            print('当前匹配出错的错切词为：',ww)
                    
        match = add_replace_map.get(ww,0)
        if match == 0:
            add_nomatch[ww] = 1
           
        i = i + 1
        if i % 1000 == 0:
            print('正在匹配第%d千个错切词'%(i//1000))
    return add_nomatch, add_replace_map


#%%
'''删掉匹配结尾的‘的，了’等'''
def clean_rep_map(words_rep,stopwords):
    rep_map = pd.DataFrame(words_rep,index=['rep_words']).T
    rep_map['rep_words_1'] = rep_map['rep_words'].apply(lambda x:x[1:] if x[0] in \
           stopwords else x)
    rep_map['rep_words_2'] = rep_map['rep_words_1'].apply(lambda x:x[0:-1] if x[-1] in\
           stopwords else x)
    rep_map = rep_map[['rep_words_2']]
    rep_map = rep_map.reset_index().rename(columns={'index':'words','rep_words_2':\
                                 'rep_words'})
    rep_map = rep_map.drop_duplicates(subset=['words'])
    rep_map = rep_map[rep_map.rep_words.apply(lambda x: len(x)>1)]
    rep_map = rep_map.loc[rep_map.words!=rep_map.rep_words,:]
    return rep_map

#%%
'''整合和保存错切词匹配map'''
def find_info_for_map(add_diff_en1,words_nz):
    rep_map = add_diff_en1.copy()
    for col in ['dop1','l_en1','r_en1','diff_en1','score1']:
        rep_map[col] = np.nan
        
    for i in rep_map.index:
            wr = rep_map.loc[i,'rep_words']
            try:
                if len(wr) > 1:         
                    nz_wr = words_nz[wr]
                    for col in ['dop1','l_en1','r_en1','score1']:
                        rep_map.loc[i,col] = nz_wr[col[0:-1]]
            except:
                print('保存时匹配不到左右熵的词：',wr)
        
    rep_map['diff_en1'] = rep_map['l_en1'] - rep_map['r_en1']
    return rep_map

#%%
''''数据整合保存'''
def data_integrate_tosave(add_nz,words_rep_nsame,words_nomatch):
    add_data = add_nz.reset_index().rename(columns={'index':'words'}) #保证words in add_nz.columns
    add_data['words'] = add_data['words'].apply(lambda x: words_rep_nsame.get(x,x))
    #add_rep_words = add_data.replace({'words':words_rep_nsame})
    words_rep_nsame = pd.DataFrame(words_rep_nsame,index=['words']).T.reset_index()
    words_rep_nsame.columns = ['raw_words','words']
    words_nomatch = pd.DataFrame(words_nomatch,index=['nomatch']).T.reset_index()
    words_nomatch.columns = ['words','nomatch']
    add_rep_words = pd.merge(add_data,words_rep_nsame,on='words',how='left')
    add_rep_words = pd.merge(add_rep_words,words_nomatch,on='words',how='left')
    add_rep_words = add_rep_words.drop_duplicates(['words'])
    return add_rep_words


#%%

def main(feature_inputfile,newgrams_file,save_path=0,words_top=0):
    if save_path==0:
        save_path = './'
    else:
        save_path = add_new_path(save_path)
    print('读取待处理的特征词')
    #通常为做过词义相似度计算的特征词
    words_df = pd.read_table(feature_inputfile)
    if words_top:
        words_df = words_df[0:words_top]
    print('需要处理的特征词量：',len(words_df))
    
    print('读取nz_data')
    key_list = ['dop', 'l_en', 'r_en', 'score'] #跟输入文件的格式有关
    words_nz = read_nz_data(newgrams_file, key_list,top_new_grams=0)
    print('美白 ：', words_nz.get('美白', '该词不存在'))
    print('清洗后识别新词个数：', len(words_nz))
  
    print('为新提取的特征词匹配左右熵')
    add_nz = find_dop_en(words_df, words_nz, key_list)           
    print('未找到左右熵的特征词量：',len(add_nz[add_nz['diff_en'].isnull()]))
    
    print('根据大左右熵差，选择切错的词汇')
    threshold_wsplit = 0.5
    add_diff_en = add_nz[(add_nz['diff_en']>threshold_wsplit) | (add_nz['diff_en']<-threshold_wsplit)]  #左右熵阈值可以选择
    print('阈值为%.2f时错切词汇量为: %d' % (threshold_wsplit, len(add_diff_en)))
    
    print('删除words_nz中左右熵差大的词，减少错词匹配时的遍历')
    words_nz = del_wsplit_words_nz(words_nz)
    print('删除左右熵差大的词后识别新词个数：', len(words_nz))
    
    print('为错切词匹配做正确匹配')            
    words_nomatch, words_rep = match_add_reco(add_diff_en, words_nz) 
    print('匹配到的错切词量：',len(words_rep))
    print('没有匹配到的错切词量：',len(words_nomatch))
    #print('没有匹配到的错切词：',words_nomatch)
    
    print('删掉匹配结尾的‘的，了’等')
    del_words = ['的','后','何','是','和','啊','都','很','太','呀','最','吧','更',
                 '啦','要','了','也','于','得']
    rep_map = clean_rep_map(words_rep,del_words)
    words_rep_nsame = {rep_map.loc[i,'words']:rep_map.loc[i,'rep_words'] for i in \
                       rep_map.index }
    
    print('整合和保存文件')
    add_diff_en = add_diff_en.reset_index().rename(columns={'index':'words'})
    add_diff_en = pd.merge(add_diff_en,rep_map,on='words',how='outer') 
    
    add_diff_en = find_info_for_map(add_diff_en,words_nz)
    add_diff_en.to_csv(save_path+'/rep_map.csv',index=False,encoding='utf_8_sig') #注意修改保存位置
    
    add_rep_words = data_integrate_tosave(add_nz,words_rep_nsame,words_nomatch)
    add_rep_words.to_csv(save_path+'/words_rep_wsplit.csv',encoding='utf_8_sig')
    words_no_en = add_rep_words[add_rep_words.diff_en.isnull()]
    words_no_en.to_csv(save_path+'/words_no_en.csv',index=False,encoding='utf_8_sig')
    print('程序运行完毕')
    return add_rep_words
#%%
if __name__ == '__main__':
    
    feature_extract_method = 'tf-idf'
    inputfile = 'words_cos_'+feature_extract_method+'.csv'
    newgrams_file = 'new_grams'
    add_rep_words = main(inputfile,newgrams_file)
#    outfile = 'words_rep_wsplit.csv'
#    add_rep_words.to_csv(outfile,index=False,encoding='utf_8_sig')
    


# In[ ]:
 




