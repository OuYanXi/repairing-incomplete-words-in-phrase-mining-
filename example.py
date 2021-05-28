import sys
sys.path.append('/home/nlp/yangmengshi/yms_server')  #match_wsplit_dict.py文件路径
import match_wsplit_dict as MWD

inputfile = "./data/gram_rates_pos_2" #'unigram_words_行业1.txt'  #输入文件’\t‘分隔，第一行为列名，第一列为要修复的词汇，列名“words”，其他各列可以为词汇的对应信息，没有列数限制
newgrams_file = '/home/nlp/yangmengshi/yms_server/new_grams'#new_grams文件路径,文件“\t”分隔，第一列为词，第二列为内聚度，第三列为左熵，第四列为右熵，第五列为综合score。
save_path = "./data/adj_pos" #'./save_path'
add_rep_words = MWD.main(inputfile,newgrams_file, save_path)  #,words_top=100) #words_top为inputfile中头部词汇条数，默认为0，即全部词汇。


