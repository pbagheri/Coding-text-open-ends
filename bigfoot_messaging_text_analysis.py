# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 15:54:06 2019

@author: payam.bagheri
"""

import pandas as pd
from os import path
from tqdm import tqdm
from difflib import SequenceMatcher
import pickle
import gib_detect_train

dir_path = path.dirname(path.dirname(path.abspath(__file__)))
#print(dir_path)
mesdic = pd.read_excel(dir_path + '/0_input_data/bigfoot_messaging_dic.xlsx')
mesdic.columns
mesdic['statement'][0].split()
mesdic['statement'] = mesdic['statement'].str.lower()

data = pd.read_excel(dir_path + '/0_input_data/messaging-open-end-all-quarters-english-test.xlsx')
data.head()

uniue_words = []
for st in mesdic['statement']:
    #print(st.split())
    for w in st.split():        
        if w not in uniue_words:
            uniue_words.append(w.lower())

uniue_words = pd.DataFrame(uniue_words)


sample_vec = []
uniue_words['codes'] = uniue_words.index

uniue_words_dict = {i: j for (i,j) in zip(uniue_words[0], uniue_words['codes'])}

messaging = data['aw_unaided_ad_message'].str.lower()
len(messaging)

#resp_vecs = pd.DataFrame(index = range(len(uniue_words)+1), columns = ['unique words'] + list(range(len(messaging))))
resp_vecs = pd.DataFrame(index = range(len(uniue_words)), columns = list(range(len(messaging))))
#resp_vecs['unique words'] =  uniue_words[0]
resp_vecs.fillna(0, inplace = True)
resp_vecs.shape
messaging.fillna(0, inplace = True)


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

similar('berger', 'beer')
similar('pric', 'pricing')
similar('fed', 'feed')

print('1/3: Creating word vectors')
for resp in tqdm(range(len(messaging))):
    if str(messaging[resp]) != '0':
        temp = [j for i in messaging[resp].split() for j in uniue_words_dict.keys() if similar(i, j) > 0.81]
        resp_word_codes = [uniue_words_dict[x] for x in temp]
        for i in resp_word_codes:
            resp_vecs[resp][i] = 1
        #resp_vecs[resp][len(uniue_words)] = messaging[resp]
   
     
mesdic_vecs = pd.DataFrame(index = range(len(uniue_words)), columns = list(mesdic['statement']))
mesdic_vecs.fillna(0, inplace = True)

     
for mes in mesdic['statement']:
    #print(mes)
    mes_codes = [uniue_words_dict[x] for x in mes.split()]
    #print(mes_codes)
    for i in mes_codes:
        mesdic_vecs[mes][i] = 1

print('2/3: Calculating overlaps')
resp_dict_dot_products = pd.DataFrame(index = (mesdic_vecs.columns), columns = resp_vecs.columns)
for i in tqdm(resp_vecs.columns):
    for j in mesdic_vecs.columns:
       resp_dict_dot_products[i][j] = sum(resp_vecs[i]*mesdic_vecs[j])

resp_codes = pd.DataFrame(index = resp_vecs.columns, columns = range(10))
resp_codes.fillna(0, inplace = True)

for i in resp_codes.index:
    overlaps = [x for x in enumerate(resp_dict_dot_products[i])]
    overlaps.sort(key=lambda tup: tup[1], reverse=True)
    high_overlaps = [x[0]+1 for x in overlaps[0:10] if x[1] >= 1]
    #print(high_overlaps)
    for j in range(len(high_overlaps)):
        resp_codes[j][i] = high_overlaps[j]

index_list = resp_codes[(resp_codes[0] == 11) | (resp_codes[1] == 11)
                       | (resp_codes[2] == 11) | (resp_codes[3] == 11)
                       | (resp_codes[4] == 11) | (resp_codes[5] == 11)
                       | (resp_codes[6] == 11) | (resp_codes[7] == 11)
                       | (resp_codes[8] == 11) | (resp_codes[9] == 11)].index.tolist()

# to remove code 11 from responses that are too long which probably don't belong to this class 
for i in index_list:
    if len(data['aw_unaided_ad_message'].loc[i].split()) > 5:
        resp_codes.loc[i].replace(11,0,inplace=True)


human_code_cols = ['aw_unaided_ad_message', 'aw_unaided_ad_message_en_sroec1', 'aw_unaided_ad_message_en_sroec2', 'aw_unaided_ad_message_en_sroec3', 'aw_unaided_ad_message_en_sroec4', 'aw_unaided_ad_message_en_sroec5', 'aw_unaided_ad_message_en_sroec6', 'aw_unaided_ad_message_en_sroec7', 'aw_unaided_ad_message_en_sroec8', 'aw_unaided_ad_message_en_sroec9', 'aw_unaided_ad_message_en_sroec10']
for col in human_code_cols:
    resp_codes[col] = data[col]

resp_codes.drop(resp_codes[resp_codes['aw_unaided_ad_message'].isnull()].index, axis=0, inplace=True)

human_code = ['aw_unaided_ad_message_en_sroec1', 'aw_unaided_ad_message_en_sroec2', 'aw_unaided_ad_message_en_sroec3', 'aw_unaided_ad_message_en_sroec4', 'aw_unaided_ad_message_en_sroec5', 'aw_unaided_ad_message_en_sroec6', 'aw_unaided_ad_message_en_sroec7', 'aw_unaided_ad_message_en_sroec8', 'aw_unaided_ad_message_en_sroec9', 'aw_unaided_ad_message_en_sroec10']
computer_code = list(range(10))
resp_codes[human_code].apply(lambda x: x.isin([13]).any(), axis=1)
resp_codes[computer_code].apply(lambda x: x.isin([13]).any(), axis=1)

resp_codes[human_code].apply(lambda x: x.isin([13]).any(), axis=1) == resp_codes[computer_code].apply(lambda x: x.isin([13]).any(), axis=1)


# to deal with bad respondents
resp_codes['aw_unaided_ad_message'] = resp_codes['aw_unaided_ad_message'].astype(str)
resp_codes['bad resp'] = resp_codes[['aw_unaided_ad_message']].apply(lambda x: gib_detect_train.avg_transition_prob(x[0], model_mat) < threshold, axis=1)
resp_codes['resp_len'] = resp_codes[['aw_unaided_ad_message']].apply(lambda x: len(x[0]), axis =1)

index_list2 = resp_codes['bad resp'][resp_codes['resp_len'] == 1].index.tolist()
for i in index_list2:
        resp_codes['bad resp'].loc[i] = True

resp_codes[0][resp_codes['bad resp']] = 18

# Calculating accuracy measures
correct_ratios = pd.DataFrame(index = range(1,29), columns = ['tp','fp','tn','fn','precision', 'recall', 'true negative rate', 'accuracy'])
print('3/3: Calculating accuracy measures')
for c in tqdm(range(1,29)):
    fn = 'fn_' + str(c)
    fp = 'fp_' + str(c)    
    tn = 'tn_' + str(c)
    tp = 'tp_' + str(c)    
    resp_codes[fn] = resp_codes[human_code].apply(lambda x: x.isin([c]).any(), axis=1) > resp_codes[computer_code].apply(lambda x: x.isin([c]).any(), axis=1)
    resp_codes[fp] = resp_codes[human_code].apply(lambda x: x.isin([c]).any(), axis=1) < resp_codes[computer_code].apply(lambda x: x.isin([c]).any(), axis=1)
    resp_codes[tn] = -resp_codes[human_code].apply(lambda x: x.isin([c]).any(), axis=1) & -resp_codes[computer_code].apply(lambda x: x.isin([c]).any(), axis=1)
    resp_codes[tp] = resp_codes[human_code].apply(lambda x: x.isin([c]).any(), axis=1) & resp_codes[computer_code].apply(lambda x: x.isin([c]).any(), axis=1)
    
    correct_ratios['tp'].loc[c] = resp_codes[tp].sum()
    correct_ratios['fp'].loc[c] = resp_codes[fp].sum()
    correct_ratios['tn'].loc[c] = resp_codes[tn].sum()
    correct_ratios['fn'].loc[c] = resp_codes[fn].sum()


correct_ratios['precision'] = correct_ratios['tp']/(correct_ratios['tp'] + correct_ratios['fp'])
correct_ratios['recall'] = correct_ratios['tp']/(correct_ratios['tp'] + correct_ratios['fn'])
correct_ratios['true negative rate'] = correct_ratios['tn']/(correct_ratios['tn'] + correct_ratios['fp'])
correct_ratios['accuracy'] = (correct_ratios['tp']+correct_ratios['tn'])/(correct_ratios['tp'] + correct_ratios['fp']+correct_ratios['tn']+correct_ratios['fn'])




cols_to_drop = list(resp_codes.columns[21:])
resp_codes.drop(cols_to_drop, axis = 1, inplace = True)

'''
mes = pd.DataFrame(index = range(1,29), columns = ['precision', 'recall', 'accuracy'])

mes['precision'] = correct_ratios['tp']/(correct_ratios['tp'] + correct_ratios['fp'])
mes['recall'] = correct_ratios['tp']/(correct_ratios['tp'] + correct_ratios['fn'])
mes['accuracy'] = (correct_ratios['tp']+correct_ratios['tn'])/(correct_ratios['tp'] + correct_ratios['fn']+correct_ratios['tn']+correct_ratios['fp'])


output = mes.to_string(formatters={
    'precision': '{:,.2f}'.format,
    'recall': '{:,.2f}'.format,
    'accuracy': '{:,.2%}'.format
})
'''

#uniue_words.to_csv(dir_path + '/0_output/bigfoot_mess_dic_unique_words.csv')
#resp_vecs.to_csv(dir_path + '/0_output/bigfoot_resp_vecs.csv')
#mesdic_vecs.to_csv(dir_path + '/0_output/bigfoot_mesdic_vecs.csv')
#resp_dict_dot_products.to_csv(dir_path + '/0_output/bigfoot_resp_dict_dot_products.csv')
resp_codes.to_csv(dir_path + '/0_output/bigfoot_resp_codes.csv')
correct_ratios.to_excel(dir_path + '/0_output/bigfoot_accuracy_measures.xlsx')
