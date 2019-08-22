# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:34:02 2019

@author: payam.bagheri
"""


import pandas as pd
from os import path
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.utils import shuffle
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

dir_path = path.dirname(path.dirname(path.abspath(__file__)))


#print(dir_path)
mesdic = pd.read_excel(dir_path + '/0_input_data/bigfoot_messaging_dic.xlsx')
mesdic.columns
mesdic['statement'][0].split()
mesdic['statement'] = mesdic['statement'].str.lower()

all_data = pd.read_excel(dir_path + '/0_input_data/messaging-open-end-all-quarters-english-cleaned.xlsx')
all_data = shuffle(all_data)

# specifying the proportions of the training, validation and test data
valid_size = 0.1
train_percent = 0.8
valid_percent = 0.9

all_data.shape[0]

train_data = all_data[:int(train_percent*all_data.shape[0])]
train_data.shape

data = all_data


min(data.aw_unaided_ad_message_en_sroec1)
data.columns

# measures the similarity of two strings
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# cleans up the messaging responses
def mess_prep(mess):
    tokens = word_tokenize(str(mess))
    words = [word.lower() for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    return words

def takeFirst(elem):
    return elem[0]

def isNaN(num):
    return num != num

# test example ******
similar('the', 'they')

h = ['hormones',
'hermones',
'homorne',
'homorones',
'hormea',
'hormnr',
'hormonse',
'horome',
'horomons']

[similar(x,'hormones') for x in h]
# ******************

# this loop finds the (condensed) unique words in the data and stores their frequency in a dictionary
# the unique words list is a condensed list, i.e. different words that are "too" similar to each other
# are represented by the one word.
unique_words_freq_dic = defaultdict(dict)
unique_words = ['notawordbyitself']
for ind in tqdm(data.index):
    st = data['aw_unaided_ad_message'][ind]
    if isNaN(st):
        st = 'isnan'
    st = mess_prep(data['aw_unaided_ad_message'][ind])
    for w in st:
        #print(w)
        is_unique = [(similar(w.lower(), x), x) for x in unique_words]
        is_unique = sorted(is_unique, key=takeFirst, reverse=True)
        if is_unique[0][0] < 0.7:
            #print(ind)
            unique_words.append(w.lower())
            unique_words_freq_dic[w.lower()][data['aw_unaided_ad_message_en_sroec1'][ind]] = 1
        else:
            try:
                unique_words_freq_dic[is_unique[0][1]][data['aw_unaided_ad_message_en_sroec1'][ind]] += 1
            except KeyError:
                unique_words_freq_dic[is_unique[0][1]][data['aw_unaided_ad_message_en_sroec1'][ind]] = 1
                
len(unique_words)

# creating a new column that contains a condensed version of the text responses where words are replaced by the
# most similar wrod from the list of the condensed unique words created above.
data['aw_unaided_ad_message_dense'] = np.nan
for ind in tqdm(data.index):
    mes = data['aw_unaided_ad_message'][ind]
    prep_mess = mess_prep(mes)
    dense_st = [] # this is a new version of each message were words are replaced by their most similar from unique_words
    for wor in prep_mess:
        #print(wor)
        sims = [(similar(wor, x), x) for x in unique_words]
        sim = sorted(sims, key=takeFirst, reverse=True)
        sim_wor = sim[0][1]
        dense_st.append(sim_wor)
    data['aw_unaided_ad_message_dense'][ind] = dense_st
        #print(sim_wor)

def str_join(lis):
    st = str()
    for i in lis:
        st = " ".join([st,i])
    return st

# joining words to make a sentense
data['aw_unaided_ad_message_dense'] = data['aw_unaided_ad_message_dense'].apply(lambda x: str_join(x))

data.to_csv(dir_path + '/0_output/bigfoot_dense_training_data.csv')       
       
# the following df is going to contain words and their respective frequency under each level
unique_words_freq = pd.DataFrame(unique_words_freq_dic)
unique_words_freq = unique_words_freq.fillna(0)
unique_words_freq = unique_words_freq.transpose()

# replaces a number with n is the number is above n
def if_replace(x,n):
    if x > n:
        r = n
    else:
        r = x
    return r

# calculating TF-IDF for the above unique words
# term frequency
unique_words_freq['idf'] = -1*np.log(unique_words_freq.sum(axis=1)/3000)
for col in unique_words_freq.columns:
    unique_words_freq['tfidf_' + str(col)] = (unique_words_freq[col]/unique_words_freq[col].sum(axis=0))*unique_words_freq['idf']

tfcols = unique_words_freq.columns[26:-1]

unique_words_freq['tfidf'] = unique_words_freq[tfcols].max(axis=1)
unique_words_freq['words'] = unique_words_freq.index

# cuts off the tfids values at 1. 
unique_words_freq['tfidf'] = unique_words_freq['tfidf'].apply(lambda x: if_replace(x,1))

tfidf_dict = dict(zip(unique_words_freq['words'],unique_words_freq['tfidf']))

unique_words_freq.to_csv(dir_path + '/0_output/unique_words_freq.csv')

# max(data.aw_unaided_ad_message[data.aw_unaided_ad_message.notnull()].apply(lambda x: len(mess_prep(str(x)))))


'''
# Load Google's pre-trained Word2Vec model.
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Payam/Downloads/GoogleNews-vectors-negative300.bin', binary=True)  

def get_vec(word):
    try:
        vec = model.get_vector(word)
    except KeyError:
        vec = np.zeros(300)
    return vec
'''

all_messages_tfidf = np.zeros((data.shape[0],50*300+2))


# this loop creates message vectors by including tfidf weights
for i in tqdm(data.index):
    mess = data.aw_unaided_ad_message_dense.loc[i]
    if not isNaN(mess):
        #print(mess)
        messpreped = mess_prep(mess)
        j = 0
        for word in messpreped:
            vec = get_vec(str(word))
            #vec = np.random.rand(300)
            all_messages_tfidf[i][j:j+300] = tfidf_dict[word]*vec # this is where the weighting is implemented
            j += 300
        all_messages_tfidf[i][-2] = data.aw_unaided_ad_message_en_sroec1.loc[i]


all_messages_tfidf = pd.DataFrame(all_messages_tfidf)
all_messages_tfidf.iloc[:,-1] = data.aw_unaided_ad_message
all_messages_tfidf.to_csv(dir_path + '/0_output/bigfoot_message_tfidf_vectors.csv', index=False)        


data_dense = pd.read_csv(dir_path + '/0_output/bigfoot_dense_data.csv')        
all_messages = np.zeros((data_dense.shape[0],50*300+2))

# this loop creates message vectors without including tfidf weights
for i in tqdm(data_dense.index):
    mess = data_dense.aw_unaided_ad_message_dense.loc[i]
    if not isNaN(mess):
        #print(mess)
        messpreped = mess_prep(mess)
        j = 0
        for word in messpreped:
            vec = get_vec(str(word))
            #vec = np.random.rand(300)
            all_messages[i][j:j+300] = vec
            j += 300
        all_messages[i][-2] = data_dense.aw_unaided_ad_message_en_sroec1.loc[i]

all_messages = pd.DataFrame(all_messages)
all_messages.iloc[:,-1] = data_dense.aw_unaided_ad_message

all_messages.to_csv(dir_path + '/0_output/bigfoot_message_vectors.csv', index=False)

'''
vecdata = pd.read_csv(dir_path + '/0_input_data/bigfoot_message_vectors.csv')
vecdata.columns
max(vecdata['15000'])
min(vecdata['15000'])

folder_dataset = dir_path + '/0_input_data'
with open(folder_dataset + "/bigfoot_message_vectors.csv") as f:
    i = 0
    for line in f:
        if i > 0 and i <= 10 :
            # Image path
            print([float(x) for x in line.split(',')[0:15]])        
            # Steering wheel label
            print(line.split(',')[-1])
            #print(type(int(line.split(',')[-1])))
        i += 1


valid_size = 0.1
train_percent = 0.8
valid_percent = 0.9
'''
'''
# convert data to a normalized torch.FloatTensor
#transform = transforms.Compose([transforms.ToTensor()])

# choose the training and test datasets


mess_vecs = pd.read_csv(dir_path + '/0_input_data/bigfoot_message_vectors.csv')
mess_vecs.shape
cols = mess_vecs.columns
num_train_set = int(train_percent*mess_vecs.shape[0])
num_valid_set = int(valid_percent*mess_vecs.shape[0])

train_vecs = mess_vecs[cols][0:num_train_set]
valid_vecs = mess_vecs[cols][num_train_set:num_valid_set]
test_vecs = mess_vecs[cols][num_valid_set:]
test_vecs.shape

train_vecs.to_csv(dir_path + '/0_input_data/bigfoot_train_vectors.csv', index=False)
valid_vecs.to_csv(dir_path + '/0_input_data/bigfoot_valid_vectors.csv', index=False)
test_vecs.to_csv(dir_path + '/0_input_data/bigfoot_test_vectors.csv', index=False)



import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

train_dataset = dir_path + '\\0_input_data\\bigfoot_train_vectors.csv'
valid_dataset = dir_path + '\\0_input_data\\bigfoot_valid_vectors.csv'
test_dataset = dir_path + '\\0_input_data\\bigfoot_test_vectors.csv'


# the 1st list is the levels that are gonna keep their identity (although relabelled) and
# the 2nd list are the levels that are gonna be combined into one level
big_levels = [1, 8, 9]
mid_levels =[13, 5, 17, 16, 7, 2, 25, 4]
big_level_conv_dic = {1:1, 8:1, 9:1} 
mid_level_conv_dic = {13:2, 5:2, 17:2, 16:2, 7:2, 2:2, 25:2, 4:2}
to_be_combined_level = [14, 11, 18, 19, 3, 10, 12, 24, 6, 20, 15, 23, 26, 21, 22]

class DriveData(Dataset):
    def __init__(self, datasetf, transform=None):
        self.__xs = []
        self.__ys = []
        self.transform = transform
        # Open and load text file including the whole training data
        with open(datasetf) as f:
            #print(datasetf)
            i = 0
            for line in f:
                # the following i>0 is for skipping the first line which is the column names 
                if i > 0:
                    # checked the resizing to 50*300 and it's correct
                    #self.__xs.append(torch.from_numpy(np.asarray([float(x) for x in line.split(',')[0:15000]])).view(1,50,300))
                    self.__xs.append(torch.from_numpy(np.asarray([float(x) for x in line.split(',')[0:15000]])))
                    #if i == 5:
                        #print(i, self.__xs)
                    self.__ys.append(line.split(',')[-1])
                i += 1
        f.close()

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        message = self.__xs[index]
        if self.transform is not None:
            message = self.transform(message)

        # Convert image and label to torch tensors
        #message = torch.from_numpy(np.asarray(message))
        # the subtraction of 1 is to make the target values range from 0 to 25 instead of 1 to 26
        label = torch.from_numpy(np.asarray(int(self.__ys[index])-1).reshape(1))

#        if int(self.__ys[index]) in big_levels:
#            label = torch.from_numpy(np.asarray(big_level_conv_dic[int(self.__ys[index])]-1).reshape(1))
#        elif int(self.__ys[index]) in mid_levels:
#            label = torch.from_numpy(np.asarray(mid_level_conv_dic[int(self.__ys[index])]-1).reshape(1))
#        else:
#            label = torch.from_numpy(np.asarray(3-1).reshape(1))

        label = label.long()
        return message, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)
    

 
# number of subprocesses to use for data loading
num_workers = 0
batch_size = 20


# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([transforms.ToTensor()])

#indices = list(range(num_train))
#np.random.shuffle(indices)
#split = int(np.floor(valid_size * num_train))
#train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
#train_sampler = SubsetRandomSampler(train_idx)
#valid_sampler = SubsetRandomSampler(valid_idx)

train_data = DriveData(train_dataset, transform=None)
valid_data = DriveData(valid_dataset, transform=None)
test_data = DriveData(test_dataset, transform=None)

#train_loader = DataLoader(train_data, batch_size=1, num_workers=num_workers)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_data, batch_size=1, num_workers=num_workers)

loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}



# specify the image classes
#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']     

# define the CNN architecture
# Each word is represented by a vector of length 300. The input matrix to the
# following matrix is a 40x300 matrix meaning each message can have up to a 
# length of 40 words. The three conv layers act on the input independently
# and then the 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 50x300x1 tensor)
        self.conv1 = nn.Conv2d(1, 1, (1,300), padding=0)
        self.conv2 = nn.Conv2d(1, 1, (2,300), padding=0)
        self.conv3 = nn.Conv2d(1, 1, (3,300), padding=0)
        self.conv4 = nn.Conv2d(1, 1, (4,300), padding=0)
        self.conv5 = nn.Conv2d(1, 1, (5,300), padding=0)
        self.conv6 = nn.Conv2d(1, 1, (6,300), padding=0)
        self.conv7 = nn.Conv2d(1, 1, (7,300), padding=0)
        self.conv8 = nn.Conv2d(1, 1, (8,300), padding=0)
        self.conv9 = nn.Conv2d(1, 1, (9,300), padding=0)
        self.conv10 = nn.Conv2d(1, 1, (10,300), padding=0)
        self.conv11 = nn.Conv2d(1, 1, (11,300), padding=0)
        self.conv12 = nn.Conv2d(1, 1, (12,300), padding=0)
        self.conv13 = nn.Conv2d(1, 1, (13,300), padding=0)
        self.conv14 = nn.Conv2d(1, 1, (14,300), padding=0)
        self.conv15 = nn.Conv2d(1, 1, (15,300), padding=0)
        self.conv16 = nn.Conv2d(1, 1, (16,300), padding=0)
        self.conv17 = nn.Conv2d(1, 1, (17,300), padding=0)
        self.conv18 = nn.Conv2d(1, 1, (18,300), padding=0)
        self.conv19 = nn.Conv2d(1, 1, (19,300), padding=0)
        self.conv20 = nn.Conv2d(1, 1, (20,300), padding=0)
        self.conv21 = nn.Conv2d(1, 1, (21,300), padding=0)
        self.conv22 = nn.Conv2d(1, 1, (22,300), padding=0)
        self.conv23 = nn.Conv2d(1, 1, (23,300), padding=0)
        self.conv24 = nn.Conv2d(1, 1, (24,300), padding=0)
        self.conv25 = nn.Conv2d(1, 1, (25,300), padding=0)
        self.conv26 = nn.Conv2d(1, 1, (26,300), padding=0)
        self.conv27 = nn.Conv2d(1, 1, (27,300), padding=0)
        self.conv28 = nn.Conv2d(1, 1, (28,300), padding=0)
        self.conv29 = nn.Conv2d(1, 1, (29,300), padding=0)
        self.conv30 = nn.Conv2d(1, 1, (30,300), padding=0)
        self.conv31 = nn.Conv2d(1, 1, (31,300), padding=0)
        self.conv32 = nn.Conv2d(1, 1, (32,300), padding=0)
        self.conv33 = nn.Conv2d(1, 1, (33,300), padding=0)
        self.conv34 = nn.Conv2d(1, 1, (34,300), padding=0)
        self.conv35 = nn.Conv2d(1, 1, (35,300), padding=0)
        self.conv36 = nn.Conv2d(1, 1, (36,300), padding=0)
        self.conv37 = nn.Conv2d(1, 1, (37,300), padding=0)
        self.conv38 = nn.Conv2d(1, 1, (38,300), padding=0)
        self.conv39 = nn.Conv2d(1, 1, (39,300), padding=0)
        self.conv40 = nn.Conv2d(1, 1, (40,300), padding=0)
        self.conv41 = nn.Conv2d(1, 1, (41,300), padding=0)
        self.conv42 = nn.Conv2d(1, 1, (42,300), padding=0)
        self.conv43 = nn.Conv2d(1, 1, (43,300), padding=0)
        self.conv44 = nn.Conv2d(1, 1, (44,300), padding=0)
        self.conv45 = nn.Conv2d(1, 1, (45,300), padding=0)
        self.conv46 = nn.Conv2d(1, 1, (46,300), padding=0)
        self.conv47 = nn.Conv2d(1, 1, (47,300), padding=0)
        self.conv48 = nn.Conv2d(1, 1, (48,300), padding=0)
        self.conv49 = nn.Conv2d(1, 1, (49,300), padding=0)
        self.conv50 = nn.Conv2d(1, 1, (50,300), padding=0)

        # max pooling layer
        self.pool1 =  nn.MaxPool2d(1,50)
        self.pool2 =  nn.MaxPool2d(1,49)
        self.pool3 =  nn.MaxPool2d(1,48)
        self.pool4 =  nn.MaxPool2d(1,47)
        self.pool5 =  nn.MaxPool2d(1,46)
        self.pool6 =  nn.MaxPool2d(1,45)
        self.pool7 =  nn.MaxPool2d(1,44)
        self.pool8 =  nn.MaxPool2d(1,43)
        self.pool9 =  nn.MaxPool2d(1,42)
        self.pool10 = nn.MaxPool2d(1,41)
        self.pool11 = nn.MaxPool2d(1,40)
        self.pool12 = nn.MaxPool2d(1,39)
        self.pool13 = nn.MaxPool2d(1,38)
        self.pool14 = nn.MaxPool2d(1,37)
        self.pool15 = nn.MaxPool2d(1,36)
        self.pool16 = nn.MaxPool2d(1,35)
        self.pool17 = nn.MaxPool2d(1,34)
        self.pool18 = nn.MaxPool2d(1,33)
        self.pool19 = nn.MaxPool2d(1,32)
        self.pool20 = nn.MaxPool2d(1,31)
        self.pool21 = nn.MaxPool2d(1,30)
        self.pool22 = nn.MaxPool2d(1,29)
        self.pool23 = nn.MaxPool2d(1,28)
        self.pool24 = nn.MaxPool2d(1,27)
        self.pool25 = nn.MaxPool2d(1,26)
        self.pool26 = nn.MaxPool2d(1,25)
        self.pool27 = nn.MaxPool2d(1,24)
        self.pool28 = nn.MaxPool2d(1,23)
        self.pool29 = nn.MaxPool2d(1,22)
        self.pool30 = nn.MaxPool2d(1,21)
        self.pool31 = nn.MaxPool2d(1,20)
        self.pool32 = nn.MaxPool2d(1,19)
        self.pool33 = nn.MaxPool2d(1,18)
        self.pool34 = nn.MaxPool2d(1,17)
        self.pool35 = nn.MaxPool2d(1,16)
        self.pool36 = nn.MaxPool2d(1,15)
        self.pool37 = nn.MaxPool2d(1,14)
        self.pool38 = nn.MaxPool2d(1,13)
        self.pool39 = nn.MaxPool2d(1,12)
        self.pool40 = nn.MaxPool2d(1,11)
        self.pool41 = nn.MaxPool2d(1,10)
        self.pool42 = nn.MaxPool2d(1,9)
        self.pool43 = nn.MaxPool2d(1,8)
        self.pool44 = nn.MaxPool2d(1,7)
        self.pool45 = nn.MaxPool2d(1,6)
        self.pool46 = nn.MaxPool2d(1,5)
        self.pool47 = nn.MaxPool2d(1,4)
        self.pool48 = nn.MaxPool2d(1,3)
        self.pool49 = nn.MaxPool2d(1,2)
        self.pool50 = nn.MaxPool2d(1,1)
        # linear layer (7 -> 500)
        self.fc0 = nn.Linear(15000, 300)
        self.fc1 = nn.Linear(300, 30)
        #self.fc2 = nn.Linear(1500, 500)
        #self.fc3 = nn.Linear(3000, 1500)
        #self.fc4 = nn.Linear(1500, 500)
        #self.fc5 = nn.Linear(500, 100)
        self.fc6 = nn.Linear(30, 30)
        #self.sig = nn.Sigmoid()
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        '''
        '''
        x1 =   self.pool1(torch.abs(F.tanh(self.conv1(x))))
        x2 =   self.pool2(torch.abs(F.tanh(self.conv2(x))))
        x3 =   self.pool3(torch.abs(F.tanh(self.conv3(x))))
        x4 =   self.pool4(torch.abs(F.tanh(self.conv4(x))))
        x5 =   self.pool5(torch.abs(F.tanh(self.conv5(x))))
        x6 =   self.pool6(torch.abs(F.tanh(self.conv6(x))))
        x7 =   self.pool7(torch.abs(F.tanh(self.conv7(x))))
        x8 =   self.pool8(F.tanh(self.conv8(x)))
        x9 =   self.pool9(F.tanh(self.conv9(x)))
        x10 = self.pool10(F.tanh(self.conv10(x)))
        x11 = self.pool11(F.tanh(self.conv11(x)))
        x12 = self.pool12(F.tanh(self.conv12(x)))
        x13 = self.pool13(F.tanh(self.conv13(x)))
        x14 = self.pool14(F.tanh(self.conv14(x)))
        x15 = self.pool15(F.tanh(self.conv15(x)))
        x16 = self.pool16(F.tanh(self.conv16(x)))
        x17 = self.pool17(F.tanh(self.conv17(x)))
        x18 = self.pool18(F.tanh(self.conv18(x)))
        x19 = self.pool19(F.tanh(self.conv19(x)))
        x20 = self.pool20(F.tanh(self.conv20(x)))
        x21 = self.pool21(F.tanh(self.conv21(x)))
        x22 = self.pool22(F.tanh(self.conv22(x)))
        x23 = self.pool23(F.tanh(self.conv23(x)))
        x24 = self.pool24(F.tanh(self.conv24(x)))
        x25 = self.pool25(F.tanh(self.conv25(x)))
        x26 = self.pool26(F.tanh(self.conv26(x)))
        x27 = self.pool27(F.tanh(self.conv27(x)))
        x28 = self.pool28(F.tanh(self.conv28(x)))
        x29 = self.pool29(F.tanh(self.conv29(x)))
        x30 = self.pool30(F.tanh(self.conv30(x)))
        x31 = self.pool31(F.tanh(self.conv31(x)))
        x32 = self.pool32(F.tanh(self.conv32(x)))
        x33 = self.pool33(F.tanh(self.conv33(x)))
        x34 = self.pool34(F.tanh(self.conv34(x)))
        x35 = self.pool35(F.tanh(self.conv35(x)))
        x36 = self.pool36(F.tanh(self.conv36(x)))
        x37 = self.pool37(F.tanh(self.conv37(x)))
        x38 = self.pool38(F.tanh(self.conv38(x)))
        x39 = self.pool39(F.tanh(self.conv39(x)))
        x40 = self.pool40(F.tanh(self.conv40(x)))
        x41 = self.pool41(F.tanh(self.conv41(x)))
        x42 = self.pool42(F.tanh(self.conv42(x)))
        x43 = self.pool43(F.tanh(self.conv43(x)))
        x44 = self.pool44(F.tanh(self.conv44(x)))
        x45 = self.pool45(F.tanh(self.conv45(x)))
        x46 = self.pool46(F.tanh(self.conv46(x)))
        x47 = self.pool47(F.tanh(self.conv47(x)))
        x48 = self.pool48(F.tanh(self.conv48(x)))
        x49 = self.pool49(F.tanh(self.conv49(x)))
        x50 = self.pool50(F.tanh(self.conv50(x)))
        # flatten image input
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
                       x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, 
                       x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, 
                       x31, x32, x33, x34, x35, x36, x37, x38, x39, x40, 
                       x41, x42, x43, x44, x45, x46, x47, x48, x49, x50),0)
        
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7),0)
        x = x.view(-1, 7)        
        #x = self.dropout(x)
        '''
        '''
        #x = F.relu(self.fc1(x))
        x = F.relu(self.fc0(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc4(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc5(x))
        x = self.fc6(x)
        #x = self.sig(x)
        return x

def weights_init_normal(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = (1.0/np.sqrt(n))
        m.weight.data.normal_(0, y)
        m.bias.data.fill_(0)

# create a complete CNN
model = Net()
model = model.float() 
print(model)

model.apply(weights_init_normal)


import torch.optim as optim
import torch.nn as nn

### loss function
# It is useful when training a classification problem with C classes. 
# If provided, the optional argument weight should be a 1D Tensor assigning 
# weight to each of the classes. This is particularly useful when you have an 
# unbalanced training set.
#ratios = [0.2360, 0.0425, 0.0130, 0.0370, 0.0603, 0.0045, 0.0464, 0.1332, 0.0827, 0.0100, 0.0272, 0.0084, 0.0749, 0.0282, 0.0010, 0.0535, 0.0561, 0.0207, 0.0172, 0.0013, 0.0006, 0.0000, 0.0010, 0.0052, 0.0382, 0.0010]
ratios = [0.235980551, 0.133225284, 0.082658023, 0.074878444, 0.060291734, 0.056077796, 0.053484603, 0.046353323, 0.042463533, 0.038249595, 0.036952998, 0.139384117]
ratios_inv = [max(ratios)/(x+0.0001) for x in ratios]
weights = torch.Tensor(ratios)

#criterion = nn.CrossEntropyLoss(weight=weights)

criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()

### optimizer
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)     

def train(n_epochs, loaders, model, optimizer, new_lr, criterion, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    valid_loss_prev = 0
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0         
        lower_lr = False
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            #print(batch_idx)
            if lower_lr == True:
                optimizer = optim.SGD(model.parameters(), lr=new_lr)
            ## find the loss and update the model parameters accordingly
            #print(new_lr)
            optimizer.zero_grad()
            output = model(data.float())
            #print(target.shape)
            target = target.squeeze_()
            #print(output.shape)
            
            loss = criterion(output, target)
            #print(loss)
        
            loss.backward()
        
            optimizer.step()
        
            #train_loss += loss.item()*data.size(0)
            ## record the average training loss, using something like
            train_loss += loss.data
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            ## update the average validation loss
            output = model(data.float())
            target = target.squeeze_()
            loss = criterion(output, target)
            valid_loss += loss.data
    
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))

        ## save the model if validation loss has decreased
        if valid_loss_prev < valid_loss:
            #print('lr is being reduced at epoch %d' % epoch)
            new_lr = new_lr*0.95
            lower_lr = True
        valid_loss_prev = valid_loss
        
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), save_path)
            model_min = model
            
    # return trained model
    return model_min


# train the model
#model_test = train(600, loaders, model, optimizer, lr, criterion, 'C:/Users/Payam/Dropbox/0_output/bigfoot_messaging_NN/model_messaging.pt')    

print('Training of the model starts!')
model_mess = train(200, loaders, model, optimizer, lr, 
                      criterion, 'C:/Users/Payam/Dropbox/0_output/bigfoot_messaging_NN/model_messaging.pt')




#import pandas as pd
#import torch
#from tqdm import tqdm

#model_mess = torch.load('C:/Users/Payam/Dropbox/0_output/bigfoot_messaging_NN/model_messaging.pt')

def takeSecond(elem):
    return elem[1]
    


def softmax(l):
    summ = sum([np.exp(i) for i in l])
    sl = [np.exp(i)/summ for i in l]
    return sl

model_mess.eval()
test_results = []
test_loss = 0
for batch_idx, (data, target) in tqdm(enumerate(loaders['test'])):
    ## update the average validation loss
    output = model_mess(data.float())
    target = target.squeeze_()
    #loss = criterion(output, target)
    #print(output.shape,target.shape)
    output = softmax(output.tolist()[0])
    indexed_output = [(i,x) for i, x in zip(range(1,13), output)]
    indexed_output = [x if x[1] > 0.05 else (0,x[1]) for x in indexed_output ]
    indexed_output_sorted = sorted(indexed_output, key=takeSecond, reverse=True)
    probs = [x[1] for x in indexed_output_sorted]
    labels = [x[0] for x in indexed_output_sorted]
    probs.extend(labels[0:10])
    test_results.append(probs)
    #test_loss += loss.data

#test_loss = test_loss/len(test_loader.dataset)
#print(test_loss) 
    
test_results=pd.DataFrame(test_results,columns=range(22))
#test_results[1] = test_results[0].apply(lambda x: 1-x))
    
test_results.head()
test_results.to_csv(dir_path + '/0_output/res.csv')
'''
