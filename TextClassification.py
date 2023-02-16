# encoding=utf-8
# Programmer: YiHsiuYen
# Date: 2021/11/13
# 使用 TextCNN 對對話語句進行分類
# 版本: v1
# 訓練準確率
# 測試準確率
# 執行時間: 每個 Epoch  秒,  Epochs 需時  分


import os
import sys
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPooling2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from keras.layers import MaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# the dataset path
Text_Data_Dir = r''
# the path for Glove embeddings
GLOVE_DIR = r''
# make the max word length to be constant
Max_Word = 10000
Max_Sequence_Length = 1000
# the percentage of train test split to be applied
Validation_Split = 0.20
# filter dimension of vectors to be used
Embedding_Dim = 100
# filter sizes of the different conv layers
filter_sizes = [3, 4, 5]
num_filters = 512
embedding_dim = 100
# dropout probability
dropout = 0.5
batch_size = 30
epochs = 2

# preparing dataset
# the list of text samples
texts = []
# dictionary mapping label name to numeric id
labels_index = {}
# the list of label ids
labels = []

for name in sorted(os.listdir(Text_Data_Dir)):
    path = os.path.join(Text_Data_Dir, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info  < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='utf-8')
                t = f.read()
                i = t.find('\n\n') # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)
print(labels_index)

print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(num_words = Max_Word)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("unique words : {}".format(len(word_index)))

data = pad_sequences(sequences, maxlen=Max_Sequence_Length)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(Validation_Split * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
