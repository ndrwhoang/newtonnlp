#https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
# Neccessities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import accuracy_score, confusion_matrix
# Keras library
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical
# NLTK
import nltk
from nltk.stem import WordNetLemmatizer


df_math = pd.read_csv('mathdataset.csv', encoding = 'utf-8')
df_science = pd.read_csv('sciencedataset.csv', encoding = 'utf-8')
df_religious = pd.read_csv('religiousdataset.csv', encoding = 'utf-8')
df_mint = pd.read_csv('mintdataset.csv', encoding = 'utf-8')
# Downsampling Mint text
df_mint = df_mint.sample(80, random_state = 100)

df_math['type'] = 'Math'
df_science['type'] = 'Science'
df_religious['type'] = 'Religious'
df_mint['type'] = 'Mint'

df = df_math.append([df_science, df_religious, df_mint])


# Minor cleaning
df = df.drop(df.columns[0], axis=1)
df.fillna('NA', inplace = True)
df = df.loc[df.language1 == 'English']
df = df[['title', 'full_text', 'type']].copy()
df['type'].value_counts()

# One hot encoding
oh_encode = pd.get_dummies(df['type'])
df = df.drop('type', axis = 1)
df = df.join(oh_encode)
list(df)

# Splitting train / test
train = df.sample(frac = 0.7, random_state = 100)
test = df.drop(train.index)

# Setting target variable
list_types = ['Math', 'Mint', 'Religious', 'Science']
y_train = train[list_types].values
y_test = test[list_types].values

# Nlp Preprocessing
list_sentences_train = train['full_text']
list_sentences_test = test['full_text']

# Lemmatizing
lemmatizer = WordNetLemmatizer()
lemmatized_text = [lemmatizer.lemmatize(i) for i in list_sentences_train]

# Tokenizing
max_features = 15000
tokenizer = Tokenizer(num_words = max_features)
# Engineering features from train text
tokenizer.fit_on_texts(lemmatized_text)

# Applying features created to train and test set
list_tokenized_train = tokenizer.texts_to_matrix(list_sentences_train, mode = 'tfidf')
list_tokenized_test = tokenizer.texts_to_matrix(list_sentences_test, mode = 'tfidf')


# Building model
# Callback funstions
# Note: look into model checkpoint
callbacks = [EarlyStopping(monitor = 'val_loss', 
                           patience = 2, 
                           min_delta = 0, 
                           mode = 'min')]
# Input layer
inp = Input(shape=(max_features, ))
# Embedding layer
embed_size = 1024
x = Embedding(max_features, embed_size)(inp)
# LSTM layer
x = LSTM(256, return_sequences = True, name = 'lstm_layer')(x)
# Pooling (reshaping the tensor from 3D to 2D)
x = GlobalMaxPool1D()(x)
# Dropout layer
x = Dropout(0.5)(x)
# Dense layer
x = Dense(128, activation = 'relu')(x)
# 1 more dropout layer
x = Dropout(0.2)(x)
# Output layer
x = Dense(4, activation = 'softmax')(x)
# Compiling model
model = Model(inputs = inp, outputs = x)
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adadelta',
              metrics = ['acc'])

# Fitting model
batch_size = 8
epochs = 20
model.fit(x = list_tokenized_train,
          y = y_train,
          batch_size = batch_size,
          epochs = epochs,
          validation_split = 0.2,
#          callbacks = callbacks,
          verbose = 1)

# Making predictions
prediction = model.predict(list_tokenized_test,
                       batch_size = batch_size,
                       verbose = 1)
predicted = np.argmax(prediction, axis = 1)
y_test_test = np.argmax(y_test, axis = 1)
accuracy_score(y_test_test, predicted)
confusion_matrix(y_test, predicted,
                 labels = ['Math', 'Mint', 'Religious', 'Science'])
