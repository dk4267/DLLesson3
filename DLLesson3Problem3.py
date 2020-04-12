from keras.models import Sequential
from keras import layers
from keras.layers import Flatten
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups



newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)
#tokenizing data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(newsgroups_train)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(newsgroups_train)

#le = preprocessing.LabelEncoder()
#y = le.fit_transform(newsgroups_test)
X_train, X_test, y_train, y_test = train_test_split(sentences, newsgroups_test, test_size=0.25, random_state=1000)
# Number of features
# print(input_dim)
model = Sequential()
#model.add(layers.Dense(300, input_dim=2000, activation='relu')) #input matches initial number of words
model.add(layers.Embedding(len(tokenizer.word_index) + 1, 50, input_length=2000))
model.add(Flatten())
model.add(layers.Dense(5, activation='softmax')) #changed layers to match inputd, and activation to softmax
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)