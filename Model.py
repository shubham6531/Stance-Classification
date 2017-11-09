import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()
import keras

## open and extract file
def open_data():
	data = pd.read_csv('Airline-data.csv')
	data.reset_index(inplace = True)
	data.drop('index', axis=1, inplace=True)
	return data

data = open_data()


## tokenize
def tokenize(tweet):
	try:
		tweet = unicode(tweet.decode('utf-8').lower())
		tokens = tokenizer.tokenize(tweet)
		tokens = filter(lambda x: not x.startswith('@'), tokens)
		tokens = filter(lambda x: not x.startswith('#'), tokens)
		tokens = filter(lambda x: not x.startswith('http'), tokens)
		return tokens #note that you pass a string and get an array of tokens
	except:
		return 'NC'


def post_process(data):
	data['token'] = map(lambda x: tokenize(x), data['text'])
	data = data[data.token != 'NC']
	return data

data = post_process(data) #therefore we got an array(as data.tokens) of arrays(of tokens) [array of arrays]

## divide into train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(np.array(data.token), np.array(data.airline_sentiment), test_size = 0.2)

## train word2vec
import gensim
from gensim.models.word2vec import Word2Vec

w2v = Word2Vec(size=100, min_count = 10)
w2v.build_vocab(x for x in x_train)
w2v.train([x for x in x_train],total_examples=w2v.corpus_count, epochs=w2v.iter)


## generate input matrix
vector_dim = 100 #must match size in Word2Vec
def tweet_vector(tweet_tokens):
	vec = np.zeros(vector_dim).reshape(1, vector_dim)
	count = 0
	for token in tweet_tokens:
		try:
			vec += w2v[token]
			count += 1
		except KeyError:
			continue
	if count != 0:
		vec /= count
	return vec

def get_matrix(x_train):
	matrix = np.concatenate(map(lambda x: tweet_vector(x), x_train))
	return matrix

train_vec = get_matrix(x_train)
test_vec = get_matrix(x_test)

## convert y_train and y_test to 'to_categorical'
	#keras imported at top
train_output = keras.utils.to_categorical(y_train, num_classes = 3)
test_output = keras.utils.to_categorical(y_test, num_classes = 3)


## build and train model
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Activation

train_vec = train_vec.reshape(len(train_vec),1,100)
test_vec = test_vec.reshape(len(test_vec),1,100)
train_output = train_output.reshape(len(train_output),1,3)
test_output = test_output.reshape(len(test_output),1,3)

#print train_vec.shape

#model = Sequential()
#model.add(LSTM(100, batch_input_shape=(32,1,100)))
#model.add(Activation('relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(3, activation='softmax'))

main_input = Input(shape=(1,100))

lstm_out = LSTM(64)(main_input)
x = Dense(64, activation='softmax')(main_input)
x = keras.layers.multiply([lstm_out,x])

main_output = Dense(3, activation='softmax')(x)

model = Model(inputs = main_input, outputs = main_output)


model.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
model.fit(train_vec, train_output, epochs=50, batch_size=32)
score = model.evaluate(test_vec, test_output, batch_size=16)

print score[1]

#x_pred = {'text': 'the flight was really amazing and the service was good too', 'token':''}
#x_pred = post_process(x_pred)
#print x_pred
#x_pred = tweet_vector(x_pred['token'])
#print x_pred
#x_pred = x_pred.reshape(1,1,100)
#print model.predict(x_pred)
