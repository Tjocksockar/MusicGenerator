import tensorflow as tf 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Activation


def create_model(input_data, n_classes): 
	model = Sequential()
	model.add(LSTM(
		256, 
		input_shape = (input_data.shape[1], input_data.shape[2]),
		return_sequences = True, 
		activation = 'relu'
	))
	model.add(Dropout(0.25))
	model.add(LSTM(
		512, 
		return_sequences = True, 
		activation = 'relu'
	))
	model.add(Dropout(0.25))
	model.add(LSTM(
		512, 
		activation = 'relu'
	))
	model.add(Dense(512))
	model.add(Dropout(0.25))
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model