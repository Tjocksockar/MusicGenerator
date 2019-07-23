import tensorflow as tf 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Activation
from tensorflow.python.keras.optimizers import Adam

# Creates the model. In this func the model architecture is set
def create_model(input_data, n_classes, eta, eta_decay): 
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
	model.add(Dropout(0.25))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(n_classes, activation='softmax'))
	opt = Adam(lr=eta, decay=eta_decay) # using decaying learning rate
	model.compile(loss='categorical_crossentropy', optimizer=opt)
	return model