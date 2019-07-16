import tensorflow as tf 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout


def create_model(): 
	model = Sequential()
