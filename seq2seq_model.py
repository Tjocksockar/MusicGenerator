from tensorflow.python.keras.models import Model 
from tensorflow.python.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.python.keras.optimizers import Adam

def create_seq2seq_model(input_data): 
	encoder_input = Input(shape=(None, input_data.shape[2]))