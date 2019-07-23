from tensorflow.python.keras.callbacks import ModelCheckpoint

from data_processor import *
from model import *


# Data preparation
processed_data = extract_data()
class_to_ind, ind_to_class = create_dictionaries(processed_data)

# Parameters
n_classes = len(class_to_ind)
n_data = len(processed_data)
seq_len = 100
n_batches = 128
n_epochs = 8
print('='*70)
print(n_classes)
eta = 1e-4 # initial learning rate
eta_decay = 1e-6

# Creating input sequences and labels for the network
input_data, labels = formate_data(processed_data, class_to_ind, seq_len, n_classes)

# Creating model architecture 
model = create_model(input_data, n_classes, eta, eta_decay)

# Training the network 
# and saving weights frequently during training
filename = "model_at_{epoch:02d}_{loss:.4f}_bigger.hdf5" 

checkpoint = ModelCheckpoint(
	filename, 
	monitor = 'loss', 
	verbose = 0, 
	save_best_only = True, 
	mode = 'min'
)
callbacks = [checkpoint]

model.fit(
	input_data, 
	labels, 
	epochs = n_epochs, 
	batch_size = n_batches, 
	callbacks = callbacks
)
