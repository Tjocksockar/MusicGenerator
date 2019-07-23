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
print('The shape of the input data numpy array is ', input_data.shape)

print('='*70)
# Creating model architecture 
model = create_model(input_data, n_classes, eta, eta_decay)
print('='*70)

"""
### ========== Training the network ======== ###

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
"""

### ========== Testing the network ======== ###

# Create the model with saved weights loaded 
weights_path = 'saved_weights/model_at_05_3.7365_bigger.hdf5' # check before running
trained_model = create_model(input_data, n_classes, eta, eta_decay)
trained_model.load_weights(weights_path)

# I use the first 100 datapoints as the start input for music generation
pred_input = input_data[0, 0:seq_len, 0]
pred_input = np.reshape(pred_input, (seq_len))

n_predictions = 4000 # number of notes/rests/durations to generate
predictions = [] # holding the generated music before turned to midi formate
for model_prediction in range(n_predictions): 
	shaped_input = np.reshape(pred_input, (1, seq_len, 1))
	prediction = trained_model.predict(shaped_input, verbose=0)
	note_ind = np.argmax(prediction)
	note = ind_to_class[note_ind]
	predictions.append(note)

	pred_input = np.append(pred_input, [note_ind/n_classes], axis = 0)
	pred_input = pred_input[1:seq_len+1]
