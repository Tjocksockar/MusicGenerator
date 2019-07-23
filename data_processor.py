import music21 as m21 
import glob
import numpy as np 

from tensorflow.python.keras.utils import to_categorical


# converts midi-files into lists of strings. the list includes notes, chords and rests. 
# Right after each note, chord or rest element its corresponding duration is appended as frac of quarter note
def extract_data(): 
	for filename in glob.glob("midi_data/*.mid"):
		midi = m21.converter.parse(filename) # creating a midi object
		notes_and_chords = midi.flat.notesAndRests # extracts notes, chords and rests only
		#print(notes_and_chords.show('text'))
		
		# Convert the midi stream to strings in a list. 
		#Notes, chords and rests with correspondint durations. 
		str_list = [] 
		for element in notes_and_chords: 
			if isinstance(element, m21.note.Note): 
				str_list.append(str(element.pitch))
				str_list.append(str(element.duration.quarterLength))
				#print(str_list[-1])
			elif isinstance(element, m21.chord.Chord):
				str_list.append('.'.join(str(n) for n in element.pitches))
				str_list.append(str(element.duration.quarterLength))
				#print(str_list[-1])
			elif isinstance(element,m21.note.Rest): 
				str_list.append('rest')
				str_list.append(element.duration.quarterLength)
				#print(str_list[-1])
	return str_list

# Create mappings from a word to an index and back
def create_dictionaries(processed_data):
	classes = set(processed_data)
	n_classes = len(classes)
	print(n_classes)
	class_to_ind = dict((word, ind) for ind, word in enumerate(classes))
	ind_to_class = dict((ind, word) for ind, word in enumerate(classes))
	return class_to_ind, ind_to_class

# Encodes words to one hot numpy representation compatible with Keras LSTM 
def formate_data(processed_data, class_to_ind, seq_len, n_classes):
	numerical_data = []
	for key in processed_data: 
		numerical_data.append(class_to_ind[key])
	input_data = []
	labels = []

	for i in range(len(numerical_data) - seq_len): 
		seq = numerical_data[i:i + seq_len]
		label = numerical_data[i+seq_len]
		input_data.append(seq)
		labels.append(label)

	n_seqs = len(input_data)
	input_data = np.reshape(input_data, (n_seqs, seq_len, 1))
	input_data = input_data/float(n_classes)

	labels = to_categorical(labels)
	return input_data, labels
