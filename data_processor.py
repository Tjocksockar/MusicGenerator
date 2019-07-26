import music21 as m21 
import glob
import numpy as np 

from tensorflow.python.keras.utils import to_categorical


# converts midi-files into lists of strings. the list includes notes, chords and rests. 
# Right after each note, chord or rest element its corresponding duration is appended as frac of quarter note
def extract_data(): 
	str_list = []
	for filename in glob.glob("midi_data/*.mid"):
		print('='*70)
		print(filename)
		midi = m21.converter.parse(filename) # creating a midi object
		notes_and_chords = midi.flat.notes # extracts notes, chords and rests only
		#print(notes_and_chords.show('text'))
		
		# Convert the midi stream to strings in a list. 
		#Notes, chords and rests with correspondint durations. 
		for i, element in enumerate(notes_and_chords[0:len(notes_and_chords)-1]): # loop through all but last element
			if isinstance(element, m21.note.Note): 
				str_list.append(str(element.pitch))
				#str_list.append(str(element.duration.quarterLength)+'*')
				str_list.append(str(notes_and_chords[i+1].offset-element.offset)+'=')
				#print(str_list[-1])
			elif isinstance(element, m21.chord.Chord):
				#str_list.append('.'.join(str(n) for n in element.pitches))
				for note in element.pitches: 
					str_list.append(str(note))
				#str_list.append(str(element.duration.quarterLength)+'*')
				str_list.append(str(notes_and_chords[i+1].offset-element.offset)+'=')
				#print(str_list[-1])
		print('list length = ', len(str_list))
	return str_list

# Create mappings from a word to an index and back
def create_dictionaries(processed_data):
	classes = set(processed_data)
	n_classes = len(classes)
	class_to_ind = dict((word, ind) for ind, word in enumerate(classes))
	ind_to_class = dict((ind, word) for ind, word in enumerate(classes))
	return class_to_ind, ind_to_class

# Encodes words to one hot numpy representation compatible with Keras LSTM 
def format_data(processed_data, class_to_ind, seq_len, n_classes):
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
	input_data = input_data/n_classes # normalizing data

	labels = to_categorical(labels)
	return input_data, labels

# Given a by the model predicted list, this creates the midi file
def create_midi(predicted_list): 
	time_elapsed = 0 # absolute time elapsed in the generated tune
	m21_predictions = [] # will contain music21 notes, chords and rests
	this_chord = []
	for element in predicted_list: 
		if '=' in element: # is offset
			this_offset = element.split('=')[0]
			if '/' in this_offset: 
				numbers = this_offset.split('/')
				this_offset = float(numbers[0])/float(numbers[1])
			else: 
				this_offset = float(this_offset)
			duration = this_offset
			if len(this_chord) == 1: # it is a note or rest
				print(this_chord)
				note = m21.note.Note(this_chord[0])
				note.offset = time_elapsed
				note.duration.quarterLength = duration
				note.storedInstrument = m21.instrument.Piano()
				m21_predictions.append(note)
				time_elapsed += 0.5
			elif len(this_chord) > 1: # it is a chord
				print(this_chord)
				these_notes = []
				for this_note in this_chord: 
					note = m21.note.Note(this_note)
					note.storedInstrument = m21.instrument.Piano()
					these_notes.append(note)
				chord = m21.chord.Chord(these_notes)
				chord.offset = time_elapsed
				chord.duration.quarterLength = duration
				m21_predictions.append(chord)
				time_elapsed += 0.5
			this_chord = []
		else: 
			this_chord.append(element)
	midi_stream = m21.stream.Stream(m21_predictions)
	midi_stream.write('midi', fp='generated_music.mid')

