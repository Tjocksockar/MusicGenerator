import music21 as m21 
import glob
import numpy as np 

from tensorflow.python.keras.utils import to_categorical


# converts midi-files into lists of strings. the list includes notes, chords and rests. 
# Right after each note, chord or rest element its corresponding duration is appended as frac of quarter note
def extract_data(): 
	str_list = []
	for filename in glob.glob("midi_data_in_use/*.mid"):
		print('='*70)
		print(filename)
		midi = m21.converter.parse(filename) # creating a midi object
		notes_and_chords = midi.flat.notesAndRests # extracts notes, chords and rests only
		#print(notes_and_chords.show('text'))
		
		# Convert the midi stream to strings in a list. 
		#Notes, chords and rests with correspondint durations. 
		for element in notes_and_chords: 
			if isinstance(element, m21.note.Note): 
				str_list.append(str(element.pitch))
				str_list.append(str(element.duration.quarterLength)+'*')
				#print(str_list[-1])
			elif isinstance(element, m21.chord.Chord):
				#str_list.append('.'.join(str(n) for n in element.pitches))
				for note in element.pitches: 
					str_list.append(str(note))
				str_list.append(str(element.duration.quarterLength)+'*')
				#print(str_list[-1])
			elif isinstance(element,m21.note.Rest): 
				str_list.append('rest')
				str_list.append(str(element.duration.quarterLength)+'*')
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
	input_data = input_data/n_classes # normalizing data

	labels = to_categorical(labels)
	return input_data, labels

# Given a by the model predicted list, this creates the midi file
def create_midi(predicted_list): 
	time_elapsed = 0 # absolute time elapsed in the generated tune
	m21_predictions = [] # will contain music21 notes, chords and rests
	this_chord = []
	for i, element in enumerate(predicted_list): 
		if '*' in element: # is a duration
			duration = element.split('*')[0]
			if '/' in duration: 
				numbers = duration.split('/')
				duration = float(numbers[0])/float(numbers[1])
			else: 
				duration = float(duration)
			if len(this_chord) == 1: # it is a note or rest
				if this_chord[0] == 'rest': 
					rest = m21.note.Rest()
					rest.offset = time_elapsed
					rest.duration = duration 
					m21_predictions.append(rest)
				else: 
					note = m21.note.Note(chord[0])
					note.offset = time_elapsed
					note.storedInstrument = instrument.Piano()
					note.duration = duration
					m21_predictions.append(note)
			elif len(this_chord) > 1: # it is a chord
				these_notes = []
				for this_note in this_chord: 
					if this_note == 'rest': 
						pass
					else: 
						note = m21.note.Note(this_note)
						note.storedInstrument = instrument.Piano()
						these_notes.append(note)
				if len(these_notes) > 0:
					chord = m21.chord.Chord(these_notes)
					chord.offset = time_elapsed
					chord.duration = duration
					m21_predictions.append(chord)
			time_elapsed += duration
			this_chord = []
		else: 
			this_chord.append(element)
	midi_stream = m21.stream.Stream(m21_predictions)
	midi_stream.write('midi', fp='generated_music.mid')

