import music21 as m21 
import glob


# converts midi-files into lists of strings. the list includes notes, chords and rests. 
# Right after each note, chord or rest element its corresponding duration is appended as frac of quarter note
def formate_data(): 
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

def create_labeled_data(processed_data, class_to_ind, seq_len, n_batches):
	ind_data = []
	ind_data.append(class_to_ind[word] for word in processed_data)
	input_seqs = [] # Input sequences to the network
	seq_labels = [] # labels to the input sequences, that is the sequence with offset 1

	for i in range(n_batches): 
		seq = processed_data[i*seq_len:(i*seq_len+seq_len-1)]
		label = processed_data[(i*seq_len+1):(i*seq_len+seq_len)]
		input_seqs.append(seq)
		seq_labels.append(label)


