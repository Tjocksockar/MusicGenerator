import music21 as m21 
from collections import Counter

def formate_data(): 
	filename = 'midi_data/testdata.mid'
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
	
	# Create dictionaries that encodes text to int and back
	seq_len = 100 # Test different values to improve results 
	classes = set(element for element in str_list)
	n_classes = len(classes)
	class_to_ind = dict((word, ind) for ind, word in enumerate(classes))
	
	

formate_data()