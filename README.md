### Summary

The aim of this project is to create a music generator by training a simple LSTM model. 
The training set contains midi files of piano music by Mozart. 
The midi files are translated to text strings on the note level. 
More specifically, each note is typed out in sequential order followed by its offset compared to the previous note. If more than one note is played at the same time, all notes are typed out before the offset is typed out. An example of a string is A3 0.5 C3 E3 G3 C2 1.0 A3 0.5. The reason for choosing the note level is to limit the number of classes. However, others have achieved great results operating on the chord level as well. 

The music generator created so far is not very good but clearly performs better than random. 
One problem the generator is facing is that is gets stuck on one note. Another problem is that it generates more offsets than notes, which is adjusted for in the convertion from string back to midi. 
However, this is an on-going project so improvements might be yet to come. 
The next step will be to implement an encoder decoder architecture. 

### Instructions for training

1. create a folder in the root of the project called 'midi_data_in_use' and place the midi files you wish to use for training in it.
2. In main.py, make sure that the section === Training the network === is not commented out. 
3. Run main.py 
4. For every epoch, the weights will be saved in the folder saved_weights

### Instructions for testing

1. In main.py comment out the section === Training the network === and make sure that the section === Testing the network === is not commented out. 
2. Change the first line in the section === Testing the network === so that the variable contains the right file name. The file name is loss dependent so it changes every time you wish to test a new trained network. 
3. Run main.py
4. To listen to the generated music, run generated_music.midi