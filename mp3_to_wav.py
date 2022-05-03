'''
Written by Alireza Rashidi.
Python 3.8.x.
This file converts *.mp3 files in common voice dataset to *.wav format.
The reasong for this is that torchaudio didn't work with *.mp3 files.
Just put this file inside this directory among *.mp3 files, create a folder named "converted", and run the file.
The bitrate must 16 KHz.
'''

import glob
import os
import pydub


for file in glob.glob('*.mp3'):
	#print(file)
	sound = pydub.AudioSegment.from_mp3(file)
	#print(sound)
	new_file = os.path.splitext(f'{file}')[0]
	#print(new_file)
	sound.export(f'converted/{new_file}.wav', format='wav', bitrate=16000)
