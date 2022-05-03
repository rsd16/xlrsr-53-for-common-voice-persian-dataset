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


def main():
	for file in glob.glob('*.mp3'):
		#print(file)

		file_name = os.path.splitext(f'{file}')[0]

		if os.path.isfile(f'converted/{file_name}.wav'):
			continue

		sound = pydub.AudioSegment.from_mp3(file)
		#print(sound)

		sound.export(f'converted/{file_name}.wav', format='wav', bitrate=16000)
		
if __name__ == '__main__':
	main()
