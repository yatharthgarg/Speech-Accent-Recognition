import pandas as pd
import urllib.request
import os
import sys
from pydub import AudioSegment

class GetAudio:

    def __init__(self, csv_filepath, destination_folder= 'audio/', wait= 1.5, debug=False ):
        '''
        Initializes GetAudio class object
        :param destination_folder (str): Folder where audio files will be saved
        :param wait (float): Length (in seconds) between web requests
        :param debug (bool): Outputs status indicators to console when True
        '''
        self.csv_filepath = csv_filepath
        self.audio_df = pd.read_csv(csv_filepath)
        self.url = 'http://chnm.gmu.edu/accent/soundtracks/{}.mp3'
        self.destination_folder = destination_folder
        self.wait = wait
        self.debug = False

    def check_path(self):
        '''
        Checks if self.distination_folder exists. If not, a folder called self.destination_folder is created
        '''
        if not os.path.exists(self.destination_folder):
            if self.debug:
                print('{} does not exist, creating'.format(self.destination_folder))
            os.makedirs('../' + self.destination_folder)

    def get_audio(self):
        '''
        Retrieves all audio files from 'language_num' column of self.audio_df
        If audio file already exists, move on to the next
        :return (int): Number of audio files downloaded
        '''

        self.check_path()

        counter = 0

        for lang_num in self.audio_df['language_num']:
            if not os.path.exists(self.destination_folder +'{}.wav'.format(lang_num)):
                if self.debug:
                    print('downloading {}'.format(lang_num))
                (filename, headers) = urllib.request.urlretrieve(self.url.format(lang_num))
                sound = AudioSegment.from_mp3(filename)
                sound.export('../' + self.destination_folder + "{}.wav".format(lang_num), format="wav")
                counter += 1

        return counter

if __name__ == '__main__':
    '''
    Example console command
    python GetAudio.py audio_metadata.csv
    '''
    csv_file = sys.argv[1]
    ga = GetAudio(csv_filepath=csv_file)
    ga.get_audio()

