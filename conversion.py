from pydub import AudioSegment
import os

# Couldn't seem to get frontend to store file was .wav so have to convert it here
def convert(input_file_path, output_file_path):
    audio = AudioSegment.from_file(input_file_path)
    audio.export(output_file_path, format="wav")

