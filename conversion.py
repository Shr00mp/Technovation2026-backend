from pydub import AudioSegment
import os

def convert(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")

convert("uploaded_mp3s/audio_recording_1775083046845.mp3", "converted_wavs/new_audio.wav")

