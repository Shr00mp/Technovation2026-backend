import pandas as pd
from feature_extraction import get_formants
from feature_extraction import get_MFCCs
from feature_extraction import get_all_features
import os

# Again, this only needs to be run once 
data = pd.read_csv("audio-details/audio_files.csv")

# List for storing feature dictionaries (for the separate audio files)
features_list = []

for idx, row in data.iterrows():
    audio_ID = row["Sample ID"]
    if (audio_ID.endswith("augmented1") or audio_ID.endswith("augmented2")):
        audio_path = "audio_files/augmented-files/" + audio_ID + ".wav"
    else:
        audio_path = "audio_files/original-files/" + audio_ID + ".wav"
    feature_dict = get_all_features(audio_path, 50, 500, "Hertz")
    feature_dict["Sample ID"] = audio_ID
    feature_dict["Label"] = row["Label"]
    features_list.append(feature_dict)

features_df = pd.DataFrame(features_list)
features_df.to_csv("audio-details/audio_features_new.csv", index=False)
