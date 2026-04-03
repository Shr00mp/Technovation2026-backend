from fastapi import FastAPI, File, UploadFile
import shutil
import os
from conversion import convert
from extract_features import get_all_features
from rf_model_imlpementation import get_analysis, train_model

app = FastAPI()

@app.post("/upload-audio/")
async def save_audio(file: UploadFile = File(...)):
    in_file_path = os.path.join("uploaded_mp3s", file.filename)
    out_file_path = os.path.join("converted_wavs", file.filename)
    # Open a local file and write the uploaded content into it
    with open(in_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    convert(
        input_file_path=in_file_path,
        output_file_path=out_file_path
    )

    feature_dict = get_all_features(
        path_to_audio_file=out_file_path,
        f0min=50,
        f0max=5000,
        unit="Hertz"
    ) 

    final_model, model_scaler, selected_features, accuracy = train_model()

    analysis = get_analysis(feature_dict, final_model, model_scaler, selected_features)
    analysis["accuracy"] = accuracy
    
    return analysis



@app.get("/hello/")
def read_root():
    return {"Hello": "World"}