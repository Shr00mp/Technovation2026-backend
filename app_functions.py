from fastapi import FastAPI, File, UploadFile
import shutil
import os
from conversion import convert
from extract_features import get_all_features
from rf_model_imlpementation import get_analysis, train_model
import os

app = FastAPI()

MODEL_STATE = {
    "model": None,
    "scaler": None,
    "features": None
}

# Originally, the model was trained every time a call was made to backend
# Model training takes a couple of seconds so it was erroring out
# On startup the model is trained once and stored in the MODEl_STATE
@app.on_event("startup")
async def startup_event():
    print("Currently training model") # Debugging 
    model, scaler, features, acc = train_model()
    MODEL_STATE["model"] = model
    MODEL_STATE["scaler"] = scaler
    MODEL_STATE["features"] = features
    print(f"Model trained with {acc*100:.1f}% accuracy.")



@app.post("/upload-audio/")
async def save_audio(file: UploadFile = File(...)):
    in_file_path = os.path.join("uploaded_mp3s", file.filename)
    out_file_path = os.path.join("converted_wavs", file.filename)

    with open(in_file_path, "wb") as local_file:
        shutil.copyfileobj(file.file, local_file)
    
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

    analysis = get_analysis(
        feature_dict, 
        MODEL_STATE["model"], 
        MODEL_STATE["scaler"], 
        MODEL_STATE["features"]
    )
    analysis["model_accuracy"] = MODEL_STATE.get("accuracy", 0)

    print(analysis)

    # At this point we have the analysis stored, so we don't need user's audio files anymore
    # Deleting off of backend ensures uer privacy 
    os.remove(in_file_path)
    os.remove(out_file_path)
    
    return analysis


# Created this one so we could test whether or not the HTTP connection was actually working 
@app.get("/hello/")
def read_root():
    return {"Hello": "World"}