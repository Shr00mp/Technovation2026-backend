# Uplift for Technovation 2026

## How to run the backend
### Step 1
Install required libraries, including skcikit-learn, pandas, numpy, etc. 
### Step 2
Run the following command in the local terminal: 
```commandline
uvicorn app_functions:app --reload
```
### Step 3
Run the following command in the device terminal: 
```commandline
ngrok http 8000
```
Press CTRL+C to stop the server.
