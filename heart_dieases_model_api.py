from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load the trained model
heart_disease_model = pickle.load(open('/Users/macbook/Downloads/heart_disease_model.sav', 'rb'))

@app.post('/heart_disease_pred')
def heart_disease_pred(input_params: ModelInput):
    input_data = input_params.json()
    input_dictionary = json.loads(input_data)

    # Extract features from the input data
    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cp = input_dictionary['cp']
    trestbps = input_dictionary['trestbps']
    chol = input_dictionary['chol']
    fbs = input_dictionary['fbs']
    restecg = input_dictionary['restecg']
    thalach = input_dictionary['thalach']
    exang = input_dictionary['exang']
    oldpeak = input_dictionary['oldpeak']
    slope = input_dictionary['slope']
    ca = input_dictionary['ca']
    thal = input_dictionary['thal']

    # Create a list of input features
    input_list = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    # Make the prediction
    prediction = heart_disease_model.predict(input_list)

    # Return the prediction result
    if prediction[0] == 0:
        return 'The Person has heart disease'
    else:
        return 'The Person has no heart disease'
