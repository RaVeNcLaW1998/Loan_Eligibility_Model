import pickle

def load_model(model_path="models/logistic_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def make_prediction(model, input_df):
    return model.predict(input_df)[0]
