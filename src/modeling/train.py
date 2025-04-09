from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model(X_train, y_train, model_path="models/logistic_model.pkl"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model
