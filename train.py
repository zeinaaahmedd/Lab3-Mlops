import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

def train_model():
    # 1. Set the Experiment Name [cite: 93]
    mlflow.set_experiment("Melbourne Housing Prediction")

    # 2. Load the training data
    train_df = pd.read_csv('data/train.csv')
    X_train = train_df.drop('Price', axis=1)
    y_train = train_df['Price']

    # 3. Start an MLflow Run [cite: 98]
    with mlflow.start_run(run_name="Random Forest Baseline"):
        # Define parameters [cite: 98]
        params = {
            "n_estimators": 100,
            "random_state": 42
        }
        mlflow.log_params(params) # [cite: 98]

        # 4. Train the Model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 5. Log Metrics [cite: 105, 106]
        # We use R2 and MSE for regression
        predictions = model.predict(X_train)
        r2 = r2_score(y_train, predictions)
        mse = mean_squared_error(y_train, predictions)
        
        mlflow.log_metric("R2_score", r2)
        mlflow.log_metric("MSE", mse)

        # 6. Save and Log Artifacts [cite: 108, 109]
        model_path = "model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Baseline model trained. R2: {r2:.4f}")

if __name__ == "__main__":
    train_model()