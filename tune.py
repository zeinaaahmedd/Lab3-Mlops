import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import os

def tune_model():
    # Set the experiment name
    mlflow.set_experiment("Melbourne Housing Prediction")

    # Load preprocessed data
    if not os.path.exists('data/train.csv'):
        print("Error: data/train.csv not found.")
        return
        
    train_df = pd.read_csv('data/train.csv')
    X_train = train_df.drop('Price', axis=1)
    y_train = train_df['Price']

    # 1. Start a Parent Run to group the tuning results [cite: 22, 23]
    with mlflow.start_run(run_name="Random Forest Hyperparameter Tuning"):
        
        # 2. Define hyperparameters to test [cite: 290]
        n_estimators_options = [50, 150]
        max_depth_options = [5, 10]

        for n_estimators in n_estimators_options:
            for max_depth in max_depth_options:
                
                # 3. Start a Nested (Child) Run [cite: 290]
                # 'nested=True' allows these to show up under the parent run in the UI
                with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}", nested=True):
                    
                    # Log parameters [cite: 29, 39]
                    params = {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "random_state": 42
                    }
                    mlflow.log_params(params)

                    # Train and Evaluate
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    
                    predictions = model.predict(X_train)
                    r2 = r2_score(y_train, predictions)
                    mse = mean_squared_error(y_train, predictions)

                    # Log metrics [cite: 29, 39]
                    mlflow.log_metric("R2_score", r2)
                    mlflow.log_metric("MSE", mse)
                    
                    print(f"Finished Child Run: n_estimators={n_estimators}, max_depth={max_depth} | R2: {r2:.4f}")

if __name__ == "__main__":
    tune_model()