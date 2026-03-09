import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def preprocess_data():
    raw_path = 'data/melbourne_housing.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    
    # DEBUG: Let's see what columns are actually there
    print("Columns found in your CSV:", df.columns.tolist())

    # Target column is usually 'Price'
    if 'Price' not in df.columns:
        print("Error: 'Price' column not found. Please check the column names printed above.")
        sys.exit(1)

    # 1. Drop rows with missing Price [cite: 75]
    df = df.dropna(subset=['Price'])

    # 2. Select ONLY numeric columns that exist in your specific file
    # This avoids the KeyError you just saw.
    numeric_df = df.select_dtypes(include=['number'])
    
    # 3. Fill missing values with median [cite: 74]
    numeric_df = numeric_df.fillna(numeric_df.median())

    # 4. Split into Train and Test [cite: 85, 96]
    X = numeric_df.drop('Price', axis=1)
    y = numeric_df['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Save processed files [cite: 82]
    os.makedirs('data', exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv('data/train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('data/test.csv', index=False)
    
    print("Successfully created data/train.csv and data/test.csv")

if __name__ == "__main__":
    preprocess_data()