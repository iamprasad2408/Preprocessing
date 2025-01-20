import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_stock_data(stock_market_transformed):
    
    try:
        data = pd.read_csv("stock_market_transformed.csv")
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    print("\nChecking for missing values:")
    print(data.isnull().sum())
    data.dropna(inplace=True)  
    print("\nMissing values handled.")

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data.sort_values(by='Date', inplace=True)
        print("\nDate column converted to datetime and sorted.")

    scaler = MinMaxScaler()
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if len(numerical_columns) > 0:
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        print("\nNumerical columns scaled.")

    categorical_columns = data.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            if col != 'Date': 
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                print(f"Categorical column '{col}' encoded.")

    print("\nProcessed Dataset Info:")
    print(data.info())

    return data


if __name__ == "__main__":
    
    dataset_path = "stock_market_transformed.csv"
    processed_data = preprocess_stock_data(dataset_path)

    if processed_data is not None:
        
        processed_data.to_csv("preprocessed_stock_data.csv", index=False)
        print("\nPreprocessed data saved to 'preprocessed_stock_data.csv'.")
