import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import redis
import pymysql
import pyarrow as pa
import pyarrow.parquet as pq
import io
import warnings

warnings.filterwarnings('ignore')

class DiabetesPreprocessor:
    def __init__(self):
        self.feature_columns = []

    def load_data_from_db(self, connection_string, table_name):
        engine = create_engine(connection_string)
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        print(f"✅ Loaded {len(df)} rows from database")
        return df

    def load_data_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows from CSV")
        return df

    def explore_data(self, df):
        print("\n📊 DATA EXPLORATION")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nData Types:")
        print(df.dtypes)
        print("\nOutcome Distribution:")
        print(df['Outcome'].value_counts())
        return df

    def clean_data(self, df):
        print("\n🧹 DATA CLEANING")
        print("=" * 50)
        df_clean = df.copy()

        if "Id" in df_clean.columns:
            df_clean.drop(columns=["Id"], inplace=True)
            print("✅ Dropped 'Id' column")

        zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_invalid_cols:
            df_clean[col] = df_clean[col].replace(0, np.nan)
            print(f"✅ Replaced 0s with NaN in {col}")

        for col in df_clean.columns:
            if df_clean[col].isnull().sum():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"✅ Filled missing in {col} with median: {median_val:.2f}")

        numeric_cols = df_clean.select_dtypes(include='float64').columns.drop('Outcome', errors='ignore')
        for col in numeric_cols:
            Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_clean[col] = np.clip(df_clean[col], lower, upper)
            print(f"✅ Clipped outliers in {col}")

        df_clean[numeric_cols] = df_clean[numeric_cols].round(3)
        print("✅ Rounded floats to 3 decimal places")
        return df_clean

    def prepare_final_dataset(self, df, target_col='Outcome'):
        print("\n📋 FINAL DATASET PREPARATION")
        print("=" * 50)
        print(f"✅ Final dataset shape: {df.shape}")
        print(f"✅ Features: {[col for col in df.columns if col != target_col]}")
        print(f"✅ Target: {target_col}")
        return df

    def save_to_mariadb_and_csv(self, df, connection_string, table_name="diabetes_data_cleaned"):
        print("\n💾 SAVING TO DATABASE & CSV")
        engine = create_engine(connection_string)
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)
        print(f"✅ Saved to MariaDB table: {table_name}")
        df.to_csv(f"./artifacts/{table_name}.csv", index=False)
        print(f"✅ Saved to CSV: ./artifacts/{table_name}.csv")

    def save_parquet_to_redis(self, df, redis_client, key):
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(table, buffer)
        redis_client.set(key, buffer.getvalue())
        print(f"✅ Saved Parquet dataset to Redis under key '{key}'")

    def full_pipeline(self, input_source, connection_string=None, table_name=None):
        df = self.load_data_from_db(connection_string, table_name) if connection_string and table_name \
             else self.load_data_from_csv(input_source)

        df_cleaned = self.clean_data(self.explore_data(df))
        df_final = self.prepare_final_dataset(df_cleaned)
        X = df_final.drop(columns=["Outcome"])
        y = df_final["Outcome"]
        return train_test_split(X, y, test_size=0.3, random_state=42)

def run_preprocessing_pipeline(csv_path, connection_string, table_name, redis_client):
    preprocessor = DiabetesPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.full_pipeline(
        input_source=csv_path,
        connection_string=connection_string,
        table_name=table_name
    )

    def store_parquet_df(key, X, y):
        combined = X.copy()
        combined['Outcome'] = y.values
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(combined)
        pq.write_table(table, buffer)
        redis_client.set(key, buffer.getvalue())
        print(f"✅ Stored Parquet DataFrame '{key}' in Redis")

    store_parquet_df("diabetes:train_df", X_train, y_train)
    store_parquet_df("diabetes:test_df", X_test, y_test)

if __name__ == "__main__":
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    run_preprocessing_pipeline(
        csv_path="file:///home/nirbisha/Desktop/Messy_Healthcare_Diabetes.csv",
        connection_string="mysql+pymysql://root:9865abc@localhost:3306/heart",
        table_name="diabetes_data",
        redis_client=redis_client
    )
