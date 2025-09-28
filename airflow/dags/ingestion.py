# ingestion.py

import pandas as pd
from sqlalchemy import create_engine

def ingest_csv_to_mariadb(csv_path, table_name, db_url):
    """
    Load a CSV and store it in a MariaDB table.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        table_name (str): Name of the target table in the database.
        db_url (str): Full SQLAlchemy-compatible DB connection URL.
        
    Returns:
        int: Number of rows ingested
    """
    # Step 1: Load CSV
    df = pd.read_csv(csv_path)

    # Step 2: Create database engine
    engine = create_engine(db_url)

    # Step 3: Upload to MariaDB
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

    row_count = len(df)
    print(f"✅ Data ingestion to table `{table_name}` complete! Total rows: {row_count}")
    return row_count

# Optional: Script test run
if __name__ == "__main__":
    rows = ingest_csv_to_mariadb(
        csv_path="file:///home/nirbisha/Desktop/Messy_Healthcare_Diabetes.csv",
        table_name="diabetes_data",
        db_url="mysql+pymysql://root:9865abc@localhost:3306/heart"
    )
    print(f"📊 Rows ingested: {rows}")