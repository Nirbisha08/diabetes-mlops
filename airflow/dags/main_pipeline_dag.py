# main_pipeline_dag.py

from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Import your actual pipeline step functions
from ingestion import ingest_csv_to_mariadb
from preprocessing import run_preprocessing_pipeline
from validation import validate_raw_diabetes_dataset
from post_validation import validate_cleaned_diabetes_dataset
from train_model import train_model
default_args = {
    "owner": "Nirbisha Shrestha",
    "depends_on_past": False,
    "email": ["nirbisha.shrestha@mail.bcu.ac.uk"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="final_project_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline for diabetes dataset using Airflow",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10),
    catchup=False,
    tags=["ml", "pipeline", "diabetes"]
) as dag:

    start_mcs_container = BashOperator(
        task_id='start_mcs_container',
        bash_command='docker start mcs_container',
    )

    start_redis_store = BashOperator(
        task_id='start_redis_store',
        bash_command='docker start redis_store',
    )

    start_mlflow_ui = BashOperator(
        task_id='start_mlflow_ui',
        bash_command="""
        if ! lsof -i:5000; then
            nohup conda run -n heart_data mlflow ui --host 0.0.0.0 --port 5000 > /tmp/mlflow_ui.log 2>&1 &
            echo "MLflow UI started"
        else
            echo "MLflow UI already running"
        fi
        """,
    )

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_csv_to_mariadb,
        op_kwargs={
            "csv_path": "/home/nirbisha/Desktop/Messy_Healthcare_Diabetes.csv",
            "table_name": "diabetes_data",
            "db_url": "mysql+pymysql://root:9865abc@localhost:3306/heart"
        }
    )

    validate_before = PythonOperator(
        task_id="initial_data_validation",
        python_callable=validate_raw_diabetes_dataset,
        op_kwargs={
            "csv_path": "./artifacts/Messy_Healthcare_Diabetes.csv"
        }
    )

    preprocessing = PythonOperator(
        task_id="data_preprocessing",
        python_callable=run_preprocessing_pipeline
    )

    validate_after = PythonOperator(
        task_id="post_data_validation",
        python_callable=validate_cleaned_diabetes_dataset,
        op_kwargs={
            "csv_path": "./artifacts/diabetes_data_cleaned.csv"
        }
    )

    model_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    # Define task dependencies
    [start_mcs_container, start_redis_store, start_mlflow_ui] >> data_ingestion
    data_ingestion >> validate_before >> preprocessing >> validate_after >> model_train