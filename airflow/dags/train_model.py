import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import redis
from io import StringIO
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.xgboost

def load_combined_df(key, host="localhost", port=6379, db=0):
    r = redis.Redis(host=host, port=port, db=db)
    csv_data = r.get(key)
    if not csv_data:
        raise ValueError(f"❌ Key '{key}' not found in Redis.")
    df = pd.read_csv(StringIO(csv_data.decode("utf-8")))
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    print(f"✅ Loaded {len(df)} rows from Redis key '{key}'")
    return X, y

def train_model():
    X, y = load_combined_df("diabetes:train_df")

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Diabetes_XGBoost_KFold")

    with mlflow.start_run(run_name="xgboost_kfold_training") as run:
        try:
            clean_params = {k: v for k, v in model.get_params().items() if v is not None}
            mlflow.log_params(clean_params)

            print("\n🔍 K-FOLD CROSS VALIDATION RESULTS")

            for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred)
                rec = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_prob)

                accuracies.append(acc)
                precisions.append(prec)
                recalls.append(rec)
                f1s.append(f1)
                aucs.append(auc)

                mlflow.log_metric(f"fold{fold}_accuracy", acc)
                mlflow.log_metric(f"fold{fold}_precision", prec)
                mlflow.log_metric(f"fold{fold}_recall", rec)
                mlflow.log_metric(f"fold{fold}_f1", f1)
                mlflow.log_metric(f"fold{fold}_auc", auc)

                print(f"\n📂 Fold {fold}")
                print(f"Accuracy:  {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall:    {rec:.4f}")
                print(f"F1 Score:  {f1:.4f}")
                print(f"AUC:       {auc:.4f}")

            avg_acc = sum(accuracies) / 5
            avg_prec = sum(precisions) / 5
            avg_rec = sum(recalls) / 5
            avg_f1 = sum(f1s) / 5
            avg_auc = sum(aucs) / 5

            mlflow.log_metric("avg_accuracy", avg_acc)
            mlflow.log_metric("avg_precision", avg_prec)
            mlflow.log_metric("avg_recall", avg_rec)
            mlflow.log_metric("avg_f1", avg_f1)
            mlflow.log_metric("avg_auc", avg_auc)

            print("\n📊 AVERAGE METRICS ACROSS 5 FOLDS")
            print(f"Avg Accuracy:  {avg_acc:.4f}")
            print(f"Avg Precision: {avg_prec:.4f}")
            print(f"Avg Recall:    {avg_rec:.4f}")
            print(f"Avg F1 Score:  {avg_f1:.4f}")
            print(f"Avg AUC:       {avg_auc:.4f}")

            model.fit(X, y)
            mlflow.xgboost.log_model(model, artifact_path="final_xgboost_model")
            print("\n✅ Final model trained and logged to MLflow.")

            # ✅ Save run_id to Redis
            run_id = run.info.run_id
            redis_client = redis.Redis(host="localhost", port=6379, db=0)
            redis_client.set("diabetes:model_run_id", run_id)
            print(f"✅ run_id '{run_id}' saved to Redis.")

        except Exception as e:
            print(f"❌ MLflow run failed: {e}")
        finally:
            mlflow.end_run()

# Run standalone
if __name__ == "__main__":
    train_model()