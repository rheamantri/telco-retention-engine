import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def build_churn_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if "customerID" in cat_cols:
        cat_cols.remove("customerID")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipe

def save_pipeline(pipe: Pipeline, path: str) -> None:
    joblib.dump(pipe, path)

def load_pipeline(path: str) -> Pipeline:
    return joblib.load(path)
