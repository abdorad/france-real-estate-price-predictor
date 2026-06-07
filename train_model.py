import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


DVF_2024_URL = "https://www.data.gouv.fr/api/1/datasets/r/99a26050-b94f-4ffc-9eb0-73ed28a895d1"
MODEL_OUTPUT = "france_real_estate_price_model.joblib"


def download_dvf(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "dvf_2024.zip"

    if not zip_path.exists():
        print("Downloading DVF 2024 dataset...")
        urlretrieve(DVF_2024_URL, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)

    data_file = output_dir / "ValeursFoncieres-2024.txt"
    if not data_file.exists():
        raise FileNotFoundError("Could not find ValeursFoncieres-2024.txt after extraction.")

    return data_file


def prepare_data(data_file):
    columns = [
        "Date mutation",
        "Nature mutation",
        "Valeur fonciere",
        "Surface reelle bati",
        "Nombre pieces principales",
        "Type local",
        "Code postal",
        "Surface terrain",
    ]

    df = pd.read_csv(
        data_file,
        sep="|",
        usecols=columns,
        low_memory=False,
        dtype={"Code postal": "string"},
    )

    df = df[df["Nature mutation"] == "Vente"]
    df = df[df["Type local"].isin(["Maison", "Appartement"])]

    numeric_columns = [
        "Valeur fonciere",
        "Surface reelle bati",
        "Nombre pieces principales",
        "Surface terrain",
    ]

    for column in numeric_columns:
        df[column] = df[column].astype(str).str.replace(",", ".", regex=False)
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Surface terrain"] = df["Surface terrain"].fillna(0)
    df["Code postal"] = df["Code postal"].astype("string").str.zfill(5)
    df["departement"] = df["Code postal"].str[:2]

    df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce", dayfirst=True)
    df["month"] = df["Date mutation"].dt.month

    df = df.dropna(
        subset=[
            "Valeur fonciere",
            "Surface reelle bati",
            "Nombre pieces principales",
            "Type local",
            "Code postal",
            "departement",
            "month",
        ]
    )

    df = df[
        (df["Surface reelle bati"] >= 10)
        & (df["Surface reelle bati"] <= 400)
        & (df["Nombre pieces principales"] >= 1)
        & (df["Nombre pieces principales"] <= 12)
        & (df["Valeur fonciere"] >= 20_000)
        & (df["Valeur fonciere"] <= 2_500_000)
    ]

    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]
    df = df[(df["prix_m2"] >= 500) & (df["prix_m2"] <= 20_000)]

    local_quantiles = (
        df.groupby(["departement", "Type local"])["prix_m2"]
        .quantile([0.02, 0.98])
        .unstack()
        .rename(columns={0.02: "q02", 0.98: "q98"})
    )

    df = df.join(local_quantiles, on=["departement", "Type local"])
    df = df[df["prix_m2"].between(df["q02"], df["q98"])]

    df["surface_log"] = np.log1p(df["Surface reelle bati"])
    df["terrain_log"] = np.log1p(df["Surface terrain"].clip(0, 10000))
    df["rooms_per_m2"] = df["Nombre pieces principales"] / df["Surface reelle bati"]

    top_cp = set(df["Code postal"].value_counts().head(150).index)
    df["cp_model"] = df["Code postal"].where(
        df["Code postal"].isin(top_cp),
        "OTHER_" + df["departement"],
    )

    return df, top_cp


def train_model(df, top_cp):
    features_num = [
        "Surface reelle bati",
        "Nombre pieces principales",
        "surface_log",
        "terrain_log",
        "rooms_per_m2",
        "month",
    ]

    features_cat = [
        "Type local",
        "departement",
        "cp_model",
    ]

    X = df[features_num + features_cat]
    y = df["Valeur fonciere"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", features_num),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                features_cat,
            ),
        ]
    )

    categorical_features = [False] * len(features_num) + [True] * len(features_cat)

    regressor = HistGradientBoostingRegressor(
        max_iter=700,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        early_stopping=True,
        random_state=42,
        categorical_features=categorical_features,
    )

    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("preprocess", preprocess),
                ("model", regressor),
            ]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)

    metrics = {
        "rows_used": len(df),
        "mae": mean_absolute_error(y_test, predictions),
        "median_absolute_error": median_absolute_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
        "r2": r2_score(y_test, predictions),
    }

    bundle = {
        "model": model,
        "top_cp": sorted(top_cp),
        "features_num": features_num,
        "features_cat": features_cat,
        "metrics": metrics,
    }

    return bundle, metrics


def main():
    parser = argparse.ArgumentParser(description="Train the France real estate price model.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Path to ValeursFoncieres-2024.txt.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract the DVF 2024 dataset before training.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory used for downloaded DVF files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(MODEL_OUTPUT),
        help="Output path for the trained model bundle.",
    )
    args = parser.parse_args()

    if args.download:
        data_file = download_dvf(args.data_dir)
    elif args.data_file is not None:
        data_file = args.data_file
    else:
        data_file = args.data_dir / "ValeursFoncieres-2024.txt"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_file}. Use --download or pass --data-file."
        )

    print(f"Loading data from {data_file}...")
    df, top_cp = prepare_data(data_file)

    print("Training model...")
    bundle, metrics = train_model(df, top_cp)

    joblib.dump(bundle, args.output)

    print(f"Model saved to {args.output}")
    print(f"Rows used: {metrics['rows_used']:,}")
    print(f"MAE: {metrics['mae']:.2f} EUR")
    print(f"Median absolute error: {metrics['median_absolute_error']:.2f} EUR")
    print(f"RMSE: {metrics['rmse']:.2f} EUR")
    print(f"R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
