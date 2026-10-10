# NAME: Zach Rangel

import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train_df = pl.read_csv("training.csv.gz")
test_df = pl.read_csv("testing.csv.gz")
submission = pl.read_csv("submission.csv", separator=",", null_values=["NA"])

train_df = train_df.with_columns([
    pl.col("distcont4").cast(pl.Float64, strict=False).fill_null(50.0),
    pl.col("distcont3").cast(pl.Float64, strict=False).fill_null(50.0),
    pl.col("distcont2").cast(pl.Float64, strict=False).fill_null(50.0),
    pl.col("shotclock").cast(pl.Float64, strict=False).fill_null(24.0),
    pl.col("shooterspeed").cast(pl.Float64, strict=False).fill_null(0.0),
    pl.col("num_contesters").cast(pl.Float64, strict=False).fill_null(0.0),
    pl.col("closestdefapproach").cast(pl.Float64, strict=False).fill_null(6.0),
    
    pl.when(pl.col("contested").cast(pl.Float64, strict=False).is_null())
      .then(pl.when(pl.col("distcont1").cast(pl.Float64, strict=False) > 6.0).then(0.0).otherwise(1.0))
      .otherwise(pl.col("contested").cast(pl.Float64, strict=False)).alias("contested")
])

test_df = test_df.with_columns([
    pl.col("distcont4").cast(pl.Float64, strict=False).fill_null(50.0),
    pl.col("distcont3").cast(pl.Float64, strict=False).fill_null(50.0),
    pl.col("distcont2").cast(pl.Float64, strict=False).fill_null(50.0),
    pl.col("shotclock").cast(pl.Float64, strict=False).fill_null(24.0),
    pl.col("shooterspeed").cast(pl.Float64, strict=False).fill_null(0.0),
    pl.col("num_contesters").cast(pl.Float64, strict=False).fill_null(0.0),
    pl.col("closestdefapproach").cast(pl.Float64, strict=False).fill_null(6.0),
    
    pl.when(pl.col("contested").cast(pl.Float64, strict=False).is_null())
      .then(pl.when(pl.col("distcont1").cast(pl.Float64, strict=False) > 6.0).then(0.0).otherwise(1.0))
      .otherwise(pl.col("contested").cast(pl.Float64, strict=False)).alias("contested")
])

shottype_map = {"dunk": 4.0, "layup": 3.0, "floater": 2.0, "jumper": 1.0, "unknown": 0.0}
gamestate_map = {"TRANSITION": 2.0, "HALFCOURT": 1.0, "unknown": 0.0}

train_df = train_df.with_columns([
    pl.col("shottype").fill_null("unknown").replace_strict(shottype_map, default=0.0, return_dtype=pl.Float64),
    pl.col("gamestate").fill_null("unknown").replace_strict(gamestate_map, default=0.0, return_dtype=pl.Float64)
])

test_df = test_df.with_columns([
    pl.col("shottype").fill_null("unknown").replace_strict(shottype_map, default=0.0, return_dtype=pl.Float64),
    pl.col("gamestate").fill_null("unknown").replace_strict(gamestate_map, default=0.0, return_dtype=pl.Float64)
])

train_df = train_df.with_columns([
    ((pl.col("shotclock") <= 4.0)).cast(pl.Float64).alias("rushed_shot")
])

test_df = test_df.with_columns([
    ((pl.col("shotclock") <= 4.0)).cast(pl.Float64).alias("rushed_shot")
])

train_df = train_df.with_columns([
    ((pl.col("locationx")**2 + pl.col("locationy")**2).sqrt()).alias("calculated_dist"),
    (
        (pl.col("three").cast(pl.Float64, strict=False) == 1.0) & (pl.col("locationy") <= 14.0)
    ).cast(pl.Float64).alias("is_corner_three")
])

test_df = test_df.with_columns([
    ((pl.col("locationx")**2 + pl.col("locationy")**2).sqrt()).alias("calculated_dist"),
    (
        (pl.col("three").cast(pl.Float64, strict=False) == 1.0) & (pl.col("locationy") <= 14.0)
    ).cast(pl.Float64).alias("is_corner_three")
])

cols = [
    "num_contesters", "distcont1", "distcont2", "distcont3", "distcont4",
    "contested", "gamestate", "shottype",
    "three", "shotclock", "closestdefapproach", "shooterspeed",
    "calculated_dist", "is_corner_three", "rushed_shot"
]

X_train = train_df.select(cols).with_columns(pl.all().cast(pl.Float64, strict=False)).to_numpy()
y_train = train_df["outcome"].to_numpy()

X_test = test_df.select(cols).with_columns(pl.all().cast(pl.Float64, strict=False)).to_numpy()

shot_prob_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
])

shot_prob_pipeline.fit(X_train, y_train)
shot_prob_predict = shot_prob_pipeline.predict_proba(X_test)[:, 1]

predictions_df = pl.DataFrame({
    "shot_id": test_df["shot_id"],
    "make_prob": shot_prob_predict
})

columns_to_drop = [col for col in ["make_prob", "make_prob_right"] if col in submission.columns]
if columns_to_drop:
    submission = submission.drop(columns_to_drop)

submission = submission.join(predictions_df, on="shot_id", how="left")
submission.write_csv("submission.csv")
print(submission.head())