"""
Walk-forward regression model.

KEY IDEA (calibration): the model does NOT predict the score from scratch.
It predicts how wrong the betting line will be:

    target = actual home margin  -  spread_line

A model with no information predicts 0, i.e. "the market is right". The
regression is penalised (ridge) so a feature only moves the prediction away
from the market if it has earned that in past seasons.

WALK-FORWARD: to predict season S we fit only on seasons before S.
The model never sees the season it is being tested on.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-1, 5, 40)


def add_model_columns(df):
    df = df.copy()
    df["elo_vs_line"] = df.pred_elo - df.spread_line
    df["line"] = df.spread_line
    return df


def _season_splits(train):
    """Leave-one-season-out folds: the penalty strength is chosen by how well
    the model predicts whole seasons it didn't see, not random single games.
    This makes the model much more cautious when it has little history."""
    seasons = train.season.to_numpy()
    return [(np.where(seasons != s)[0], np.where(seasons == s)[0]) for s in np.unique(seasons)]


def _fit(train, features, target="result", line="spread_line"):
    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=ALPHAS, cv=_season_splits(train),
                                  scoring="neg_mean_squared_error"))
    y = train[target] - train[line]
    model.fit(train[features], y)
    resid = y - model.predict(train[features])
    # Centre the model: its average miss vs the line in training is NOT an
    # edge. Scores are lopsided (blowouts/shootouts pull the average above
    # the line even when overs and unders each hit ~50%), so we only keep
    # the game-specific part of the prediction.
    model.offset_ = float(model.predict(train[features]).mean())
    return model, float(resid.std())


def predict_dev(model, X):
    """Predicted points away from the posted line, centred (see _fit)."""
    return model.predict(X) - getattr(model, "offset_", 0.0)


def walk_forward(df, features, test_seasons, train_start=2002,
                 target="result", line="spread_line", out_col="pred_margin"):
    """Returns test-season rows with a prediction column and sd (spread of outcomes).
    Spreads: target=result (home margin), line=spread_line.
    Totals:  target=total (combined points), line=total_line, out_col=pred_total."""
    df = add_model_columns(df)
    ok = df[target].notna() & df[line].notna()
    parts, last_model = [], None
    for season in test_seasons:
        train = df[ok & (df.season >= train_start) & (df.season < season)]
        test = df[ok & (df.season == season)].copy()
        if test.empty:
            continue
        model, sd = _fit(train, features, target, line)
        test[out_col] = test[line] + predict_dev(model, test[features])
        test["sd"] = sd
        parts.append(test)
        last_model = model
    return pd.concat(parts), last_model


def fit_latest(df, features, train_start=2002, target="result", line="spread_line"):
    """Fit on every completed game so far (for predicting upcoming games)."""
    df = add_model_columns(df)
    train = df[df[target].notna() & df[line].notna() & (df.season >= train_start)]
    return _fit(train, features, target, line)


def explain(model, features):
    """Coefficient of each feature, in points per 1 standard deviation."""
    ridge = model[-1]
    return (pd.Series(ridge.coef_, index=features)
            .sort_values(key=abs, ascending=False).round(3))
