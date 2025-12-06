import random

import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
import optuna
import xgboost as xgb
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Dense, Concatenate, Embedding, Flatten,
                                     MultiHeadAttention, LayerNormalization, Lambda,
                                     LSTM, Reshape, Conv1D, GlobalMaxPooling1D,
                                     Dropout, BatchNormalization, Add, Activation)
import json
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

# ======================================================================
# 1. Wczytanie danych
# ======================================================================
df = pl.read_csv("replays_data.csv")
df = df.drop("winnerTeam")

TARGET = "duration"
FEATURES_NUM = [c for c in df.columns if c not in ["map", TARGET]]

df = df.sample(fraction=1, shuffle=True)

test_size = int(0.2 * df.height)
df_test  = df.head(test_size)
df_train = df.tail(-test_size)

# ======================================================================
# 2. Map → int32
# ======================================================================
unique_maps = df.select("map").unique()["map"].to_list()
map2id = {m: i for i, m in enumerate(unique_maps)}

df_train = df_train.with_columns(
    pl.col("map").replace(map2id).cast(pl.Int32).alias("map_id")
)
df_test = df_test.with_columns(
    pl.col("map").replace(map2id).cast(pl.Int32).alias("map_id")
)

# ======================================================================
# 3. Konwersja do NumPy
# ======================================================================
X_train_num = df_train.select(FEATURES_NUM).to_numpy()
X_test_num  = df_test.select(FEATURES_NUM).to_numpy()

X_train_map = df_train["map_id"].to_numpy().astype("int32").reshape(-1, 1)
X_test_map  = df_test["map_id"].to_numpy().astype("int32").reshape(-1, 1)

y_train = df_train[TARGET].to_numpy().astype("float32")
y_test  = df_test[TARGET].to_numpy().astype("float32")

# ======================================================================
# 4. Skalowanie
# ======================================================================
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num  = scaler.transform(X_test_num)


# ======================================================================
# 5. Ewaluacja modeli Keras
# ======================================================================
def evaluate(model):
    pred = model.predict([X_test_map, X_test_num]).flatten()
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return mae, rmse


# ======================================================================
# 6. MODELE
# ======================================================================

def build_mlp_model(trial=None):

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3

    n_layers = trial.suggest_int("n_layers", 1, 5) if trial else 2

    input_map = Input(shape=(1,), dtype="int32")
    emb = Embedding(len(map2id), 8)(input_map)
    emb = Flatten()(emb)

    input_num = Input(shape=(len(FEATURES_NUM),), dtype="float32")

    x = Concatenate()([emb, input_num])

    for i in range(n_layers):
        units = trial.suggest_int(f"units_layer_{i}", 16, 256) if trial else 64
        drop  = trial.suggest_float(f"drop_layer_{i}", 0.1, 0.5) if trial else 0.3

        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(drop)(x)

    out = Dense(1)(x)

    model = Model([input_map, input_num], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model


def build_wide_deep_model(trial=None):

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3
    n_layers = trial.suggest_int("n_layers_wd", 1, 5) if trial else 2

    input_map = Input(shape=(1,), dtype="int32")
    emb = Embedding(len(map2id), 16)(input_map)
    emb = Flatten()(emb)

    input_num = Input(shape=(len(FEATURES_NUM),), dtype="float32")

    wide = Dense(1)(input_num)

    deep = Concatenate()([emb, input_num])

    for i in range(n_layers):
        units = trial.suggest_int(f"wd_units_{i}", 32, 256) if trial else 128
        drop  = trial.suggest_float(f"wd_drop_{i}", 0.1, 0.5) if trial else 0.3

        deep = Dense(units, activation=None)(deep)
        deep = BatchNormalization()(deep)
        deep = Activation("relu")(deep)
        deep = Dropout(drop)(deep)

    out = Dense(1)(Concatenate()([wide, deep]))

    model = Model([input_map, input_num], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model


def build_attention_model(trial=None):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3
    dim = trial.suggest_int("att_dim", 8, 32) if trial else 16
    n_layers = trial.suggest_int("att_layers", 1, 5) if trial else 2

    input_map = Input(shape=(1,), dtype="int32")
    emb = Embedding(len(map2id), dim)(input_map)
    emb = Flatten()(emb)

    input_num = Input(shape=(len(FEATURES_NUM),))
    num_seq = Lambda(lambda x: tf.expand_dims(x, axis=1))(input_num)

    attn_out = MultiHeadAttention(
        num_heads=2,
        key_dim=dim,
        dropout=0.1,
    )(num_seq, num_seq)

    attn_out = Add()([num_seq, attn_out])
    attn_out = LayerNormalization()(attn_out)
    attn_out = Flatten()(attn_out)

    x = Concatenate()([emb, attn_out])

    for i in range(n_layers):
        units = trial.suggest_int(f"att_units_{i}", 32, 256) if trial else 128
        drop = trial.suggest_float(f"att_drop_{i}", 0.1, 0.5) if trial else 0.3

        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(drop)(x)

    out = Dense(1)(x)

    model = Model([input_map, input_num], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model


def build_lstm_model(trial=None):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3
    lstm_units = trial.suggest_int("lstm_units", 16, 64) if trial else 32
    n_layers = trial.suggest_int("lstm_layers", 1, 4) if trial else 2

    input_map = Input(shape=(1,), dtype="int32")
    emb = Embedding(len(map2id), 8)(input_map)
    emb = Flatten()(emb)

    input_num = Input(shape=(len(FEATURES_NUM),))
    seq = Reshape((len(FEATURES_NUM), 1))(input_num)

    lstm_out = LSTM(lstm_units, return_sequences=False)(seq)
    lstm_out = BatchNormalization()(lstm_out)

    x = Concatenate()([emb, lstm_out])

    for i in range(n_layers):
        units = trial.suggest_int(f"lstm_dense_units_{i}", 32, 256) if trial else 128
        drop  = trial.suggest_float(f"lstm_dense_drop_{i}", 0.1, 0.5) if trial else 0.3

        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(drop)(x)

    out = Dense(1)(x)

    model = Model([input_map, input_num], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model

def build_cnn_model(trial=None):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3
    filters = trial.suggest_int("filters", 16, 64) if trial else 32
    n_layers = trial.suggest_int("cnn_layers", 1, 4) if trial else 2

    input_map = Input(shape=(1,), dtype="int32")
    emb = Embedding(len(map2id), 8)(input_map)
    emb = Flatten()(emb)

    input_num = Input(shape=(len(FEATURES_NUM),))
    seq = Reshape((len(FEATURES_NUM), 1))(input_num)

    conv = Conv1D(filters=filters, kernel_size=3, activation=None)(seq)
    conv = BatchNormalization()(conv)
    conv = Activation("relu")(conv)

    pool = GlobalMaxPooling1D()(conv)

    x = Concatenate()([emb, pool])

    for i in range(n_layers):
        units = trial.suggest_int(f"cnn_units_{i}", 32, 256) if trial else 128
        drop = trial.suggest_float(f"cnn_drop_{i}", 0.1, 0.5) if trial else 0.3

        x = Dense(units, activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(drop)(x)

    out = Dense(1)(x)

    model = Model([input_map, input_num], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model


# ======================================================================
# 7. Optuna — wspólna funkcja do tuningu modeli Keras
# ======================================================================
def run_optuna_keras(build_fn, n_trials=20):
    def objective(trial):
        model = build_fn(trial)
        model.fit(
            [X_train_map, X_train_num], y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=128,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=8,
                    restore_best_weights=True,
                    monitor="val_loss"
                ),
                tf.keras.callbacks.TerminateOnNaN()
            ]
        )
        pred = model.predict([X_test_map, X_test_num]).flatten()
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value

# ======================================================================
# Funkcja tuningu XGBoost
# ======================================================================
def run_optuna_xgb(n_trials=30):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "eta": trial.suggest_float("eta", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        }

        dtrain = xgb.DMatrix(X_train_num, label=y_train)
        dvalid = xgb.DMatrix(X_test_num, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        pred = model.predict(dvalid)
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # najlepsze parametry i wartość
    return study.best_params, study.best_value

# Random forest
def run_optuna_random_forest(n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train_num, y_train)

        pred = model.predict(X_test_num)
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


# ExtraTrees
def run_optuna_extra_trees(n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        model = ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train_num, y_train)

        pred = model.predict(X_test_num)
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def run_optuna_gbr(n_trials=30):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }

        model = GradientBoostingRegressor(**params)
        model.fit(X_train_num, y_train)

        pred = model.predict(X_test_num)
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def run_optuna_knn(n_trials=20):
    def objective(trial):
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        }

        model = KNeighborsRegressor(**params)
        model.fit(X_train_num, y_train)

        pred = model.predict(X_test_num)
        mae = mean_absolute_error(y_test, pred)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


# ======================================================================
# Uruchomienie tuningu dla wszystkich modeli
# ======================================================================
models = {
    # "MLP + Embedding": build_mlp_model,
    # "Wide & Deep": build_wide_deep_model,
    # "Attention MLP": build_attention_model,
    # "LSTM": build_lstm_model,
    # "1D CNN": build_cnn_model
}

results = []

# Tuning modeli Keras
for name, builder in models.items():
    print(f"\n=== Optuna: {name} ===")
    best_params, best_mae = run_optuna_keras(builder, n_trials=15)
    results.append({"model": name, "mae": best_mae, "params": best_params})

# Tuning XGBoost
print("\n=== Optuna: XGBoost ===")
best_params, best_mae = run_optuna_xgb(n_trials=30)
results.append({"model": "XGBoost", "mae": best_mae, "params": best_params})

print("\n=== Optuna: Random Forest ===")
best_params, best_mae = run_optuna_random_forest()
results.append({"model": "RandomForest", "mae": best_mae, "params": best_params})

print("\n=== Optuna: Extra Trees ===")
best_params, best_mae = run_optuna_extra_trees()
results.append({"model": "ExtraTrees", "mae": best_mae, "params": best_params})

print("\n=== Optuna: GradientBoosting ===")
best_params, best_mae = run_optuna_gbr()
results.append({"model": "GradientBoosting", "mae": best_mae, "params": best_params})

print("\n=== Optuna: KNN ===")
best_params, best_mae = run_optuna_knn()
results.append({"model": "KNN", "mae": best_mae, "params": best_params})


# ======================================================================
# Wypisanie wyników w estetycznej tabeli
# ======================================================================
print("\n\n=== Najlepsze wyniki Optuna dla wszystkich modeli ===")
table = [[r["model"], round(r["mae"], 4)] for r in results]
print(tabulate(table, headers=["Model", "MAE"], tablefmt="fancy_grid"))

print("\n=== Parametry wybrane przez Optuna ===")
for r in results:
    print(f"\n--- {r['model']} ---")
    print(json.dumps(r["params"], indent=4))
