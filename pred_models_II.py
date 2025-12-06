import numpy as np
import random
import json

import polars as pl
from tabulate import tabulate

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Activation,
    BatchNormalization, Embedding, Flatten,
    Concatenate
)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna


# ======================================================================
# USTAWIENIE ZIAREN LOSOWOŚCI
# ======================================================================
tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)


# ======================================================================
# 1. Wczytanie danych
# ======================================================================
df = pl.read_csv('replays_data_II.csv')
df = df.drop("winner")
df = df.sample(fraction=1.0, shuffle=True, seed=42)

TARGET = "duration"
FEATURES_NUM = [c for c in df.columns if c not in ["display_name", TARGET]]

test_size = int(0.2 * df.height)
df_test = df.head(test_size)
df_train = df.tail(-test_size)


# ======================================================================
# 2. Map → int32
# ======================================================================
unique_maps = df.select("display_name").unique()["display_name"].to_list()
map2id = {m: i for i, m in enumerate(unique_maps)}

df_train = df_train.with_columns(
    pl.col("display_name").replace(map2id).cast(pl.Int32).alias("map_id")
)
df_test = df_test.with_columns(
    pl.col("display_name").replace(map2id).cast(pl.Int32).alias("map_id")
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
# 5. Model — MLP + Embedding
# ======================================================================
def build_mlp_model(trial=None):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3
    n_layers = trial.suggest_int("n_layers", 1, 5) if trial else 2

    # ===== Wejście mapy =====
    input_map = Input(shape=(1,), dtype="int32")
    map_emb = Embedding(len(map2id), 8)(input_map)
    map_emb = Flatten()(map_emb)

    # ===== Wejście cech liczbowych =====
    input_num = Input(shape=(len(FEATURES_NUM),), dtype="float32")

    cross_input = Concatenate()([map_emb, input_num])

    cross = Dense(32, activation="relu")(cross_input)
    cross = BatchNormalization()(cross)
    x = Concatenate()([cross_input, cross])

    # ===== MLP =====
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


# ======================================================================
# 6. Optuna dla jednej mapy
# ======================================================================
def run_optuna_for_single_map(map_id_value, n_trials=25):

    # Wybranie rekordów dotyczących tej mapy
    train_mask = X_train_map.flatten() == map_id_value
    test_mask  = X_test_map.flatten()  == map_id_value

    X_train_num_m = X_train_num[train_mask]
    X_test_num_m  = X_test_num[test_mask]

    X_train_map_m = X_train_map[train_mask]
    X_test_map_m  = X_test_map[test_mask]

    y_train_m = y_train[train_mask]
    y_test_m  = y_test[test_mask]

    # Aby uniknąć map z małą liczbą próbek
    if len(y_train_m) < 50 or len(y_test_m) < 10:
        return None

    # ----------------------------
    # Objective Optuny (maksymalizujemy R² → minimalizujemy -R²)
    # ----------------------------
    def objective(trial):
        model = build_mlp_model(trial)

        model.fit(
            [X_train_map_m, X_train_num_m], y_train_m,
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

        pred = model.predict([X_test_map_m, X_test_num_m], verbose=0).flatten()
        r2 = r2_score(y_test_m, pred)
        return -r2

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # ----------------------------
    # Trening finalny z najlepszymi parametrami
    # ----------------------------
    best_model = build_mlp_model(optuna.trial.FixedTrial(study.best_params))
    best_model.fit(
        [X_train_map_m, X_train_num_m], y_train_m,
        validation_split=0.2,
        epochs=50,
        batch_size=128,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=8,
                restore_best_weights=True,
                monitor="val_loss"
            )
        ]
    )

    pred = best_model.predict([X_test_map_m, X_test_num_m], verbose=0).flatten()

    mae = mean_absolute_error(y_test_m, pred)
    r2 = r2_score(y_test_m, pred)

    return {
        "map_id": map_id_value,
        "map_name": [k for k, v in map2id.items() if v == map_id_value][0],
        "mae": mae,
        "r2": r2,
        "params": study.best_params,
    }


# ======================================================================
# 7. Uruchomienie dla każdej mapy
# ======================================================================
results_maps = []

for map_id_value in sorted(map2id.values()):
    print(f"\n=== Optuna dla mapy: {map_id_value} ===")
    res = run_optuna_for_single_map(map_id_value, n_trials=20)
    if res is not None:
        results_maps.append(res)


# ======================================================================
# 8. Tabela wyników
# ======================================================================
print("\n\n=== Wyniki dla każdej mapy ===")
table = [
    [r["map_id"], r["map_name"], round(r["mae"], 4), round(r["r2"], 4)]
    for r in results_maps
]

print(tabulate(table, headers=["Map ID", "Mapa", "MAE", "R²"], tablefmt="fancy_grid"))


# ======================================================================
# 9. Parametry Optuny
# ======================================================================
print("\n=== Parametry wybrane przez Optuna ===")
for r in results_maps:
    print(f"\n--- {r['map_name']} (ID {r['map_id']}) ---")
    print(json.dumps(r["params"], indent=4))
