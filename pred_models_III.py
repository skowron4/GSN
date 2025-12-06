import random
import numpy as np
import polars as pl
import json
import optuna
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Dense, Concatenate, Embedding, Flatten,
                                     MultiHeadAttention, LayerNormalization, Lambda,
                                     LSTM, Reshape, Conv1D, GlobalMaxPooling1D,
                                     Dropout, BatchNormalization, Add, Activation)

tf.keras.utils.set_random_seed(42)
np.random.seed(42)
random.seed(42)

# ======================================================================
# 1. Wczytanie danych
# ======================================================================
df = pl.read_csv('replays_data_III.csv')
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
# 5. Ewaluacja modeli Keras
# ======================================================================
def evaluate(model):
    pred = model.predict([X_test_map, X_test_num]).flatten()
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return mae, rmse


# ======================================================================
# 6. CROSS LAYER (DCN)
# ======================================================================
class CrossLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.w = self.add_weight(name="w", shape=(dim, 1), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(dim,), initializer="zeros")

    def call(self, x0, x):
        # x_{l+1} = x0 * (x_l W) + b + x_l
        dot = tf.matmul(x, self.w)                     # (batch,1)
        cross = x0 * dot                               # broadcasting
        return cross + self.b + x


# ======================================================================
# 7. MODEL
# ======================================================================
def build_mlp_model(trial=None):

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else 1e-3
    n_layers = trial.suggest_int("n_layers", 1, 5) if trial else 2

    # ====== MAPA ======
    input_map = Input(shape=(1,), dtype="int32")
    map_emb = Embedding(len(map2id), 8)(input_map)
    map_emb = Flatten()(map_emb)

    # ====== NUMERYCZNE FEATURES ======
    input_num = Input(shape=(len(FEATURES_NUM),), dtype="float32")

    # ====== POŁĄCZENIE ======
    x0 = Concatenate()([map_emb, input_num])   # wektor bazowy

    # ========= CROSS NETWORK =========
    cross = x0
    cross_layer = CrossLayer()
    for _ in range(2):                # dwie warstwy cross – optymalnie
        cross = cross_layer(x0, cross)

    # ======== DEEP MLP ========
    deep = x0
    for i in range(n_layers):
        units = trial.suggest_int(f"units_layer_{i}", 32, 256) if trial else 128
        drop  = trial.suggest_float(f"drop_layer_{i}", 0.1, 0.5) if trial else 0.25

        deep = Dense(units)(deep)
        deep = BatchNormalization()(deep)
        deep = Activation("relu")(deep)
        deep = Dropout(drop)(deep)

    # ======== POŁĄCZENIE CROSS + DEEP ========
    combined = Concatenate()([cross, deep])

    out = Dense(1)(combined)

    model = Model([input_map, input_num], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")

    return model


# ======================================================================
# 7. Optuna — wspólna funkcja do tuningu modeli Keras
# ======================================================================
def run_optuna_keras(build_fn, n_trials=120):
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
# Uruchomienie tuningu dla wszystkich modeli
# ======================================================================
models = {
    "MLP + Embedding": build_mlp_model,
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