import os

import numpy as np
import random
import json
import time

import polars as pl
from tabulate import tabulate

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import optuna
from sklearn.svm import SVR
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
import joblib

# ======================================================================
# Ustawienie ziaren
# ======================================================================
np.random.seed(42)
random.seed(42)

# ======================================================================
# 1. Wczytanie danych
# ======================================================================
df = pl.read_csv("replays_data_agg_class_tier.csv")

df = df.drop("winner")
df = df.sample(fraction=1.0, shuffle=True, seed=42)

TARGET = "duration"
FEATURES_NUM = [c for c in df.columns if c not in ["display_name", "battle_id", TARGET]]

test_size = int(0.2 * df.height)
df_test = df.head(test_size)
df_train = df.tail(-test_size)

# ======================================================================
# 2. Map → ID
# ======================================================================
unique_maps = df.select("display_name").unique()["display_name"].to_list()
map2id = {m: i for i, m in enumerate(unique_maps)}

df_train = df_train.with_columns(
    pl.col("display_name").replace(map2id).cast(pl.Int32).alias("map_id")
)
df_test = df_test.with_columns(
    pl.col("display_name").replace(map2id).cast(pl.Int32).alias("map_id")
)

print("\n=== Mapowanie nazw map na ID ===")

table_data = [[name, id_value] for name, id_value in map2id.items()]
table_data.sort(key=lambda x: x[1])

print(tabulate(
    table_data,
    headers=["Nazwa Mapy", "ID"],
    tablefmt="fancy_grid"
))

# ======================================================================
# 3. Konwersja do NumPy i łączenie cech
# ======================================================================
# Dane dla cech numerycznych
X_train_num = df_train.select(FEATURES_NUM).to_numpy()
X_test_num = df_test.select(FEATURES_NUM).to_numpy()

# Dane dla map_id (używane jako dodatkowa cecha i do filtrowania)
X_train_map = df_train["map_id"].to_numpy().reshape(-1, 1)
X_test_map = df_test["map_id"].to_numpy().reshape(-1, 1)

y_train = df_train[TARGET].to_numpy().astype("float32")
y_test = df_test[TARGET].to_numpy().astype("float32")

# Pełna macierz cech nieskalowanych (z ID mapy)
X_train_unscaled = np.hstack([X_train_num, X_train_map])
X_test_unscaled = np.hstack([X_test_num, X_test_map])

# Wracamy do pierwotnych, nieskalowanych ID mapy DLA FILTROWANIA.
X_train_map_orig = df_train["map_id"].to_numpy().reshape(-1)
X_test_map_orig = df_test["map_id"].to_numpy().reshape(-1)

# ======================================================================
# 4. Skalowanie (tylko do utworzenia X_scaled)
# ======================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_unscaled)
X_test_scaled = scaler.transform(X_test_unscaled)


# ======================================================================
# 5. Funkcje pomocnicze - ZMODYFIKOWANE
# ======================================================================
def get_map_data(map_id_value, use_scaled=True):
    """
    Filtruje dane dla pojedynczej mapy, usuwa kolumnę 'map_id' (ostatnia kolumna)
    i zwraca podzbiory do treningu i testowania.
    Argument use_scaled=False zwraca dane nieskalowane.
    """
    train_mask = X_train_map_orig == map_id_value
    test_mask = X_test_map_orig == map_id_value

    if use_scaled:
        X_train_full = X_train_scaled
        X_test_full = X_test_scaled
    else:
        X_train_full = X_train_unscaled
        X_test_full = X_test_unscaled

    X_train_m = X_train_full[train_mask]
    X_test_m = X_test_full[test_mask]

    X_train_m = X_train_m[:, :-1]
    X_test_m = X_test_m[:, :-1]

    y_train_m = y_train[train_mask]
    y_test_m = y_test[test_mask]

    return X_train_m, X_test_m, y_train_m, y_test_m


# ======================================================================
# 6. Optuna dla pojedynczej mapy (SVR i XGBoost) - ZMODYFIKOWANE
# ======================================================================
def run_optuna_for_single_map(map_id_value, model_type, n_trials=120):
    """
    Optymalizuje hiperparametry i trenuje model (SVR lub XGBoost) dla pojedynczej mapy.
    """
    use_scaled_data = (model_type == "SVR")

    X_train_m, X_test_m, y_train_m, y_test_m = get_map_data(map_id_value, use_scaled=use_scaled_data)

    if len(y_train_m) < 50 or len(y_test_m) < 10:
        return None

    def objective(trial):
        if model_type == "SVR":
            C = trial.suggest_float("C", 1e-1, 1e2, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-2, 1e0, log=True)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
            model = SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)

        elif model_type == "XGBoost":
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'n_jobs': -1,
                'early_stopping_rounds': 10,
            }
            model = XGBRegressor(**params)

        if model_type == "XGBoost":
            val_size = len(X_train_m) // 5
            X_train_split, X_val_split = X_train_m[val_size:], X_train_m[:val_size]
            y_train_split, y_val_split = y_train_m[val_size:], y_train_m[:val_size]

            model.fit(
                X_train_split, y_train_split,
                eval_set=[(X_val_split, y_val_split)],
            )

        else:
            model.fit(X_train_m, y_train_m)

        pred = model.predict(X_test_m)
        mae = mean_absolute_error(y_test_m, pred)
        return mae

    # -------------------------------------------------------------------
    # Optuna - Trenowanie końcowego modelu
    # -------------------------------------------------------------------
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    if model_type == "SVR":
        final_model = SVR(**best_params)
    elif model_type == "XGBoost":
        final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)

    final_model.fit(X_train_m, y_train_m)

    pred = final_model.predict(X_test_m)

    mae = mean_absolute_error(y_test_m, pred)
    r2 = r2_score(y_test_m, pred)

    return {
        "map_id": map_id_value,
        "map_name": [k for k, v in map2id.items() if v == map_id_value][0],
        "mae": mae,
        "r2": r2,
        "params": best_params,
        "model": final_model
    }


# ======================================================================
# 7. Uruchomienie dla każdej mapy i obu modeli - ZMODYFIKOWANE
# ======================================================================

# Definicja wspólnego katalogu
BASE_OUTPUT_DIR = "results/svr_xgb"


def train_and_evaluate_models(model_type, n_trials=120):
    """Uruchamia Optunę i trenuje modele dla wszystkich map."""

    results_maps = []
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    for map_id_value in sorted(map2id.values()):
        map_name = [k for k, v in map2id.items() if v == map_id_value][0]

        start_time = time.time()
        res = run_optuna_for_single_map(map_id_value, model_type, n_trials=n_trials)
        end_time = time.time()

        if res is not None:
            map_name_safe = res['map_name'].replace(' ', '_').replace('/', '_')
            filename = f"{model_type.lower()}_model_{map_id_value}_{map_name_safe}.joblib"
            filepath = os.path.join(BASE_OUTPUT_DIR, filename)

            try:
                joblib.dump(res['model'], filepath)
            except Exception as e:
                print(f"Błąd podczas zapisu pliku {filepath}: {e}")

            del res['model']
            results_maps.append(res)

    return results_maps


results_svr = train_and_evaluate_models("SVR", n_trials=120)

results_xgb = train_and_evaluate_models("XGBoost", n_trials=120)


# ======================================================================
# 8. Wyniki
# ======================================================================
def print_results(model_type, results):
    """Wypisuje końcowe wyniki i wybrane hiperparametry."""
    if not results:
        print(f"\n--- Brak wyników dla modelu {model_type} ---")
        return

    print(f"\n\n=== Wyniki dla każdej mapy ({model_type}) ===")
    table = [
        [r["map_id"], r["map_name"], round(r["mae"], 4), round(r["r2"], 4)]
        for r in results
    ]

    print(tabulate(table, headers=["Map ID", "Mapa", "MAE", "R²"], tablefmt="fancy_grid"))

    print(f"\n=== Parametry wybrane przez Optuna ({model_type}) ===")
    for r in results:
        print(f"\n--- {r['map_name']} (ID {r['map_id']}) ---")
        print(json.dumps(r["params"], indent=4))


# Wyniki dla SVR
print_results("SVR", results_svr)

# Wyniki dla XGBoost
print_results("XGBoost", results_xgb)