import numpy as np
import random
import json
import time

import polars as pl
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import optuna

# ======================================================================
# Ustawienie ziaren
# ======================================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ======================================================================
# 1. Wczytanie danych
# ======================================================================
df = pl.read_csv("replays_data_II.csv")
df = df.drop("winner")
df = df.sample(fraction=1.0, shuffle=True, seed=42)

TARGET = "duration"
FEATURES_NUM = [c for c in df.columns if c not in ["display_name", TARGET]]

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

# ======================================================================
# 3. Konwersja do NumPy
# ======================================================================
X_train_num = df_train.select(FEATURES_NUM).to_numpy()
X_test_num  = df_test.select(FEATURES_NUM).to_numpy()

X_train_map = df_train["map_id"].to_numpy().astype("int64").reshape(-1)
X_test_map  = df_test["map_id"].to_numpy().astype("int64").reshape(-1)

y_train = df_train[TARGET].to_numpy().astype("float32")
y_test  = df_test[TARGET].to_numpy().astype("float32")

# ======================================================================
# 4. Skalowanie
# ======================================================================
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num  = scaler.transform(X_test_num)

# ======================================================================
# 5A. DCN: Cross Network
# ======================================================================
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, xl):
        # x0 * (w^T xl) + b + xl
        cross = torch.sum(xl * self.w, dim=1, keepdim=True) * x0
        return cross + self.b + xl


class CrossNetwork(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([CrossLayer(dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        xl = x
        for layer in self.layers:
            xl = layer(x0, xl)
        return xl


# ======================================================================
# 5B. Model: MLP + DCN
# ======================================================================
class DCNModel(nn.Module):
    def __init__(self, num_features, n_maps, embed_dim,
                 n_layers, units, drops,
                 cross_layers):
        super().__init__()

        self.embedding = nn.Embedding(n_maps, embed_dim)

        cross_in = num_features + embed_dim

        self.cross_net = CrossNetwork(cross_in, cross_layers)

        # deep network
        layers = []
        in_dim = cross_in

        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, units[i]))
            layers.append(nn.BatchNorm1d(units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drops[i]))
            in_dim = units[i]

        self.deep = nn.Sequential(*layers)

        # połączone wyjścia DCN + MLP
        self.final = nn.Linear(in_dim + cross_in, 1)

    def forward(self, map_id, x_num):
        emb = self.embedding(map_id)
        x = torch.cat([emb, x_num], dim=1)

        x_cross = self.cross_net(x)
        x_deep = self.deep(x)

        out = torch.cat([x_cross, x_deep], dim=1)
        out = self.final(out)
        return out.squeeze(1)


class CauchyLoss(nn.Module):
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def forward(self, pred, target):
        return torch.mean(torch.log(1 + ((pred - target) / self.c)**2))



# ======================================================================
# 6. Trening PyTorch
# ======================================================================
def train_model(model, X_map, X_num, y, epochs=500, batch_size=128):

    X_map = torch.tensor(X_map, dtype=torch.long, device=DEVICE)
    X_num = torch.tensor(X_num, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.float32, device=DEVICE)

    dataset_size = len(y)
    optimizer = optim.Adam(model.parameters(), lr=model.lr)
    loss_fn = CauchyLoss(c=model.c_param)

    best_loss = float("inf")
    patience = 8
    patience_counter = 0

    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(dataset_size)
        X_map_sh = X_map[perm]
        X_num_sh = X_num[perm]
        y_sh = y[perm]

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            b_map = X_map_sh[start:end]
            b_num = X_num_sh[start:end]
            b_y = y_sh[start:end]

            pred = model(b_map, b_num)
            loss = loss_fn(pred, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation (20% split taken inside arrays)
        val_size = dataset_size // 5
        with torch.no_grad():
            pred_val = model(X_map[:val_size], X_num[:val_size])
            val_loss = loss_fn(pred_val, y[:val_size]).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ======================================================================
# 7. Optuna dla jednej mapy
# ======================================================================
def run_optuna_for_single_map(map_id_value, n_trials=40):

    train_mask = X_train_map == map_id_value
    test_mask  = X_test_map == map_id_value

    X_train_num_m = X_train_num[train_mask]
    X_test_num_m  = X_test_num[test_mask]

    X_train_map_m = X_train_map[train_mask]
    X_test_map_m  = X_test_map[test_mask]

    y_train_m = y_train[train_mask]
    y_test_m  = y_test[test_mask]

    if len(y_train_m) < 50 or len(y_test_m) < 10:
        return None

    def objective(trial):
        # hiperparametry
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 5)

        units = [trial.suggest_int(f"units_{i}", 16, 256) for i in range(n_layers)]
        drops = [trial.suggest_float(f"drop_{i}", 0.1, 0.5) for i in range(n_layers)]

        cross_layers = trial.suggest_int("cross_layers", 1, 4)
        c_param = trial.suggest_float("cauchy_c", 0.5, 5.0)

        # === TU TWORZYMY MODEL ===
        model = DCNModel(
            num_features=len(FEATURES_NUM),
            n_maps=len(map2id),
            embed_dim=8,
            n_layers=n_layers,
            units=units,
            drops=drops,
            cross_layers=cross_layers
        ).to(DEVICE)

        model.lr = lr
        model.c_param = c_param

        # trenowanie
        train_model(model, X_train_map_m, X_train_num_m, y_train_m)

        # walidacja
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.tensor(X_test_map_m, dtype=torch.long, device=DEVICE),
                torch.tensor(X_test_num_m, dtype=torch.float32, device=DEVICE),
            ).cpu().numpy()

        r2 = r2_score(y_test_m, pred)

        # minimalizujemy -R² (czyli maksymalizujemy R²)
        return -r2

    # -------------------------------------------------------------------
    # Optuna
    # -------------------------------------------------------------------

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    n_layers = best["n_layers"]
    units = [best[f"units_{i}"] for i in range(n_layers)]
    drops = [best[f"drop_{i}"] for i in range(n_layers)]

    # finalny model
    model = DCNModel(
        num_features=len(FEATURES_NUM),
        n_maps=len(map2id),
        embed_dim=8,
        n_layers=n_layers,
        units=units,
        drops=drops,
        cross_layers=best["cross_layers"]
    ).to(DEVICE)

    model.lr = best["lr"]
    model.c_param = best["cauchy_c"]

    train_model(model, X_train_map_m, X_train_num_m, y_train_m, epochs=500)

    # wyniki końcowe
    with torch.no_grad():
        pred = model(
            torch.tensor(X_test_map_m, dtype=torch.long, device=DEVICE),
            torch.tensor(X_test_num_m, dtype=torch.float32, device=DEVICE),
        ).cpu().numpy()

    mae = mean_absolute_error(y_test_m, pred)
    r2 = r2_score(y_test_m, pred)

    return {
        "map_id": map_id_value,
        "map_name": [k for k, v in map2id.items() if v == map_id_value][0],
        "mae": mae,
        "r2": r2,
        "cauchy_c": best["cauchy_c"],
        "params": best
    }



# ======================================================================
# 8. Uruchomienie dla każdej mapy
# ======================================================================
results_maps = []

for map_id_value in sorted(map2id.values()):
    print(f"\n=== Optuna dla mapy: {map_id_value} ===")
    res = run_optuna_for_single_map(map_id_value, n_trials=40)
    if res is not None:
        results_maps.append(res)

# ======================================================================
# 9. Wyniki
# ======================================================================
print("\n\n=== Wyniki dla każdej mapy ===")
table = [
    [r["map_id"], r["map_name"], round(r["mae"], 4), round(r["r2"], 4)]
    for r in results_maps
]

print(tabulate(table, headers=["Map ID", "Mapa", "MAE", "R²"], tablefmt="fancy_grid"))

print("\n=== Parametry wybrane przez Optuna ===")
for r in results_maps:
    print(f"\n--- {r['map_name']} (ID {r['map_id']}) ---")
    print(json.dumps(r["params"], indent=4))
