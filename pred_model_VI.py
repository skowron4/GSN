import os
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
# Ustawienie ziaren (powtarzalność wyników)
# ======================================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używane urządzenie: {DEVICE}")

# Folder na wyniki
os.makedirs("results", exist_ok=True)
os.makedirs("results/mapy", exist_ok=True)

# ======================================================================
# 1. Wczytanie danych
# ======================================================================
print("Wczytywanie danych...")
# Upewnij się, że plik csv jest w tym samym folderze
df = pl.read_csv("replays_data_agg_class_tier.csv")
df = df.drop("winner")
df = df.sample(fraction=1.0, shuffle=True, seed=42)

TARGET = "duration"
FEATURES_NUM = [c for c in df.columns if c not in ["display_name", "battle_id", TARGET]]

test_size = int(0.2 * df.height)
df_test = df.head(test_size)
df_train = df.tail(-test_size)

# ======================================================================
# 2. Mapowanie nazw map na ID
# ======================================================================
unique_maps = df.select("display_name").unique()["display_name"].to_list()
map2id = {m: i for i, m in enumerate(unique_maps)}
id2map = {i: m for m, i in map2id.items()}  # Słownik odwrotny do raportów

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
X_test_num = df_test.select(FEATURES_NUM).to_numpy()

X_train_map = df_train["map_id"].to_numpy().astype("int64").reshape(-1)
X_test_map = df_test["map_id"].to_numpy().astype("int64").reshape(-1)

y_train = df_train[TARGET].to_numpy().astype("float32")
y_test = df_test[TARGET].to_numpy().astype("float32")

# ======================================================================
# 4. Skalowanie danych numerycznych
# ======================================================================
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

X_test_map_tensor = torch.tensor(X_test_map, dtype=torch.long, device=DEVICE)
X_test_num_tensor = torch.tensor(X_test_num, dtype=torch.float32, device=DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)


# ======================================================================
# 5. Definicja Modelu (DCN)
# ======================================================================
class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, xl):
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


class DCNModel(nn.Module):
    def __init__(self, num_features, n_maps, embed_dim, n_layers, units, drops, cross_layers):
        super().__init__()
        self.embedding = nn.Embedding(n_maps, embed_dim)
        cross_in = num_features + embed_dim
        self.cross_net = CrossNetwork(cross_in, cross_layers)

        layers = []
        in_dim = cross_in
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, units[i]))
            layers.append(nn.BatchNorm1d(units[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drops[i]))
            in_dim = units[i]

        self.deep = nn.Sequential(*layers)
        self.final = nn.Linear(in_dim + cross_in, 1)

    def forward(self, map_id, x_num):
        emb = self.embedding(map_id)
        x = torch.cat([emb, x_num], dim=1)
        x_cross = self.cross_net(x)
        x_deep = self.deep(x)
        out = torch.cat([x_cross, x_deep], dim=1)
        return self.final(out).squeeze(1)


# ======================================================================
# 6. Funkcja Treningowa (Standardowa)
# ======================================================================
def train_model(model, X_map, X_num, y, epochs=100, batch_size=256):
    X_map_t = torch.tensor(X_map, dtype=torch.long, device=DEVICE)
    X_num_t = torch.tensor(X_num, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=model.lr)

    loss_fn = nn.HuberLoss(delta=model.delta_param)

    dataset_size = len(y)
    best_loss = float("inf")
    patience = 8
    patience_counter = 0
    best_state = model.state_dict()

    val_cut = int(dataset_size * 0.9)
    X_map_tr, X_map_val = X_map_t[:val_cut], X_map_t[val_cut:]
    X_num_tr, X_num_val = X_num_t[:val_cut], X_num_t[val_cut:]
    y_tr, y_val = y_t[:val_cut], y_t[val_cut:]

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr))

        for start in range(0, len(y_tr), batch_size):
            end = start + batch_size
            idx = perm[start:end]
            pred = model(X_map_tr[idx], X_num_tr[idx])
            loss = loss_fn(pred, y_tr[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Walidacja
        model.eval()
        with torch.no_grad():
            pred_val = model(X_map_val, X_num_val)
            val_loss = loss_fn(pred_val, y_val).item()

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
# 7. Optuna - ZOPTYMALIZOWANA POD TWÓJ ZBIÓR
# ======================================================================
def objective(trial):

    lr = trial.suggest_float("lr", 5e-4, 1e-2, log=True)
    n_layers = trial.suggest_int("n_layers", 2, 5)  # Minimum 2 warstwy
    embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 24, 32])  # Wywalamy 4, za mało

    units = [trial.suggest_int(f"units_{i}", 64, 512) for i in range(n_layers)]
    drops = [trial.suggest_float(f"drop_{i}", 0.1, 0.4) for i in range(n_layers)]

    cross_layers = trial.suggest_int("cross_layers", 1, 4)

    delta_param = trial.suggest_float("huber_delta", 10.0, 90.0)

    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    model = DCNModel(
        num_features=len(FEATURES_NUM),
        n_maps=len(map2id),
        embed_dim=embed_dim,
        n_layers=n_layers,
        units=units,
        drops=drops,
        cross_layers=cross_layers
    ).to(DEVICE)

    model.lr = lr
    model.delta_param = delta_param
    loss_fn = nn.HuberLoss(delta=delta_param)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # --- 3. Pętla treningowa z PRUNINGIEM ---
    # Używamy ręcznej pętli tutaj, żeby Optuna mogła przerywać słabe triale

    X_map_t = torch.tensor(X_train_map, dtype=torch.long, device=DEVICE)
    X_num_t = torch.tensor(X_train_num, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    dataset_len = len(y_train)
    n_epochs = 80

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(dataset_len)

        for start in range(0, dataset_len, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X_map_t[idx], X_num_t[idx])
            loss = loss_fn(pred, y_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_test_map_tensor, X_test_num_tensor)
            val_loss = loss_fn(pred_val, y_test_tensor).item()

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


print("Rozpoczynam Optunę...")
# Używamy MedianPruner - bardzo skuteczny do szybkiego ucinania słabych modeli
study = optuna.create_study(direction="minimize",
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
study.optimize(objective, n_trials=120)

best_params = study.best_params
print("\nNajlepsze parametry:", json.dumps(best_params, indent=4))

# ======================================================================
# 8. Trening ZWYCIĘSKIEGO modelu i ZAPIS METADANYCH
# ======================================================================
print("\n=== Trenuję ostateczny (najlepszy) model... ===")

n_layers = best_params["n_layers"]
units = [best_params[f"units_{i}"] for i in range(n_layers)]
drops = [best_params[f"drop_{i}"] for i in range(n_layers)]
embed_dim = best_params["embed_dim"]

final_model = DCNModel(
    num_features=len(FEATURES_NUM),
    n_maps=len(map2id),
    embed_dim=embed_dim,
    n_layers=n_layers,
    units=units,
    drops=drops,
    cross_layers=best_params["cross_layers"]
).to(DEVICE)

final_model.lr = best_params["lr"]
final_model.delta_param = best_params["huber_delta"]
final_batch_size = best_params["batch_size"]

final_model = train_model(final_model, X_train_map, X_train_num, y_train, epochs=1000, batch_size=final_batch_size)

print("Zapisywanie modelu i parametrów...")

torch.save(final_model.state_dict(), "results/BEST_MODEL_weights.pth")

metadata = {
    "hyperparameters": best_params,
    "map_mapping": map2id,
    "features_list": FEATURES_NUM,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist()
}

with open("results/model_config.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print("  -> Zapisano wagi do: results/BEST_MODEL_weights.pth")
print("  -> Zapisano konfigurację do: results/model_config.json")

# ======================================================================
# 9. GENEROWANIE RAPORTÓW (GLOBALNE + PER MAPA)
# ======================================================================
print("\n=== Generowanie raportów... ===")

final_model.eval()
with torch.no_grad():
    all_preds = final_model(X_test_map_tensor, X_test_num_tensor).cpu().numpy()

mae_global = mean_absolute_error(y_test, all_preds)
r2_global = r2_score(y_test, all_preds)

print("\n" + "=" * 40)
print(" WYNIKI GLOBALNE (CAŁY ZBIÓR TESTOWY)")
print("=" * 40)
print(f"Global MAE: {mae_global:.4f} sekund")
print(f"Global R2:  {r2_global:.4f}")
print("=" * 40 + "\n")

results_list = []
unique_test_ids = np.unique(X_test_map)

for map_name, m_id in map2id.items():

    mask = (X_test_map == m_id)
    count = np.sum(mask)

    mae_m = 0.0
    r2_m = 0.0
    y_true_m = []
    y_pred_m = []

    if count > 0:
        y_true_m = y_test[mask]
        y_pred_m = all_preds[mask]
        mae_m = mean_absolute_error(y_true_m, y_pred_m)
        r2_m = r2_score(y_true_m, y_pred_m)

    results_list.append({
        "ID": m_id,
        "Mapa": map_name,
        "MAE (sekundy)": round(mae_m, 4),
        "R2": round(r2_m, 4),
        "Próbki": count
    })

    safe_name = map_name.replace(" ", "_").replace("'", "")
    filename = f"results/mapy/Wynik_{safe_name}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Mapa: {map_name} (ID: {m_id})\n")
        f.write("=" * 30 + "\n")
        f.write(f"Liczba próbek testowych: {count}\n")
        f.write(f"MAE (średni błąd):       {mae_m:.4f}\n")
        f.write(f"R2 (dopasowanie):        {r2_m:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Przykładowe predykcje (True vs Pred):\n")

        if count > 0:
            sample_size = min(5, count)
            indices = np.random.choice(len(y_true_m), size=sample_size, replace=False)
            for idx in indices:
                yt = y_true_m[idx]
                yp = y_pred_m[idx]
                diff = yp - yt
                f.write(f"  {yt:.2f} vs {yp:.2f} (diff: {diff:.2f})\n")
        else:
            f.write("  Brak próbek w zbiorze testowym dla tej mapy.\n")

df_res = pl.DataFrame(results_list).sort("MAE (sekundy)", descending=True)
df_res.write_csv("results/PODSUMOWANIE_MAP.csv")

print("Gotowe! Wyniki szczegółowe w folderze results/mapy/")

try:
    print(tabulate(df_res.to_pandas(), headers="keys", tablefmt="fancy_grid", showindex=False))
except ImportError:
    print(df_res)