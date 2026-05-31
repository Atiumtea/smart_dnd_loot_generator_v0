import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
import pickle
import os
import random
from models import DnDItemRanker, ITEM_TYPES

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 1. ДАТАСЕТ
# ==========================================
class DnDDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. ПОДГОТОВКА ДАННЫХ
# ==========================================
def load_training_data():
    print("📦 Загрузка размеченного датасета (LLM + Эвристики)...")
    try:
        df = pd.read_csv('llm_gold_standard.csv', sep=';')

        if len(df) < 50:
            print("⚠️ В датасете мало данных! Рекомендуется сгенерировать больше через llm_annotator.py")

        print(f"✨ Найдено {len(df)} эталонных оценок.")
        return df

    except FileNotFoundError:
        print("❌ Датасет 'llm_gold_standard.csv' не найден. Сначала запусти llm_annotator.py!")
        exit()

# ==========================================
# 3. ОСНОВНОЙ СКРИПТ ОБУЧЕНИЯ
# ==========================================
def train_and_evaluate():
    set_seed(42)
    df = load_training_data()

    type_features = [f'type_{t.replace(" ", "_")}' for t in ITEM_TYPES]
    continuous_features = ['loc_score', 'party_score', 'story_importance', 'rarity_val', 'level_rarity_delta',
                           'synergy_density']
    binary_features = ['is_consumable', 'is_duplicate'] + type_features
    features = continuous_features + binary_features

    for col in features + ['target_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    X = df[features].values.astype(np.float32)
    y = df['target_y'].values.astype(np.float32)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42
    )

    cont_indices = list(range(len(continuous_features)))
    bin_indices = list(range(len(continuous_features), len(features)))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), cont_indices),
            ('bin', 'passthrough', bin_indices)
        ])

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    with open('preprocessor_hybrid.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    train_dataset = DnDDataset(X_train_scaled, y_train)
    val_dataset = DnDDataset(X_val_scaled, y_val)
    test_dataset = DnDDataset(X_test_scaled, y_test)

    # Балансировка выборки с помощью весов
    bins = np.digitize(y_train, bins=[0.33, 0.66])
    class_counts = np.array([len(np.where(bins == t)[0]) for t in np.unique(bins)])
    class_counts = np.maximum(class_counts, 1)
    weight_dict = {t: 1.0 / count for t, count in zip(np.unique(bins), class_counts)}
    samples_weight = np.array([weight_dict[t] for t in bins])

    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = DnDItemRanker(input_size=len(features))

    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 40
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print("\n🚀 Начало обучения...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item() * batch_X.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'dnd_hybrid_weights.pth')
            is_best = "⭐"
        else:
            is_best = ""

        if (epoch + 1) % 5 == 0 or is_best:
            print(
                f"Эпоха [{epoch + 1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} {is_best}")

    model.load_state_dict(torch.load('dnd_hybrid_weights.pth', weights_only=True))

    # ==========================================
    # 4. АНАЛИТИКА
    # ==========================================
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

    # Итоговые метрики (тут MSE/MAE уместны как метрики оценки качества, а не как Loss)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mbe_global = np.mean(y_pred - y_test)

    y_pred_final = np.clip(y_pred, 0.0, 1.0)

    spearman_corr, _ = spearmanr(y_test, y_pred_final)
    k_val = min(50, len(y_test))
    if k_val > 1:
        ndcg_k = ndcg_score(y_test.reshape(1, -1), y_pred_final.reshape(1, -1), k=k_val)
        top_k_indices = np.argsort(y_pred_final)[::-1][:k_val]
        real_scores_in_top = y_test[top_k_indices]
        precision_k = np.sum(real_scores_in_top >= 0.70) / k_val
    else:
        ndcg_k, precision_k = 0.0, 0.0

    print("\n" + "=" * 50)
    print(" 📊 МЕТРИКИ ДЛЯ ОТЧЕТА (TEST SET) ")
    print("=" * 50)
    print("--- Технические метрики оценки ---")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | MBE (Global Bias): {mbe_global:.4f}")
    print("\n--- Бизнес-метрики (Ранжирование) ---")
    print(f"Spearman Corr: {spearman_corr:.4f} (ближе к 1.0 = идеальная сортировка)")
    print(f"NDCG@{k_val}:      {ndcg_k:.4f} (качество Топ-{k_val})")
    print(f"Precision@{k_val}: {precision_k:.1%} (доля хитов с Y >= 0.70 в Топ-{k_val})")
    print("=" * 50)

    os.makedirs("model_report_plots", exist_ok=True)

    # 1. Кривая обучения (Smooth L1 Loss)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (Smooth L1)', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss (Smooth L1)', color='orange', linestyle='--')
    plt.title('Кривая обучения (Train vs Validation)', fontsize=14, fontweight='bold')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss (Smooth L1)')
    plt.legend()
    plt.savefig('model_report_plots/1_learning_curve.png', dpi=300)
    plt.close()

    # 2. Предсказания vs Реальность
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5, color='green', s=15)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    plt.title('Предсказания vs Реальность (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Реальный Target Y (LLM Score)')
    plt.ylabel('Предсказание Модели')
    plt.savefig('model_report_plots/2_predictions.png', dpi=300)
    plt.close()

    # 3. Распределение
    plt.figure()
    sns.histplot(y_pred, bins=50, kde=True, color='purple', stat="density", label='Предсказания')
    sns.histplot(y_test, bins=50, kde=True, color='orange', stat="density", alpha=0.4, label='Реальность')
    plt.title('Распределение предсказаний на Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Скор (Relevance)')
    plt.legend()
    plt.savefig('model_report_plots/3_distribution.png', dpi=300)
    plt.close()

    # 4. График остатков и Смещения (Bias)
    residuals = y_pred - y_test
    plt.figure()
    sns.histplot(residuals, bins=50, kde=True, color='teal', stat="density")
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Идеал (Ошибка = 0)')
    plt.axvline(x=mbe_global, color='blue', linestyle='-', linewidth=2, label=f'Смещение (Bias): {mbe_global:.4f}')
    plt.title('Распределение остатков (Анализ систематического смещения)', fontsize=14, fontweight='bold')
    plt.xlabel('Величина ошибки (Prediction - True)')
    plt.ylabel('Плотность (Density)')
    plt.legend()
    plt.savefig('model_report_plots/4_residuals_bias.png', dpi=300)
    plt.close()

    print("✅ Графики аналитики сохранены в папку 'model_report_plots'.")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" 🧠 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ")
    print("=" * 50)
    train_and_evaluate()