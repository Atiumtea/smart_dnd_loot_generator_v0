import pandas as pd
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

        df = df.fillna(0.0)
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
    continuous_features = ['loc_score', 'party_score', 'story_importance', 'level_rarity_delta']
    binary_features = ['is_duplicate', 'synergy_flag'] + type_features
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

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = DnDItemRanker(input_size=15)
    criterion = nn.MSELoss()
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
                f"Эпоха [{epoch + 1}/{epochs}] | Train MSE: {epoch_train_loss:.4f} | Val MSE: {epoch_val_loss:.4f} {is_best}")

    model.load_state_dict(torch.load('dnd_hybrid_weights.pth', weights_only=True))

    # ==========================================
    # 4. АНАЛИТИКА И СОХРАНЕНИЕ LUT
    # ==========================================
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mbe_global = np.mean(y_pred - y_test)

    # --- Детализированный анализ локального смещения (Шаг 0.05) ---
    df_analysis = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'residual': y_pred - y_test
    })

    bins = np.arange(0.0, 1.05, 0.05)
    df_analysis['score_bucket'] = pd.cut(df_analysis['y_pred'], bins=bins, include_lowest=True)
    bias_by_bucket = df_analysis.groupby('score_bucket', observed=False)['residual'].mean()
    bias_by_bucket = bias_by_bucket.ffill().bfill()

    # Вывод данных для отчета (как ты просил)
    print("\n" + "=" * 50)
    print(" 🔍 ЛОКАЛЬНОЕ СМЕЩЕНИЕ (BIAS ПО СЕГМЕНТАМ 0.05) ")
    print("=" * 50)
    for bucket, local_mbe in bias_by_bucket.items():
        print(f"Диапазон {bucket}: MBE = {local_mbe:+.4f}")

    # Инвертируем смещение (минус на минус) и сохраняем в JSON
    calibration_lut = [-float(mbe) for mbe in bias_by_bucket.values]
    with open('calibration_lut.json', 'w') as f:
        json.dump(calibration_lut, f)
    print(f"\n✅ Таблица калибровки (LUT) успешно сохранена в 'calibration_lut.json' ({len(calibration_lut)} бакетов).")

    # Симуляция работы LUT для графиков
    def apply_calibration(val):
        idx = int(val / 0.05)
        idx = min(max(0, idx), len(calibration_lut) - 1)
        return val + calibration_lut[idx]

    y_pred_calibrated = np.array([apply_calibration(v) for v in y_pred])
    y_pred_calibrated = np.clip(y_pred_calibrated, 0.0, 1.0)  # Ограничиваем от 0 до 1

    # --- 2. Бизнес-метрики (Ранжирование) ---
    spearman_corr, _ = spearmanr(y_test, y_pred)
    k_val = min(50, len(y_test))
    if k_val > 1:
        ndcg_k = ndcg_score(y_test.reshape(1, -1), y_pred.reshape(1, -1), k=k_val)
        top_k_indices = np.argsort(y_pred)[::-1][:k_val]
        real_scores_in_top = y_test[top_k_indices]
        precision_k = np.sum(real_scores_in_top >= 0.70) / k_val
    else:
        ndcg_k, precision_k = 0.0, 0.0

    print("\n" + "=" * 50)
    print(" 📊 МЕТРИКИ ДЛЯ ОТЧЕТА (TEST SET) ")
    print("=" * 50)
    print("--- Технические метрики ---")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | MBE (Global Bias): {mbe_global:.4f}")
    print("\n--- Бизнес-метрики (Ранжирование) ---")
    print(f"Spearman Corr: {spearman_corr:.4f} (ближе к 1.0 = идеальная сортировка)")
    print(f"NDCG@{k_val}:      {ndcg_k:.4f} (качество Топ-{k_val})")
    print(f"Precision@{k_val}: {precision_k:.1%} (доля хитов с Y >= 0.70 в Топ-{k_val})")
    print("=" * 50)

    os.makedirs("model_report_plots", exist_ok=True)

    # 1. Кривая обучения
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (MSE)', color='blue')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss (MSE)', color='orange', linestyle='--')
    plt.title('Кривая обучения (Train vs Validation)', fontsize=14, fontweight='bold')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('model_report_plots/1_learning_curve.png', dpi=300)
    plt.close()

    # 2. Предсказания vs Реальность (Сырые)
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
    sns.histplot(y_pred, bins=50, kde=True, color='purple', stat="density", label='Предсказания (до LUT)')
    sns.histplot(y_test, bins=50, kde=True, color='orange', stat="density", alpha=0.4, label='Реальность')
    plt.title('Распределение предсказаний на Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Скор (Relevance)')
    plt.legend()
    plt.savefig('model_report_plots/3_distribution.png', dpi=300)
    plt.close()

    # 4. График остатков и Байоса (Residuals & Bias)
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

    # 5. НОВЫЙ: Сравнение предсказаний ДО и ПОСЛЕ калибровки (Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Левый график (ДО)
    axes[0].scatter(y_test, y_pred, alpha=0.4, color='green', s=15)
    axes[0].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    axes[0].set_title('ДО калибровки (Сырые предсказания)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Реальный Target Y')
    axes[0].set_ylabel('Предсказание Модели')

    # Правый график (ПОСЛЕ)
    axes[1].scatter(y_test, y_pred_calibrated, alpha=0.4, color='blue', s=15)
    axes[1].plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    axes[1].set_title('ПОСЛЕ LUT-калибровки', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Реальный Target Y')

    plt.suptitle('Эффект применения табличной калибровки (LUT) на тестовой выборке', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_report_plots/5_calibration_effect.png', dpi=300)
    plt.close()

    print("✅ Графики аналитики сохранены в папку 'model_report_plots'.")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" 🧠 ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ ")
    print("=" * 50)
    train_and_evaluate()