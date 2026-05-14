import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from models import DnDItemRanker, ITEM_TYPES

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# ⚙️ НАСТРОЙКИ ОБУЧЕНИЯ
# ==========================================
USE_SYNTHETIC = True

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
# 2. ПОДГОТОВКА ГИБРИДНЫХ ДАННЫХ
# ==========================================
def load_hybrid_data():
    if USE_SYNTHETIC:
        print("📦 Загрузка синтетической базы (ВКЛЮЧЕНА)...")
        try:
            synth_df = pd.read_csv('dnd_mlp_training_data.csv', sep=';')
            synth_df['is_manual'] = 0
        except FileNotFoundError:
            print("⚠️ Синтетика не найдена! Учимся только на ручной разметке.")
            synth_df = pd.DataFrame()
    else:
        print("🚫 Синтетическая база ОТКЛЮЧЕНА. Режим чистого ML.")
        synth_df = pd.DataFrame()

    try:
        print("🧑‍🏫 Поиск ручной разметки Мастера...")
        gold_df = pd.read_csv('manual_gold_standard.csv', sep=';')

        if len(gold_df) < 5 and not USE_SYNTHETIC:
            print("❌ Слишком мало ручных данных для отключения синтетики! Аварийное завершение.")
            exit()

        print(f"✨ Найдено {len(gold_df)} эталонных оценок.")
        gold_df['is_manual'] = 1

        final_df = pd.concat([synth_df, gold_df], ignore_index=True)
        final_df = final_df.fillna(0.0)

        print(f"📊 Итоговый объем датасета: {len(final_df)} примеров.")
        return final_df

    except FileNotFoundError:
        if not USE_SYNTHETIC:
            print("❌ Ручной датасет не найден, а синтетика отключена. Не на чем учиться!")
            exit()
        return synth_df

# ==========================================
# 3. ОСНОВНОЙ СКРИПТ ОБУЧЕНИЯ
# ==========================================
def train_and_evaluate():
    df = load_hybrid_data()

    base_features = ['loc_score', 'party_score', 'story_importance', 'level_rarity_delta', 'is_duplicate', 'synergy_flag']
    type_features = [f'type_{t.replace(" ", "_")}' for t in ITEM_TYPES]
    features = base_features + type_features

    for col in features + ['target_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    X = df[features].values.astype(np.float32)
    y = df['target_y'].values.astype(np.float32)
    is_manual = df['is_manual'].values.astype(int)

    X_train, X_test, y_train, y_test, is_man_train, _ = train_test_split(
        X, y, is_manual, test_size=0.15, random_state=42, stratify=is_manual
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open('scaler_hybrid.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # === ИСПРАВЛЕНИЕ: ПРОФЕССИОНАЛЬНЫЙ СЭМПЛИНГ ===
    # Используем WeightedRandomSampler для балансировки синтетики и ручной разметки.
    # Ручные примеры имеют вес в 50 раз больше, поэтому сеть будет учиться на них активнее.
    sample_weights = np.where(is_man_train == 1, 50.0, 1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = DnDDataset(X_train_scaled, y_train)
    # Передаем sampler вместо shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
    test_loader = DataLoader(DnDDataset(X_test_scaled, y_test), batch_size=128, shuffle=False)

    model = DnDItemRanker(input_size=15)

    # === ИСПРАВЛЕНИЕ: МАТЕМАТИЧЕСКИ КОРРЕКТНЫЙ LOSS ===
    # Для задачи регрессии (предсказание числа от 0.0 до 1.0) используем MSELoss.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 40
    train_losses, test_losses = [], []

    print("\n🚀 Начало гибридного обучения (Knowledge Distillation)...")
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
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_test_loss += loss.item() * batch_X.size(0)

        epoch_test_loss /= len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Эпоха [{epoch + 1}/{epochs}] | Train Loss (MSE): {epoch_train_loss:.4f} | Test Loss (MSE): {epoch_test_loss:.4f}")

    torch.save(model.state_dict(), 'dnd_hybrid_weights.pth')

    # ==========================================
    # 4. АНАЛИТИКА
    # ==========================================
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(" 📊 МЕТРИКИ ГИБРИДА ДЛЯ ОТЧЕТА ")
    print("=" * 40)
    print(f"1. MSE: {mse:.4f}")
    print(f"2. MAE: {mae:.4f}")
    print(f"3. R²:  {r2:.4f}")
    print("=" * 40)

    os.makedirs("hybrid_report_plots", exist_ok=True)

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (MSE)', color='blue')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss (MSE)', color='red', linestyle='--')
    plt.title('Кривая обучения (Knowledge Distillation)', fontsize=14, fontweight='bold')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('hybrid_report_plots/1_hybrid_learning_curve.png', dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3, color='green', s=10)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    plt.title('Гибрид: Предсказания vs Смешанная Реальность', fontsize=14, fontweight='bold')
    plt.xlabel('Реальный Target Y')
    plt.ylabel('Предсказание Модели')
    plt.savefig('hybrid_report_plots/2_hybrid_predictions.png', dpi=300)
    plt.close()

    plt.figure()
    sns.histplot(y_pred, bins=50, kde=True, color='purple', stat="density", label='Предсказания Гибрида')
    sns.histplot(y_test, bins=50, kde=True, color='orange', stat="density", alpha=0.4, label='Реальные данные')
    plt.title('Распределение предсказаний vs реальность', fontsize=14, fontweight='bold')
    plt.xlabel('Скор (Probability)')
    plt.legend()
    plt.savefig('hybrid_report_plots/3_hybrid_distribution.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()