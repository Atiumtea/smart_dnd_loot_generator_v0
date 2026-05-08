import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ==========================================
# 1. АРХИТЕКТУРА И ДАТАСЕТ
# ==========================================
class DnDItemRanker(nn.Module):
    def __init__(self, input_size=7):
        super(DnDItemRanker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


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
    print("📦 Загрузка синтетической базы...")
    synth_df = pd.read_csv('dnd_mlp_training_data.csv', sep=';')

    try:
        print("🧑‍🏫 Поиск ручной разметки Мастера...")
        gold_df = pd.read_csv('manual_gold_standard.csv', sep=';')

        if len(gold_df) < 5:
            print("⚠️ В 'manual_gold_standard.csv' слишком мало данных. Учимся только на синтетике.")
            return synth_df

        print(f"✨ Найдено {len(gold_df)} эталонных оценок. Начинаем интеграцию...")

        copies = []
        for _ in range(100):
            noisy_copy = gold_df.copy()
            noise = np.random.normal(0, 0.015, size=len(gold_df))
            noisy_copy['target_y'] = np.clip(noisy_copy['target_y'] + noise, 0.0, 1.0)
            copies.append(noisy_copy)

        gold_repeated = pd.concat(copies, ignore_index=True)

        # Объединяем датасеты
        final_df = pd.concat([synth_df, gold_repeated], ignore_index=True)
        print(f"📊 Итоговый гибридный датасет: {len(final_df)} примеров.")
        return final_df

    except FileNotFoundError:
        print("❌ Файл 'manual_gold_standard.csv' не найден! Учимся только на синтетике.")
        return synth_df


# ==========================================
# 3. ОСНОВНОЙ СКРИПТ ОБУЧЕНИЯ
# ==========================================
def train_and_evaluate():
    df = load_hybrid_data()

    features = ['loc_score', 'party_score', 'story_importance', 'level_rarity_delta', 'is_duplicate', 'type_id',
                'synergy_flag']
    X = df[features].values
    y = df['target_y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Перезаписываем scaler для боевого генератора
    with open('scaler_hybrid.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    train_loader = DataLoader(DnDDataset(X_train_scaled, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(DnDDataset(X_test_scaled, y_test), batch_size=128, shuffle=False)

    model = DnDItemRanker(input_size=7)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 40
    train_losses, test_losses = [], []

    print("\n🚀 Начало гибридного обучения...")
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
            print(f"Эпоха [{epoch + 1}/{epochs}] | Train MSE: {epoch_train_loss:.4f} | Test MSE: {epoch_test_loss:.4f}")

    # Сохраняем гибрид в отдельный файл
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
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', color='red', linestyle='--')
    plt.title('Гибридная Кривая Обучения', fontsize=14, fontweight='bold')
    plt.xlabel('Эпохи')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('hybrid_report_plots/1_hybrid_learning_curve.png', dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3, color='green', s=10)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2)
    plt.title('Гибрид: Предсказания vs Смешанная Реальность', fontsize=14, fontweight='bold')
    plt.xlabel('Реальный Target Y (Синтетика + Человек)')
    plt.ylabel('Предсказание Модели')
    plt.savefig('hybrid_report_plots/2_hybrid_predictions.png', dpi=300)
    plt.close()

    plt.figure()
    sns.histplot(y_pred, bins=50, kde=True, color='purple', stat="density", label='Предсказания Гибрида')
    sns.histplot(y_test, bins=50, kde=True, color='orange', stat="density", alpha=0.4, label='Данные')
    plt.title('Смещение распределения после вмешательства человека', fontsize=14, fontweight='bold')
    plt.xlabel('Скор (Probability)')
    plt.legend()
    plt.savefig('hybrid_report_plots/3_hybrid_distribution.png', dpi=300)
    plt.close()


if __name__ == "__main__":
    train_and_evaluate()