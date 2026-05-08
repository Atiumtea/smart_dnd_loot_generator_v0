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

# Настройки для красивых графиков
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
            nn.Dropout(0.1),  # Защита от переобучения
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
# 2. ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ
# ==========================================
def train_and_evaluate():
    print("📦 Загрузка данных...")
    df = pd.read_csv('dnd_mlp_training_data.csv', sep=';')

    features = ['loc_score', 'party_score', 'story_importance', 'level_rarity_delta', 'is_duplicate', 'type_id',
                'synergy_flag']
    X = df[features].values
    y = df['target_y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    train_loader = DataLoader(DnDDataset(X_train_scaled, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(DnDDataset(X_test_scaled, y_test), batch_size=128, shuffle=False)

    model = DnDItemRanker(input_size=7)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 40
    train_losses, test_losses = [], []

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

        # Валидация
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

    torch.save(model.state_dict(), 'dnd_ranker_weights.pth')

    # ==========================================
    # 3. ПОЛНАЯ АНАЛИТИКА ДЛЯ ОТЧЕТА
    # ==========================================
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print(" 📊 МЕТРИКИ ДЛЯ ОТЧЕТА ")
    print("=" * 40)
    print(f"1. MSE (Среднеквадратичная ошибка): {mse:.4f}")
    print(f"2. MAE (Средняя абсолютная ошибка):  {mae:.4f}")
    print(f"3. R² (Коэффициент детерминации):  {r2:.4f}")
    print("=" * 40)

    # Создаем папку для сохранения графиков
    os.makedirs("report_plots", exist_ok=True)

    # ГРАФИК 1: Кривая обучения (Learning Curve)
    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (Обучающая)', color='blue', linewidth=2)
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss (Тестовая)', color='red', linestyle='--', linewidth=2)
    plt.title('Кривая обучения (Loss Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Эпохи', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_plots/1_learning_curve.png', dpi=300)
    plt.close()

    # ГРАФИК 2: Предсказания vs Реальность (Scatter Plot)
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.3, color='purple', s=10)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=2, label='Идеальное совпадение')
    plt.title('Предсказанные значения vs Реальные (Оракул)', fontsize=14, fontweight='bold')
    plt.xlabel('Реальный Target Y (Синтетика)', fontsize=12)
    plt.ylabel('Предсказание Модели (Predicted Y)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_plots/2_predictions_scatter.png', dpi=300)
    plt.close()

    # ГРАФИК 3: Распределение предсказаний (Histogram)
    plt.figure()
    sns.histplot(y_pred, bins=50, kde=True, color='teal', stat="density", label='Предсказания')
    sns.histplot(y_test, bins=50, kde=True, color='orange', stat="density", alpha=0.4, label='Реальные')
    plt.title('Распределение вероятностей (Генератор Лута)', fontsize=14, fontweight='bold')
    plt.xlabel('Скор (Probability)', fontsize=12)
    plt.ylabel('Плотность', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_plots/3_distribution.png', dpi=300)
    plt.close()

    print("\n✅ Тестирование завершено! 3 графика сохранены в папке 'report_plots'.")


if __name__ == "__main__":
    train_and_evaluate()