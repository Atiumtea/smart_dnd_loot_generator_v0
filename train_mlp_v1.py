import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# ==========================================
# 1. ПОДГОТОВКА ДАННЫХ (DATASET & DATALOADER)
# ==========================================
class DnDDataset(Dataset):
    def __init__(self, X, y):
        # Конвертируем данные в тензоры PyTorch (тип float32 обязателен для весов)
        self.X = torch.tensor(X, dtype=torch.float32)
        # y должен иметь форму (N, 1), а не просто (N,)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. АРХИТЕКТУРА НЕЙРОСЕТИ (MLP)
# ==========================================
class DnDItemRanker(nn.Module):
    def __init__(self, input_size=5):  # Убедись, что тут 5!
        super(DnDItemRanker, self).__init__()

        # Слои должны быть сдвинуты на 8 пробелов от края файла
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Этот метод должен быть на том же уровне, что и __init__
        return self.network(x)

    def forward(self, x):
        return self.network(x)


# ==========================================
# 3. ОСНОВНОЙ СКРИПТ ОБУЧЕНИЯ
# ==========================================
def train_model():
    print("Загрузка данных...")
    # Укажи тут свой разделитель, если сохранял с sep=';'
    df = pd.read_csv('dnd_mlp_training_data.csv', sep=';')

    # Разделяем на фичи (X) и целевую переменную (y)
    # Проверь, что список колонок совпадает с твоим новым CSV
    X = df[['location_score', 'party_score', 'story_importance', 'level_rarity_delta', 'is_duplicate']].values
    y = df['target_y'].values

    # Разделяем на обучающую (80%) и тестовую (20%) выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # МАСШТАБИРОВАНИЕ (CRITICAL!)
    # Нейросети плохо работают, если одни фичи от 0 до 1, а другие от -4 до 4.
    # StandardScaler приведет всё к нормальному распределению (среднее 0, дисперсия 1).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Сохраняем Scaler (он понадобится при боевом поиске для нормализации новых запросов!)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Создаем датасеты и загрузчики батчей (пакетов по 64 примера)
    train_dataset = DnDDataset(X_train_scaled, y_train)
    test_dataset = DnDDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Инициализация модели, функции потерь и оптимизатора
    model = DnDItemRanker(input_size=5)
    # MSELoss (Mean Squared Error) отлично подходит для предсказания конкретного скора
    criterion = nn.MSELoss()
    # Adam - лучший универсальный оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    epochs = 30  # Количество полных проходов по датасету

    print("\nНачало обучения...")
    for epoch in range(epochs):
        model.train()  # Переводим модель в режим обучения
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            # 1. Обнуляем градиенты с прошлого шага
            optimizer.zero_grad()

            # 2. Прямой проход (Forward pass) - предсказываем вероятности
            predictions = model(batch_X)

            # 3. Считаем ошибку (сравниваем с идеалом оракула)
            loss = criterion(predictions, batch_y)

            # 4. Обратный проход (Backward pass) - считаем градиенты
            loss.backward()

            # 5. Шаг оптимизатора - обновляем веса
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # Валидация (проверка на данных, которые сеть не видела)
        model.eval()  # Режим оценки (отключает dropout/batchnorm, если бы они были)
        test_loss = 0.0
        with torch.no_grad():  # Отключаем расчет градиентов для ускорения
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        test_loss /= len(test_loader.dataset)

        # Выводим прогресс каждые 5 эпох
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss (MSE): {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    print("\nОбучение завершено!")

    # СОХРАНЕНИЕ ВЕСОВ МОДЕЛИ
    torch.save(model.state_dict(), 'dnd_ranker_weights.pth')
    print("Веса модели сохранены в 'dnd_ranker_weights.pth'")


if __name__ == "__main__":
    train_model()
