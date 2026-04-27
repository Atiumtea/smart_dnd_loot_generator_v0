import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


# ==========================================
# 1. АРХИТЕКТУРА (7 входов)
# ==========================================
class DnDItemRanker(nn.Module):
    def __init__(self, input_size=7):
        super(DnDItemRanker, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def train_with_gold_standard():
    print("📦 Загрузка данных...")

    # 1. Загружаем синтетику
    synth_df = pd.read_csv('dnd_mlp_training_data.csv', sep=';')
    # Приводим названия колонок к единому стандарту
    synth_df = synth_df.rename(columns={'location_score': 'loc_score'})

    # 2. Загружаем Золотой Стандарт (твои 200 правок)
    try:
        gold_df = pd.read_csv('manual_gold_standard.csv', sep=';')
        print(f"✨ Найдено {len(gold_df)} экспертных оценок.")

        # --- ТРЮК 1: Jittering (размытие ступенек) ---
        # Оценки 0, 0.25, 0.5... превращаем в 0.49, 0.51 и т.д.
        noise = np.random.normal(0, 0.02, size=len(gold_df))
        gold_df['target_y'] = np.clip(gold_df['target_y'] + noise, 0.0, 1.0)

        # Добавляем колонку дубликатов (пока везде 0), чтобы размерность совпала
        if 'is_duplicate' not in gold_df.columns:
            gold_df['is_duplicate'] = 0

        # --- ТРЮК 2: Oversampling (усиление голоса Мастера) ---
        # Мы хотим, чтобы ручные данные составляли ~30% всего датасета
        # Повторяем твои 200 правок 50 раз
        gold_repeated = pd.concat([gold_df] * 50, ignore_index=True)

        # Объединяем
        final_df = pd.concat([synth_df, gold_repeated], ignore_index=True)
        print(f"📊 Итоговый объем выборки: {len(final_df)} строк.")

    except FileNotFoundError:
        print("⚠️ Файл 'manual_gold_standard.csv' не найден. Учимся только на синтетике.")
        final_df = synth_df

    # Подготовка признаков
    features = [
        'loc_score', 'party_score', 'story_importance',
        'level_rarity_delta', 'is_duplicate', 'type_id', 'synergy_flag'
    ]
    X = final_df[features].values
    y = final_df['target_y'].values

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Сохраняем скейлер (ВАЖНО использовать новый!)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Разбивка
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15)

    # В тензоры
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Обучение
    model = DnDItemRanker(input_size=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n🚀 Начало гибридного обучения...")
    for epoch in range(200):  # Увеличим до 150 эпох, чтобы закрепить "Золото"
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test), y_test)
                print(f"Эпоха [{epoch + 1}/200] | Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    # Сохранение весов
    torch.save(model.state_dict(), 'dnd_ranker_weights.pth')
    print("\n✅ Модель успешно переобучена с учетом твоих правок!")


if __name__ == "__main__":
    train_with_gold_standard()