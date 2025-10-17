# bank-investment-prediction
https://official.contest.yandex.ru/contest/79548/problems/B/
# B. BASE: Найди инвестора (Сбер)

Предсказание отклика клиентов банка на инвестиционное предложение.

## Установка
```bash
pip install -r requirements.txt
```

## Использование

1. Обучение модели:
```bash
python train.py
```

2. Предсказание:
```bash
python predict.py
```

## Структура проекта

- `train.py` - обучение модели
- `predict.py` - предсказание на тесте
- `requirements.txt` - зависимости

## Метрики

- F1 Score: ~0.85
- ROC AUC: ~0.90
