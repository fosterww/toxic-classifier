# Toxicity Classifier API

Простой сервис для классификации токсичности комментариев.
Стек: **Python 3.12.3**, **scikit-learn**, **FastAPI**, **Docker**, (опционально **PostgreSQL**).

## 1. Overview

Задача: бинарная классификация комментариев на классы:

- `clean` — не токсичный
- `toxic` — токсичный

Сервис предоставляет REST API для онлайн-проверки комментариев и поддерживает цикл "feedback → retrain".

Основные цели проекта:

- показать end-to-end пайплайн: данные → модель → API → Docker → retrain;
- сделать код понятным для джуна, но с аккуратными инженерными решениями.

---

## 2. Features

-  Модель: `TfidfVectorizer + LogisticRegression(class_weight="balanced")`
-  Метрики: macro-F1, ROC-AUC, confusion matrix
-  FastAPI:
  - `GET /health` — статус и версия модели
  - `POST /predict` — предсказание токсичности
  - `POST /feedback` — сбор ручной разметки (опционально)
-  Docker: готовый образ для запуска API
-  Feedback loop: `data/feedback.csv` + `scripts/retrain.py`
-  Тесты: pytest для модели и API

---

## 3. Data

Используемые источники:

- EN: [Jigsaw Toxic Comment Classification Challenge] (подмножество)
- Доп. данные (RU/UA): `data/raw/extra_ru_ua.csv` — 200–500 строк ручной разметки

После подготовки (`scripts/prepare_data.py`) все данные приводятся к схеме:

```text
text,label
"comment text here",0 or 1
