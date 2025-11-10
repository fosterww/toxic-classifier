#  Toxicity Classifier API

Простой, но production-style сервис для классификации токсичности комментариев.
Полный цикл: **данные → модель → API → Docker → feedback loop → retrain.**

---

##  1. Overview

Проект классифицирует текстовые комментарии на 2 класса:

- `clean` — не токсичный
- `toxic` — токсичный

Сервис предоставляет REST API на FastAPI и поддерживает переобучение модели на новых примерах (feedback loop).

**Цель:** показать, как сделать end-to-end ML-сервис уровня pet-production.

---

## 2. Features

-  **Модель:** Logistic Regression + TF-IDF
-  **Метрики:** macro-F1, ROC-AUC, Confusion matrix
-  **FastAPI endpoints:**
  - `GET /health` — статус сервиса
  - `POST /predict` — классификация текста
  - `POST /feedback` — сохранение размеченных примеров
-  **Feedback loop:** retrain модели на собранных примерах
-  **Docker** — готовый образ для деплоя
-  **Тесты:** pytest для модели и API

---

## 3. Данные

Источник: **AI Generated texts**

Структура:

```text
text,label
"you are amazing",0
"you are stupid",1
```

Сплит: **70% train / 15% val / 15% test**
Хранятся в `data/processed/`.

---

## 4. Модель и метрики

Baseline-pipeline:

```python
Pipeline([
  ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=100_000)),
  ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=-1))
])
```

Файлы модели:

```
models/
 ├─ model_20251105_2008.joblib
 └─ metadata.json
```

---

## 5. Структура проекта

```text
toxic-classifier/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
│  ├─ model_*.joblib
│  └─ metadata.json
├─ notebooks/
│  ├─ 01_eda_baseline.ipynb
│  └─ confusion_test.png
├─ app/
│  ├─ main.py
│  ├─ predict.py
│  ├─ schemas.py
│  └─ utils.py
├─ scripts/
│  ├─ prepare_data.py
│  ├─ train.py
│  ├─ eval.py
├─ tests/
│  ├─ test_model_load.py
│  ├─ test_api_integration.py
│  └─ test_utils.py
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
└─ README.md
```

---

## 6. Запуск локально

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# обучение модели
python scripts/train.py

# запуск API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict      -H "Content-Type: application/json"      -d '{"text": "you are awesome"}'
```

---

## 7. Запуск через Docker

###  Один контейнер

```bash
docker build -t toxicity-api:latest .
docker run --rm -p 8000:8000 toxicity-api:latest
```

Параметры окружения:

| Переменная | Описание | Default |
|-------------|-----------|----------|
| `THRESHOLD` | Порог токсичности | 0.6 |
| `LOW_CONF_FLOOR` | Минимум для confident предсказаний | 0.65 |
| `SHORT_LEN` | Минимальная длина текста | 8 |
| `LOG_LEVEL` | Уровень логов | INFO |

---

###  Docker-Compose (с PostgreSQL)

`docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: toxic
      POSTGRES_PASSWORD: toxic
      POSTGRES_DB: toxicdb
    ports:
      - "5433:5432"

  api:
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      - db
```

`.env`:

```env
DATABASE_URL=postgresql+psycopg2://toxic:toxic@db:5432/toxicdb
THRESHOLD=0.6
LOW_CONF_FLOOR=0.65
SHORT_LEN=8
LOG_LEVEL=INFO
```

Запуск:

```bash
docker-compose up --build
```

---

## 8. API Reference

### `GET /health`

```json
{"status":"ok","model_version":"20251105_2008"}
```

### `POST /predict`

**Request:**
```json
{"text": "you are awful"}
```

**Response:**
```json
{"label":"toxic","prob":0.93,"low_confidence":false}
```

**Policy:**
- toxic ↔ prob ≥ THRESHOLD
- `low_confidence = true` если:
  - `prob < max(THRESHOLD, LOW_CONF_FLOOR)`
  - или `len(clean_text) < SHORT_LEN`

### `POST /feedback`

```json
{"text": "you are awful", "true_label": 1}
```
Сохраняет запись в `data/feedback.csv`.

---

## 9. Feedback Loop

1. API принимает фидбек → `data/feedback.csv`
3. Новая модель сохраняется → `models/model_YYYYMMDD_HHMM.joblib`
4. `metadata.json` обновляется автоматически.

---

Автор: fosterww
