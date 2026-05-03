# BiblioParser

Веб-сервис извлечения метаданных статей (URL / DOI / название), экспорт BibTeX/RIS.

Материалы к защите (видео, локальный exe, сравнение с ручной работой, метрики полей): **`docs/ZASHCHITA_PREPOD.md`**.

## Запуск локально

```bash
pip install -r requirements.txt
python app.py
```

Открыть `http://127.0.0.1:5000`. Параметры запроса: `?doi=`, `?url=`, `?title=` подставляются в форму.

---

## Опциональный этап (по порядку)

### 1) Docker + Redis

Требуется [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS).

Из каталога проекта:

```bash
docker compose up --build
```

Сайт: **http://127.0.0.1:5000**

**Телеграм-бот** в том же репозитории (отдельный контейнер). Скопируйте `.env.example` в `.env`, задайте `TELEGRAM_BOT_TOKEN` и при необходимости `SP_PUBLIC_URL` (на проде — `https://ваш-домен`). Запуск:

```bash
docker compose --profile bot up --build -d
```

В `docker-compose.yml` уже задано:

| Переменная | Назначение |
|------------|------------|
| `REDIS_URL` | Redis DB **0** → кэш Flask-Caching (`/api/work` и др.) |
| `RATELIMIT_STORAGE_URI` | Redis DB **1** → счётчики лимитера по IP (не путать с `..._URL`) |

Папка `./data` на хосте монтируется в контейнер как `/app/data` (SQLite `app_state.db`).

**VPS с HTTPS (Caddy + Let’s Encrypt):** файл `docker-compose.stack.yml` и `deploy/Caddyfile`. Переменные в `.env`: `SP_DOMAIN`, `FLASK_SECRET_KEY`, для бота — `TELEGRAM_BOT_TOKEN`, `SP_PUBLIC_URL=https://…`. Команда:

```bash
docker compose -f docker-compose.stack.yml --profile bot up --build -d
```

Альтернатива без Caddy в Docker — nginx на хосте, см. `deploy/nginx-snippet.conf`.

Остановка: `Ctrl+C` или `docker compose down`.

---

### 2) Обучение модели блоков

Нужен файл разметки `data/annotated_dataset.json`.

```bash
python train_evaluate.py
```

Результаты: `models/block_classifier.joblib`, отчёты в каталоге `reports/` (если скрипт их пишет).

---

### 3) SBERT при обучении

Устанавливаются тяжёлые зависимости (PyTorch + sentence-transformers):

```bash
pip install -r requirements-ml.txt
```

PowerShell:

```powershell
$env:USE_SBERT="1"
python train_evaluate.py
```

Без GPU обучение дольше; при ошибке импорта проверь, что виртуальное окружение то же, что для проекта.

---

### 4) Telegram-бот

Локально (второе окно) или через Docker (`docker compose --profile bot up -d`).

PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN="ВАШ_ТОКЕН_ОТ_BOTFATHER"
$env:SP_PUBLIC_URL="http://127.0.0.1:5000"
python telegram_bot.py
```

На сервере задайте `SP_PUBLIC_URL=https://ваш-домен` (тот же URL, что открываете в браузере).

Нужен **интернет до api.telegram.org** (ошибка WinError 10051 — сеть/VPN/фаервол).

### 4b) Тонкий клиент (.exe без тяжёлых библиотек)

Каталог `desktop_client`: окно на **tkinter**, запросы к **`GET /api/work`** на вашем сервере (`doi`, `url` или `title`). Сборка:

```powershell
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File desktop_client/build_windows.ps1
```

Получится `dist/ScientificParserThin.exe`. В поле URL укажите продакшен `https://…`.

---

### 5) Расширение Chrome

1. Открой `chrome://extensions`
2. Режим разработчика → **Загрузить распакованное расширение**
3. Укажи папку **`extensions/chrome`**

По умолчанию расширение шьётся к **http://127.0.0.1:5000** — сервер должен быть запущен.

---

## Тесты

```bash
pytest
```

## Прочее

- JSON API: `GET /api/work?doi=...` или `?url=...` или `?title=...` (ровно один параметр)
- Локально без Docker Redis опционален: лимиты и кэш работают в памяти процесса.
