# Deploy on Render

## 1) Push project to GitHub

1. Create a GitHub repository.
2. Push this project branch to GitHub.

## 2) Create Render web service

### Option A: Blueprint (recommended)

1. In Render dashboard, click **New +** -> **Blueprint**.
2. Select your GitHub repository.
3. Render will detect `render.yaml`.
4. Click **Apply**.

### Option B: Manual web service

1. In Render dashboard, click **New +** -> **Web Service**.
2. Select your GitHub repository.
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 90 --max-requests 1000 --max-requests-jitter 120 wsgi:app`
   - **Health Check Path**: `/health`
4. Add env var:
   - `FLASK_SECRET_KEY` -> any random long string

## 3) Verify deployment

After deploy finishes:

1. Open `https://<your-service>.onrender.com/health` and verify `{"status":"ok"}`.
2. Open main page and test:
   - DOI search
   - URL search
   - Title search with typo
   - Export BibTeX / RIS

## Notes for conference demo

- Render free instances can sleep after inactivity. Open `/health` a few minutes before demo.
- SQLite data is stored on ephemeral disk in free plan. For persistent shared storage, move selected papers to managed Postgres.
- Keep fallback local demo: `python app.py`.

