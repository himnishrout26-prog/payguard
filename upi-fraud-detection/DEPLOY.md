# Deploying PayGuard's API

Two free options. Render is generally the more reliable free tier as of
2025-2026; Railway's free tier requires a card on file but has a smoother
CLI. Steps for both below — pick one.

Either way, you need artifacts/ populated FIRST (run `python train.py` or
`python train_compare.py` locally, commit the resulting `artifacts/*.joblib`
and `.json` files to your repo — they're small, a few MB at most).

## Option A: Render

1. Push this repo to GitHub (public or private, Render supports both).
2. Go to https://render.com → New → Web Service → connect your repo.
3. Render auto-detects the `Dockerfile` at the repo root. If it asks:
   - **Environment**: Docker
   - **Region**: closest to you
   - **Instance type**: Free
4. Render builds the image and deploys automatically on every push to `main`.
5. Your API is live at `https://<your-service-name>.onrender.com`. Test it:
   ```bash
   curl https://<your-service-name>.onrender.com/health
   ```

**Free-tier caveat, be upfront about this**: Render's free web services spin
down after 15 minutes of no traffic and take ~30-60s to wake up on the next
request. That's fine for a resume link (say so in your README: "may take a
few seconds to wake up on first request") — don't let a recruiter think
your API is just broken if they hit it cold.

## Option B: Railway

1. Install the CLI: `npm install -g @railway/cli` (or use the web dashboard)
2. From the project root:
   ```bash
   railway login
   railway init
   railway up
   ```
3. Railway also auto-detects the Dockerfile. Once deployed:
   ```bash
   railway domain   # generates a public URL
   ```
4. Set the port explicitly if needed — Railway injects `$PORT`, so update
   the Dockerfile's CMD to respect it:
   ```dockerfile
   CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
   ```
   (swap this in for the CMD line in Dockerfile if you go with Railway)

## After deploying: update your README and resume

Add the live URL:
```markdown
**Live API**: https://your-service.onrender.com/docs (interactive Swagger UI,
FastAPI generates this automatically -- good for a recruiter to poke at
without writing curl commands)
```

FastAPI auto-generates a Swagger UI at `/docs` on any deployment — that's
genuinely the best link to put on a resume, since it lets someone try the
API in their browser with zero setup.

## Sanity-check before you share the link

```bash
curl -X POST https://your-service.onrender.com/score \
  -H "Content-Type: application/json" \
  -d '{"step":14,"type":"TRANSFER","amount":45230.0,"nameOrig":"C123",
       "oldbalanceOrg":45230.0,"newbalanceOrig":0.0,"nameDest":"C456",
       "oldbalanceDest":120.0,"newbalanceDest":36304.0}'
```
Should return a JSON risk score. If it 500s, check the deployment logs —
the most common cause is `artifacts/` not being committed to the repo (the
Dockerfile copies it in, but if it's empty/missing in git, the image builds
fine and then crashes at runtime on `joblib.load`).
