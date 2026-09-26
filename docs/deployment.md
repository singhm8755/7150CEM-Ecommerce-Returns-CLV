# Deployment

Three ways to run this, in increasing order of effort.

---

## Streamlit Community Cloud (free, gives you a public link)

The dashboard is the most demo-able part of the project, and Community Cloud
hosts it free from this repository.

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **New app** → pick `singhm8755/7150CEM-Ecommerce-Returns-CLV`.
3. Set:
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **Python version**: 3.11
4. Deploy. The first build takes a few minutes while the dependencies install.

The app works immediately because the trained bundle
(`models/return_risk_model.joblib`) and the scored customer table
(`outputs/reports/customer_clv.csv`) are committed — see the note in
`.gitignore`. Nothing is trained at startup.

**Resource note.** Community Cloud allocates about 1 GB of RAM. The dashboard
loads a 1.5 MB model and a 2.1 MB table, so it fits comfortably. The SHAP
explanation on the scoring page is the heaviest operation and runs in well under
a second.

**Redeploying after a retrain.** Run `make all`, commit the regenerated
`models/return_risk_model.joblib` and `outputs/`, and push — Community Cloud
redeploys on every push to the tracked branch.

---

## The API in Docker

The image carries the package and the API but no model, so the same image works
across retrains:

```bash
make docker-build
make docker-run     # mounts ./models read-only on port 8000
```

Then `http://localhost:8000/docs` for the interactive OpenAPI page.

Run it without a model and the service still starts: `/health` reports
`degraded` and explains what is missing, rather than crash-looping. That is
deliberate — a container that dies on startup tells an orchestrator nothing
useful.

Configuration is found via `$RETURNS_CLV_CONFIG`, then the working directory,
then the source tree. In the image `configs/` sits next to the working
directory, so the default works with no environment variables set.

To point the container at a model somewhere else:

```bash
docker run --rm -p 8000:8000 \
  -v /path/to/models:/models:ro \
  -e RETURNS_CLV_MODEL_PATH=/models \
  returns-clv:latest
```

---

## Locally, from source

```bash
make setup      # virtualenv and an editable install
make all        # the full pipeline, about seven minutes on four cores
make api        # http://localhost:8000/docs
make dashboard  # http://localhost:8501
```

`make help` lists every target.

---

## What is not set up here

This is a portfolio project, not a production service. Deploying it for real
would need, at minimum: a model registry rather than a committed artefact,
scheduled retraining triggered by drift in the monitored return rate, request
and prediction logging, an authentication layer on the API, and an alert when
the live return rate diverges from the training-period rate. The model card
lists the retraining triggers.
