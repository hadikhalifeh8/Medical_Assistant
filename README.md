(Deployment)
----------

- **Render start command**: Ensure Render runs the FastAPI app from the `server` package. A `Procfile` is included with:

```
web: uvicorn server.main:app --host 0.0.0.0 --port $PORT
```

- **/ask/**: The API accepts `POST` (form-encoded `question`) for client use. A convenience `GET` endpoint is also available for quick browser testing (use `?question=...`).

