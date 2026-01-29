# Frontend (Vite + React)

Minimal UI to upload/drag-drop a PDF, `POST /api/process` as `multipart/form-data`, and render the returned `graphHtml` inside a responsive iframe.

## Run (dev)

```bash
cd frontend
npm install
npm run dev
```

### Backend on a different port?

If your backend is running elsewhere, set a dev proxy target:

```bash
cd frontend
VITE_API_PROXY_TARGET=http://localhost:8000 npm run dev
```

Then `fetch("/api/process")` will be proxied to `http://localhost:8000/api/process` during dev.

