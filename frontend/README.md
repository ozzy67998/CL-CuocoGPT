# Chatbot Frontend (Vite + React)

A lightweight React frontend for the Knowledge-and-Language chatbot.

## Features
- Vite dev server for fast iteration
- Simple chat UI (user/bot bubbles)
- Calls `POST /api/chat` (proxied to your backend during dev)
- Minimal styling with plain CSS

## Getting started

1) Install dependencies:

```bash
cd frontend
npm install
```

2) Run the dev server:

```bash
npm run dev
```

This starts Vite at http://localhost:5173.

## Backend proxy
- Dev server proxies `"/api"` to `VITE_BACKEND_URL` (defaults to `http://localhost:8000`).
- Create a `.env` file if you need a different URL:

```
VITE_BACKEND_URL=http://127.0.0.1:8000
```

## Expected API contract
The UI sends:

```http
POST /api/chat
Content-Type: application/json

{ "text": "user message" }
```

Respond with:

```json
{ "reply": "bot message" }
```

Any additional fields are ignored by the UI for now.

## Build for production

```bash
npm run build
npm run preview # optional local preview of the built app
```

## Customize
- Edit `src/App.jsx` for UI logic
- Edit `src/index.css` for styles
- Update proxy target in `vite.config.js` if needed
