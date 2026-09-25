# CertificateGuard AI Frontend

This React/Vite frontend matches the FastAPI backend endpoint:
`POST /api/predict`

## Development

```powershell
$env:TEMP="D:\npm-temp"
$env:TMP="D:\npm-temp"
npm --cache "D:\npm-cache" install
npm run dev
```

## One-URL production mode

Build the frontend:

```powershell
npm run build
```

The build will be created in `frontend/dist`. Configure FastAPI to serve that folder at `/` after the API routes. Then the whole project is available from:

`http://127.0.0.1:8000`
