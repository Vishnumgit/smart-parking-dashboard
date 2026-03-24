# Smart Parking Dashboard 🅿️🚗

A complete smart parking monitoring system with live CCTV analysis, real-time occupancy detection (green/red overlays), motion alerts, and prediction charts.

## 🏗️ Architecture

- **Backend Service** — Python FastAPI + OpenCV for parking space detection
- **Frontend Dashboard** — Next.js 16 + TypeScript + Tailwind CSS 4 + shadcn/ui + Recharts
- **Live URL** — https://smart-parking.codewords.run

## 🔧 Features

- 📹 **Multi-camera support** — 3 demo cameras + add your own CCTV feeds
- 🟢/🔴 **Green/Red overlays** — OpenCV edge detection + pixel variance
- 📊 **Parking stats** — Total/Occupied/Available/Occupancy %
- ⚡ **Motion detection** — Frame differencing with alert log
- 📈 **Occupancy trend chart** — Recharts area chart
- 🌀 **Auto-refresh** — 10s/15s/30s/60s polling intervals
- 🔮 **Prediction panel** — Low/Moderate/High/Near-full demand
- 📱 **Space map grid** — Individual space status visualization

## 📁 Project Structure

```
backend/
└── smart_parking_analyzer.py    # OpenCV parking analyzer (FastAPI service)

frontend/
├── app/page.tsx                  # Main dashboard page
├── app/api/analyze/route.ts     # API route to backend
└── components/                  # shadcn/ui components
```

## 🚀 Getting Started

1. Open [smart-parking.codewords.run](https://smart-parking.codewords.run)
2. Select a camera tab
3. Click "Analyze Now" to run analysis
4. Toggle "Auto" for continuous monitoring
5. Add custom camera URLs via the input at top

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11, FastAPI, OpenCV, NumPy |
| Frontend | Next.js 16, TypeScript, Tailwind CSS 4 |
| Components | shadcn/ui, Framer Motion, Recharts |
| Platform | CodeWords by Agemo |

## 📜 License

MIT
