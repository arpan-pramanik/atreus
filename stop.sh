#!/usr/bin/env bash
# One-click script to stop both Python backend and Next.js frontend
cd "$(dirname "$0")"
echo "Stopping Atreus Streaming Platform..."
npm run stop
