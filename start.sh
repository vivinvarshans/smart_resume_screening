#!/bin/bash

# Resume Matcher - Quick Start Script
# This script starts both backend and frontend servers

echo "🚀 Starting Resume Matcher..."
echo ""

# Check if we're in the correct directory
if [ ! -d "apps/backend" ] || [ ! -d "apps/frontend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Check if ports are available
if check_port 8000; then
    echo "⚠️  Warning: Port 8000 is already in use. Backend may already be running."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if check_port 5173; then
    echo "⚠️  Warning: Port 5173 is already in use. Frontend may already be running."
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start backend in a new terminal window
echo "📦 Starting backend server on http://localhost:8000..."
osascript -e 'tell application "Terminal" to do script "cd \"'$(pwd)'/apps/backend\" && source venv/bin/activate 2>/dev/null || true && uvicorn app.main:app --reload"' > /dev/null 2>&1 &

sleep 2

# Start frontend in a new terminal window
echo "🎨 Starting frontend server on http://localhost:5173..."
osascript -e 'tell application "Terminal" to do script "cd \"'$(pwd)'/apps/frontend\" && npm run dev"' > /dev/null 2>&1 &

sleep 2

echo ""
echo "✅ Resume Matcher is starting!"
echo ""
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/api/docs"
echo "Frontend: http://localhost:5173"
echo ""
echo "📝 Check the new terminal windows for server logs"
echo ""
echo "To stop the servers:"
echo "  - Press Ctrl+C in each terminal window"
echo "  - Or run: killall -9 uvicorn node"
echo ""
