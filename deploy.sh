#!/bin/bash

# Resume Matcher - Railway Deployment Script
# This script helps you deploy the application to Railway

set -e

echo "🚀 Resume Matcher - Railway Deployment"
echo "======================================"
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed."
    echo ""
    echo "Install it with:"
    echo "  macOS/Linux: curl -fsSL https://railway.app/install.sh | sh"
    echo "  Windows: iwr https://railway.app/install.ps1 | iex"
    echo ""
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Check if logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway..."
    railway login
fi

echo "✅ Logged in to Railway"
echo ""

# Check if project is initialized
if ! railway status &> /dev/null; then
    echo "📦 Initializing Railway project..."
    railway init
    echo ""
fi

echo "✅ Railway project initialized"
echo ""

# Prompt for environment variables
echo "🔧 Setting up environment variables..."
echo ""

read -p "Enter your Groq API Key (from console.groq.com): " GROQ_KEY
if [ -z "$GROQ_KEY" ]; then
    echo "❌ Groq API Key is required!"
    exit 1
fi

read -p "Enter a secret key for sessions (or press Enter for random): " SECRET_KEY
if [ -z "$SECRET_KEY" ]; then
    SECRET_KEY=$(openssl rand -hex 32)
    echo "Generated random secret key"
fi

echo ""
echo "Setting environment variables..."

railway variables set LLM_API_KEY="$GROQ_KEY"
railway variables set LLM_PROVIDER="groq"
railway variables set LL_MODEL="llama-3.3-70b-versatile"
railway variables set LLM_BASE_URL="https://api.groq.com/openai/v1"
railway variables set SESSION_SECRET_KEY="$SECRET_KEY"
railway variables set SYNC_DATABASE_URL="sqlite:///./app.db"
railway variables set ASYNC_DATABASE_URL="sqlite+aiosqlite:///./app.db"
railway variables set PYTHONDONTWRITEBYTECODE="1"
railway variables set EMBEDDING_PROVIDER="none"
railway variables set EMBEDDING_MODEL="none"

echo "✅ Environment variables set"
echo ""

# Deploy
echo "🚀 Deploying to Railway..."
echo ""

railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 View your deployment:"
echo "  Dashboard: railway open"
echo "  Logs: railway logs"
echo ""
echo "🌐 Your app will be available at the URL shown above"
echo ""
echo "📚 Next steps:"
echo "  1. Note your Railway app URL"
echo "  2. Deploy frontend to Vercel with VITE_API_BASE_URL=<your-railway-url>/api/v1"
echo "  3. Update CORS settings in apps/backend/app/core/config.py"
echo ""
echo "🎉 Happy deploying!"
