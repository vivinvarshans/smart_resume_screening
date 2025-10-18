#!/bin/bash

# Resume Matcher - Frontend Deployment to Vercel
# This script helps you deploy the frontend to Vercel

set -e

echo "🚀 Resume Matcher - Frontend Deployment to Vercel"
echo "=================================================="
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI is not installed."
    echo ""
    echo "Install it with:"
    echo "  npm install -g vercel"
    echo ""
    exit 1
fi

echo "✅ Vercel CLI found"
echo ""

# Navigate to frontend directory
cd apps/frontend

echo "📦 Installing dependencies..."
npm install

echo ""
echo "🔧 Building frontend..."
npm run build

echo ""
echo "🚀 Deploying to Vercel..."
echo ""

# Deploy to Vercel
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📚 Next steps:"
echo "  1. Note your Vercel URL from above"
echo "  2. Update CORS in backend: apps/backend/app/core/config.py"
echo "  3. Add your Vercel URL to ALLOWED_ORIGINS"
echo "  4. Commit and push to redeploy backend"
echo ""
echo "🎉 Your Resume Matcher is now live!"
