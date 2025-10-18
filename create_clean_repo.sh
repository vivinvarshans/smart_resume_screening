#!/bin/bash

# Script to create a clean repository with only your commits
# Run this after renaming the old repo on GitHub

echo "Creating clean repository..."

# Create new directory
NEW_DIR="/Users/vivinvarshans/Downloads/smart_resume_screening_clean"
mkdir -p "$NEW_DIR"

# Copy all files except .git
echo "Copying files..."
rsync -av --exclude='.git' --exclude='node_modules' --exclude='.next' --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' /Users/vivinvarshans/Downloads/Resume_Matcher/ "$NEW_DIR/"

# Initialize new git repo
cd "$NEW_DIR"
git init
git branch -m main

# Configure git
git config user.name "Vivin Varshan S"
git config user.email "s.vivinvarshannd@gmail.com"

# Create initial commit
git add .
git commit -m "Initial commit: AI-Powered Smart Resume Screening System

Features:
- AI-powered resume analysis and job matching
- Real-time keyword extraction and scoring
- Interactive dashboard with visualizations
- PDF resume parsing and processing
- RESTful API with FastAPI backend
- Modern React/Next.js frontend
- Deployed on Railway (backend) and Vercel (frontend)"

echo ""
echo "✅ Clean repository created at: $NEW_DIR"
echo ""
echo "Next steps:"
echo "1. Go to GitHub and rename 'smart_resume_screening' to 'smart_resume_screening_old'"
echo "2. Create a new repository named 'smart_resume_screening' on GitHub"
echo "3. Run these commands:"
echo ""
echo "   cd $NEW_DIR"
echo "   git remote add origin https://github.com/vivinvarshans/smart_resume_screening.git"
echo "   git push -u origin main"
echo ""
