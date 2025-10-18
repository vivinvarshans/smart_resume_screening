# 🚂 Railway Quick Start Guide

Deploy your Resume Matcher to Railway in 5 minutes!

## 🎯 Quick Deploy (3 Steps)

### Step 1: Install Railway CLI

```bash
# macOS/Linux
curl -fsSL https://railway.app/install.sh | sh

# Windows (PowerShell)
iwr https://railway.app/install.ps1 | iex
```

### Step 2: Run Deployment Script

```bash
./deploy.sh
```

The script will:
- Login to Railway
- Initialize your project
- Set up environment variables
- Deploy your application

### Step 3: Get Your URL

After deployment completes, Railway will show your app URL:
```
https://your-app-name.up.railway.app
```

---

## 🌐 Deploy via Railway Dashboard (No CLI)

### 1. Push to GitHub

```bash
git add .
git commit -m "Ready for Railway deployment"
git push origin main
```

### 2. Connect to Railway

1. Go to [railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose `vivinvarshans/smart_resume_screening`

### 3. Add Environment Variables

Click **"Variables"** and add:

```
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_PROVIDER=groq
LL_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
SESSION_SECRET_KEY=your-random-secret-key
SYNC_DATABASE_URL=sqlite:///./app.db
ASYNC_DATABASE_URL=sqlite+aiosqlite:///./app.db
PYTHONDONTWRITEBYTECODE=1
EMBEDDING_PROVIDER=none
EMBEDDING_MODEL=none
```

### 4. Deploy!

Railway will automatically build and deploy. Wait 2-5 minutes.

---

## ✅ Verify Deployment

### Test Health Endpoint

```bash
curl https://your-app.up.railway.app/health
```

### View API Docs

Open in browser:
```
https://your-app.up.railway.app/docs
```

---

## 🎨 Deploy Frontend (Vercel)

### 1. Go to Vercel

Visit [vercel.com](https://vercel.com) and import your repo

### 2. Configure

- **Root Directory**: `apps/frontend`
- **Framework**: Vite
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

### 3. Add Environment Variable

```
VITE_API_BASE_URL=https://your-railway-app.up.railway.app/api/v1
```

### 4. Deploy

Click **"Deploy"** and wait ~2 minutes.

---

## 🔧 Update CORS (Important!)

After deploying frontend, update backend CORS:

Edit `apps/backend/app/core/config.py`:

```python
ALLOWED_ORIGINS: List[str] = [
    "https://your-frontend.vercel.app",  # Add your Vercel URL
    "http://localhost:5173",
    "http://localhost:3000"
]
```

Commit and push:
```bash
git add .
git commit -m "Update CORS for production"
git push
```

Railway will auto-redeploy.

---

## 📊 Monitor Your App

### View Logs
```bash
railway logs
```

### Open Dashboard
```bash
railway open
```

### Check Status
```bash
railway status
```

---

## 🆘 Troubleshooting

### Build Fails?
- Check `railway logs`
- Ensure all dependencies are in `requirements.txt`

### App Won't Start?
- Verify environment variables are set
- Check that `PORT` is used correctly

### 502 Error?
- App might be starting up (wait 30 seconds)
- Check logs for errors

### Groq API Errors?
- Verify API key is correct
- Check quota at console.groq.com

---

## 💡 Pro Tips

1. **Use PostgreSQL for Production**
   - Add PostgreSQL database in Railway
   - More reliable than SQLite

2. **Enable Auto-Deploy**
   - Railway auto-deploys on git push
   - Perfect for continuous deployment

3. **Monitor Usage**
   - Check Railway dashboard for metrics
   - Set up alerts for errors

4. **Backup Database**
   - Export data regularly
   - Use PostgreSQL for better persistence

---

## 🎉 You're Live!

Your Resume Matcher is now deployed and ready to use!

**Backend**: `https://your-app.up.railway.app`
**Frontend**: `https://your-app.vercel.app`
**API Docs**: `https://your-app.up.railway.app/docs`

Share it with the world! 🚀
