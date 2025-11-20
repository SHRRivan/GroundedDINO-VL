# Label Studio Integration Guide

This guide explains how to run Label Studio with GroundedDINO-VL as an ML backend for automated object detection and annotation.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Dockerfile Details](#dockerfile-details)
- [Persistent Data Storage](#persistent-data-storage)
- [Connecting GroundedDINO-VL Backend](#connecting-groundeddino-vl-backend)
- [Complete Workflow](#complete-workflow)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

This setup provides:
- **Label Studio** on port 8080 for data annotation
- **GroundedDINO-VL** on port 9090 as ML backend
- **Persistent storage** for annotations and projects
- **Docker Compose** orchestration for easy deployment

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                                                             │
│  ┌──────────────────┐          ┌────────────────────────┐  │
│  │  Label Studio    │          │  GroundedDINO-VL       │  │
│  │  Port: 8080      │◄────────►│  Port: 9090            │  │
│  │  User: ls        │  ML API  │  Runtime: NVIDIA       │  │
│  │                  │          │  GPU Support           │  │
│  └──────────────────┘          └────────────────────────┘  │
│          │                              │                   │
│          ▼                              ▼                   │
│  ┌──────────────────┐          ┌────────────────────────┐  │
│  │ label-studio-data│          │  models/               │  │
│  │ (Volume)         │          │  groundingdino_swint...│  │
│  └──────────────────┘          └────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Download Model Checkpoint

```bash
# Create models directory and download checkpoint
mkdir -p models
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth \
  -O models/groundingdino_swinb_cogcoor.pth
```

### 2. Start All Services

```bash
# Start both Label Studio and GroundedDINO-VL
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Access Label Studio

Open your browser and navigate to:
```
http://localhost:8080
```

**First-time setup:**
1. Create an admin account (email + password)
2. You'll be redirected to the main dashboard

### 4. Configure ML Backend

In Label Studio:
1. Go to **Settings** (gear icon)
2. Select **Machine Learning**
3. Click **Add Model**
4. Enter:
   - **URL:** `http://groundeddino-vl:9090`
   - **Title:** GroundedDINO-VL
   - **Description:** Zero-shot object detection
5. Click **Validate and Save**

---

## Dockerfile Details

### Base Configuration

The Label Studio Dockerfile ([Dockerfile.labelstudio](Dockerfile.labelstudio)) includes:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    wget \
    libpq-dev \
    git \
    libmagic1

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash ls

# Install Label Studio
RUN pip install label-studio label-studio-sdk

# Configuration
ENV LABEL_STUDIO_HOST=0.0.0.0
ENV LABEL_STUDIO_PORT=8080

EXPOSE 8080

ENTRYPOINT ["label-studio", "start", "--host", "0.0.0.0", "--port", "8080"]
```

### Key Features

- **Non-root user:** Runs as `ls` user (UID 1000) for security
- **Persistent data:** Volume mount at `/label-studio/data`
- **Health checks:** Monitors service availability
- **Auto-restart:** Configured with `restart: unless-stopped`

---

## Persistent Data Storage

### Volume Mounting

All Label Studio data is stored in `./label-studio-data/`:

```
label-studio-data/
├── media/              # Uploaded images and files
├── sqlite.db           # Project database (SQLite)
├── projects/           # Project configurations
└── annotations/        # Annotation exports
```

### Data Persistence Commands

```bash
# View data directory
ls -la label-studio-data/

# Backup data
tar -czf label-studio-backup-$(date +%Y%m%d).tar.gz label-studio-data/

# Restore from backup
tar -xzf label-studio-backup-20251119.tar.gz

# Clear all data (CAUTION: destroys all projects)
docker-compose down
rm -rf label-studio-data/
docker-compose up -d
```

### Using Named Volumes

The docker-compose.yml also creates a named volume:

```yaml
volumes:
  label-studio-data:
    driver: local
```

To use this instead of a bind mount:

```yaml
volumes:
  - label-studio-data:/label-studio/data  # Named volume
  # - ./label-studio-data:/label-studio/data  # Bind mount (default)
```

---

## Connecting GroundedDINO-VL Backend

### Step-by-Step Configuration

#### 1. Ensure Both Services Are Running

```bash
# Check service status
docker-compose ps

# Expected output:
# NAME                    STATUS
# groundeddino-vl-server  Up (healthy)
# label-studio            Up (healthy)
```

#### 2. Test Backend Connectivity

From within the Label Studio container:

```bash
# Test connection
docker exec label-studio curl http://groundeddino-vl:9090/health

# Expected response:
# {"status":"ok","model_loaded":true}
```

#### 3. Add ML Backend in Label Studio UI

1. **Login to Label Studio:** http://localhost:8080
2. **Navigate to Settings:** Click gear icon (⚙️) in top right
3. **Select Machine Learning** from left sidebar
4. **Click "Add Model"** button
5. **Fill in the form:**
   ```
   URL: http://groundeddino-vl:9090
   Title: GroundedDINO-VL
   Description: Zero-shot object detection with natural language queries
   ```
6. **Click "Validate and Save"**

The backend should show a green checkmark indicating successful connection.

---

## Complete Workflow

### Example: Object Detection Project

#### 1. Create a New Project

1. Click **Create Project**
2. Enter **Project Name:** "Object Detection Dataset"
3. Click **Create**

#### 2. Configure Labeling Interface

In the project settings, go to **Labeling Interface** and use this template:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="person" background="red"/>
    <Label value="car" background="blue"/>
    <Label value="bicycle" background="green"/>
    <Label value="dog" background="yellow"/>
    <Label value="cat" background="orange"/>
  </RectangleLabels>
</View>
```

Click **Save**.

#### 3. Import Images

1. Go to **Import** tab
2. Upload images (or paste URLs)
3. Click **Import**

#### 4. Connect ML Backend to Project

1. In project settings, go to **Machine Learning**
2. Click **Connect Model**
3. Select **GroundedDINO-VL**
4. Click **Add to Project**

#### 5. Enable Auto-Annotations

1. Go to any unlabeled task
2. Click **Auto-annotate** button (magic wand icon)
3. GroundedDINO-VL will detect objects based on your label classes

#### 6. Review and Correct

1. Review the auto-generated annotations
2. Adjust bounding boxes as needed
3. Click **Submit** to save

#### 7. Export Annotations

1. Go to **Export** tab
2. Select format (COCO, Pascal VOC, YOLO, etc.)
3. Click **Export**

---

## Troubleshooting

### Issue: Cannot Connect to ML Backend

**Symptoms:**
- "Connection failed" error in Label Studio
- Backend shows red X instead of green checkmark

**Solutions:**

1. **Check if both containers are running:**
   ```bash
   docker-compose ps
   ```

2. **Verify network connectivity:**
   ```bash
   docker exec label-studio ping groundeddino-vl
   ```

3. **Check GroundedDINO-VL logs:**
   ```bash
   docker-compose logs groundeddino-vl
   ```

4. **Restart services:**
   ```bash
   docker-compose restart
   ```

### Issue: Label Studio Won't Start

**Symptoms:**
- Container exits immediately
- Port 8080 already in use

**Solutions:**

1. **Check port availability:**
   ```bash
   lsof -i :8080
   # or
   netstat -tuln | grep 8080
   ```

2. **Use different port:**
   Edit `docker-compose.yml`:
   ```yaml
   ports:
     - "8888:8080"  # Map to different host port
   ```

3. **Check logs:**
   ```bash
   docker-compose logs label-studio
   ```

### Issue: Data Not Persisting

**Symptoms:**
- Projects disappear after restart
- Uploaded images are lost

**Solutions:**

1. **Verify volume mount:**
   ```bash
   docker inspect label-studio | grep -A 10 Mounts
   ```

2. **Check permissions:**
   ```bash
   ls -la label-studio-data/
   # Should be owned by UID 1000 (ls user)
   ```

3. **Fix permissions:**
   ```bash
   sudo chown -R 1000:1000 label-studio-data/
   ```

### Issue: Auto-annotation Not Working

**Symptoms:**
- "Auto-annotate" button does nothing
- No predictions appear

**Solutions:**

1. **Verify ML backend is connected:**
   - Check green checkmark in Settings > Machine Learning
   - Must be **connected to project**, not just added globally

2. **Check model compatibility:**
   - Ensure label names match detection classes
   - GroundedDINO-VL works with natural language prompts

3. **Test backend directly:**
   ```bash
   curl -X POST http://localhost:9090/predict \
     -H "Content-Type: application/json" \
     -d '{
       "data": {"image": "https://example.com/test.jpg"},
       "prompt": "person. car. dog."
     }'
   ```

---

## Advanced Configuration

### Custom Port Configuration

To run on different ports, edit `docker-compose.yml`:

```yaml
label-studio:
  ports:
    - "8888:8080"  # External:Internal
  environment:
    LABEL_STUDIO_PORT: 8080  # Keep internal port
```

### Using PostgreSQL Instead of SQLite

For production deployments, use PostgreSQL:

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: labelstudio
      POSTGRES_USER: labelstudio
      POSTGRES_PASSWORD: changeme
    volumes:
      - postgres-data:/var/lib/postgresql/data

  label-studio:
    environment:
      DJANGO_DB: default
      POSTGRE_NAME: labelstudio
      POSTGRE_USER: labelstudio
      POSTGRE_PASSWORD: changeme
      POSTGRE_PORT: 5432
      POSTGRE_HOST: postgres
    depends_on:
      - postgres

volumes:
  postgres-data:
```

### Enable Cloud Storage (S3)

```yaml
label-studio:
  environment:
    ENABLE_S3: 'true'
    AWS_ACCESS_KEY_ID: your-access-key
    AWS_SECRET_ACCESS_KEY: your-secret-key
    AWS_STORAGE_BUCKET_NAME: your-bucket
    AWS_S3_REGION_NAME: us-west-2
```

### Multi-User Setup with Authentication

```yaml
label-studio:
  environment:
    LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK: 'true'
    LABEL_STUDIO_USERNAME: admin@example.com
    LABEL_STUDIO_PASSWORD: SecurePassword123
```

### Custom Labeling Prompts for GroundedDINO-VL

Create a custom preprocessing hook to convert Label Studio labels to GroundedDINO prompts:

```python
# custom_ml_backend.py
from label_studio_ml.model import LabelStudioMLBase

class CustomGroundedDINO(LabelStudioMLBase):
    def predict(self, tasks, **kwargs):
        # Extract label names from task schema
        labels = self.parsed_label_config.get('RectangleLabels')[0]['labels']
        prompt = '. '.join(labels) + '.'

        # Call GroundedDINO backend with custom prompt
        # ...
```

---

## Production Deployment Checklist

- [ ] Use PostgreSQL instead of SQLite
- [ ] Configure HTTPS/SSL with reverse proxy (nginx)
- [ ] Set strong admin passwords
- [ ] Enable user authentication
- [ ] Configure regular backups
- [ ] Set up monitoring and logging
- [ ] Use persistent named volumes
- [ ] Configure resource limits
- [ ] Enable Redis for task queue (optional)
- [ ] Set up cloud storage (S3/GCS) for large datasets

---

## Useful Commands

```bash
# Start only Label Studio
docker-compose up -d label-studio

# Start only GroundedDINO-VL
docker-compose up -d groundeddino-vl

# Rebuild Label Studio image
docker-compose build label-studio

# Access Label Studio shell
docker exec -it label-studio bash

# View Label Studio logs
docker-compose logs -f label-studio

# Export Label Studio database
docker exec label-studio sqlite3 /label-studio/data/sqlite.db .dump > backup.sql

# Reset Label Studio admin password
docker exec -it label-studio label-studio reset_password
```

---

## Resources

- **Label Studio Docs:** https://labelstud.io/guide/
- **ML Backend Guide:** https://labelstud.io/guide/ml.html
- **GroundedDINO-VL Repo:** https://github.com/ghostcipher1/GroundedDINO-VL
- **Docker Hub - Label Studio:** https://hub.docker.com/r/heartexlabs/label-studio

---

**Last Updated:** 2025-11-19
**Version:** 1.0.0
**Maintainer:** Trent Adams (ghostcipher02@gmail.com)
