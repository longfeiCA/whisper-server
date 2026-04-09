# Whisper Server ARM64 Docker 部署指南

## 环境要求

- **架构**: ARM64 (aarch64) / x86_64
- **操作系统**: Linux (推荐 Ubuntu/Debian)
- **内存**: 至少 2GB RAM (small 模型 int8 量化需要约 300-500MB)
- **Docker**: 已安装并运行
- **Docker Compose**: v2.0+

## 项目概述

本部署使用 faster-whisper (CTranslate2 后端) 构建 REST API 服务，通过 FastAPI 框架提供兼容 OpenAI API 格式的语音转文字服务。

**与 openai-whisper 相比的优势**:
- 内存占用减少 80%+ (int8 量化，small 模型仅需 ~300MB)
- 推理速度提升 3-4 倍 (CTranslate2 算子融合优化)
- 准确度与原版几乎相同 (CTranslate2 官方声称 bit-exact)
- 镜像体积更小 (无需完整 PyTorch)

**服务特点**:
- 支持 ARM64 和 x86_64 架构
- 使用 small 模型 + int8 量化
- 提供 `/v1/audio/transcriptions` 端点
- 支持多种音频格式 (mp3, wav, ogg, m4a 等)

---

## 部署步骤

### 1. 准备工作目录

```bash
# 创建工作目录
mkdir -p /opt/whisper-server
cd /opt/whisper-server

# 创建模型缓存目录
mkdir -p models
```

**备注**: 模型文件会下载到 `models` 目录并持久化，避免容器重启后重新下载。

---

### 2. 检查端口占用

```bash
# 检查 8081 端口是否被占用
sudo ss -tlnp | grep 8081

# 或者使用 netstat (如果已安装)
sudo netstat -tlnp | grep 8081
```

**备注**: 本部署使用 8081 端口。如果 8081 端口已被占用（如其他 Web 服务等），需要修改 docker-compose.yml 更换为其他端口（如 8082、8083 等）。

---

### 3. 创建 Dockerfile

创建 `Dockerfile` 文件：

```dockerfile
FROM python:3.10-slim

# 安装系统依赖
# ffmpeg: 用于音频格式转换
# wget: 用于健康检查
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
# fastapi: Web 框架
# uvicorn: ASGI 服务器
# python-multipart: 处理文件上传
# faster-whisper: CTranslate2 优化的 Whisper，支持 int8 量化
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    faster-whisper

# 设置工作目录
WORKDIR /app

# 复制服务脚本
COPY server.py .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget -q --method=GET -O /dev/null http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "server.py"]
```

**重要备注**:
- 使用 `python:3.10-slim` 基础镜像
- 不再需要 PyTorch，faster-whisper 使用轻量的 CTranslate2 后端
- int8 量化在 CPU 上运行，无需 GPU 支持
- 模型在容器启动时自动下载并加载

---

### 4. 创建 server.py

创建 `server.py` 文件：

```python
import tempfile
import os
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel

app = FastAPI(title="Whisper API Server")

# 使用 int8 量化加载 small 模型
print("Loading Whisper model 'small' with faster-whisper (int8)...")
model_name = "small"
model = WhisperModel(model_name, device="cpu", compute_type="int8")
print(f"Model {model_name} loaded!")

@app.get("/")
def root():
    return {"status": "ok", "model": model_name}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "model_name": model_name}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(None)
):
    """
    语音转文字 API

    Args:
        file: 音频文件 (支持 mp3, wav, ogg, m4a 等格式)
        language: 语言代码，如 zh, en (可选，自动检测)

    Returns:
        text: 转写文本
        language: 检测到的语言
        segments: 分段数量
        processing_time: 处理耗时(秒)
    """
    start_time = time.time()
    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        segments, info = model.transcribe(tmp_path, language=language)
        os.unlink(tmp_path)

        text_parts = []
        segment_list = list(segments)
        for seg in segment_list:
            text_parts.append(seg.text)
        full_text = "".join(text_parts)

        processing_time = round(time.time() - start_time, 3)
        return JSONResponse(content={
            "text": full_text,
            "language": info.language if info.language else (language or "auto"),
            "segments": len(segment_list),
            "processing_time": processing_time,
            "model": model_name
        })
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return JSONResponse(
            content={"error": str(e), "detail": error_detail},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 5. 创建 Docker Compose 文件

创建 `docker-compose.yml` 文件：

```yaml
services:
  whisper:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: whisper-server
    restart: unless-stopped
    ports:
      - "0.0.0.0:8081:8000"
    volumes:
      # 持久化模型文件，避免重复下载
      # faster-whisper 使用 HuggingFace 缓存目录
      - ./models:/root/.cache/huggingface
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "wget", "-q", "--method=GET", "-O", "/dev/null", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

networks:
  default:
    name: whisper-network
```

**关键配置说明**:
- `0.0.0.0:8081:8000`: 将容器内的 8000 端口映射到主机的 8081 端口，并监听所有网络接口
- `./models:/root/.cache/huggingface`: faster-whisper 通过 HuggingFace 下载 CTranslate2 格式模型，挂载卷可持久化存储
- `restart: unless-stopped`: 除非手动停止，否则总是重启容器
- `healthcheck`: 每 30 秒检查一次服务健康状态

---

### 6. 构建并启动服务

```bash
# 进入工作目录
cd /opt/whisper-server

# 构建镜像并启动（首次构建可能需要 3-5 分钟）
docker compose up --build -d

# 查看构建和启动日志
docker compose logs -f
```

**首次启动过程**:
1. 下载 Python 基础镜像
2. 安装系统依赖 (ffmpeg 等)
3. 安装 Python 包 (faster-whisper 比 openai-whisper 小得多)
4. 首次请求时下载 CTranslate2 格式的 small 模型 (~150MB)
5. 加载模型到内存

**备注**: 模型下载可能需要几分钟，取决于网络速度。可以通过 `docker compose logs -f` 查看进度。

---

### 7. 验证服务状态

```bash
# 查看容器状态
docker ps | grep whisper-server

# 预期输出示例:
# xxxxx   whisper-server-whisper   "python server.py"   Up 2 minutes   0.0.0.0:8081->8000/tcp   whisper-server

# 检查健康状态
docker inspect --format='{{.State.Health.Status}}' whisper-server

# 查看日志
docker compose logs --tail=50
```

---

### 8. 测试 API

#### 8.1 健康检查

```bash
# 测试根端点
curl http://localhost:8081/

# 预期响应:
# {"status":"ok","model":"small"}

# 测试健康检查端点
curl http://localhost:8081/health

# 预期响应:
# {"status":"healthy","model_loaded":true,"model_name":"small"}
```

#### 8.2 语音转文字测试

**准备测试音频文件**:

```bash
# 创建测试目录
mkdir -p /tmp/test-audio

# 使用 ffmpeg 生成 5 秒测试音频（需要安装 ffmpeg）
# 或者使用现有音频文件
```

**测试 API 调用**:

```bash
# 使用 curl 测试（替换为你的音频文件路径）
curl -X POST http://localhost:8081/v1/audio/transcriptions \
  -F "file=@/path/to/your/audio.mp3" \
  -F "language=zh"

# 预期响应格式:
# {"text":"转写后的文本内容","language":"zh","segments":5,"processing_time":1.234,"model":"small"}
```

**备注**:
- 支持的语言代码: zh (中文), en (英文), ja (日语) 等
- 不传 `language` 参数会自动检测语言
- 支持的音频格式: mp3, wav, ogg, m4a, webm 等

---

### 9. 从外部网络访问

由于我们配置了 `0.0.0.0:8081`，服务会监听所有网络接口。

```bash
# 获取服务器 IP
hostname -I

# 从其他机器测试（替换为你的服务器 IP）
curl http://<your-server-ip>:8081/
```

**安全提示**:
- 如果服务器有公网 IP，建议配置防火墙限制访问
- 生产环境建议添加认证（如 API Key、反向代理等）

---

### 10. 常用管理命令

```bash
# 停止服务
docker compose down

# 停止并删除容器和镜像
docker compose down --rmi all

# 查看实时日志
docker compose logs -f

# 重启服务
docker compose restart

# 进入容器内部调试
docker exec -it whisper-server /bin/bash

# 查看容器资源使用
docker stats whisper-server
```

---

### 11. 故障排查

#### 问题 1: 端口被占用

**错误信息**:
```
Error starting userland proxy: listen tcp4 0.0.0.0:8081: bind: address already in use
```

**解决方案**:
```bash
# 查找占用端口的进程
sudo ss -tlnp | grep 8081

# 修改 docker-compose.yml 中的端口映射
# 例如改为 8082: "0.0.0.0:8082:8000"

# 重启服务
docker compose down
docker compose up -d
```

#### 问题 2: 模型下载失败

**错误信息**:
```
ConnectionError: Couldn't reach https://huggingface.co/...
```

**解决方案**:
```bash
# 检查网络连接
ping huggingface.co

# 手动下载模型并放置到 models 目录
# CTranslate2 模型会自动从 HuggingFace 下载
```

#### 问题 3: 内存不足

**错误信息**:
```
Killed (OOM)
```

**解决方案**:
```bash
# 使用更小的模型 (tiny 替代 small)
# 修改 server.py 中的模型加载代码:
# model = WhisperModel("tiny", device="cpu", compute_type="int8")

# 或者添加 swap 分区
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 问题 4: 容器启动后立即退出

**排查步骤**:
```bash
# 查看详细日志
docker compose logs --tail=100

# 检查端口冲突
sudo ss -tlnp | grep 8081

# 检查模型目录权限
ls -la /opt/whisper-server/models

# 手动运行容器查看错误
docker run -it --rm whisper-server-whisper python server.py
```

---

## 文件清单

部署完成后，工作目录应包含以下文件：

```
/opt/whisper-server/
├── docker-compose.yml    # Docker Compose 配置文件
├── Dockerfile            # Docker 镜像构建文件
├── server.py             # FastAPI 服务脚本
├── models/               # 模型缓存目录（自动生成）
│   └── hub/             # HuggingFace 缓存（CTranslate2 模型）
└── DEPLOY.md            # 本说明文档
```

---

## 性能参考

### faster-whisper (当前方案)

| 模型 | 量化 | 内存占用 | 推理速度 (CPU) |
|------|------|----------|----------------|
| tiny | int8 | ~80 MB | ~4-6x 实时 |
| base | int8 | ~150 MB | ~2-4x 实时 |
| small | int8 | ~300 MB | ~1-2x 实时 |
| medium | int8 | ~600 MB | ~0.3-0.5x 实时 |

### 对比 openai-whisper (原方案)

| 模型 | 内存占用 | 推理速度 (CPU) |
|------|----------|----------------|
| small (float32) | ~2.8 GB | ~0.3-0.5x 实时 |

> faster-whisper small int8 相比 openai-whisper small float32：内存节省 ~89%，速度提升 3-4 倍

---

## 安全建议

1. **防火墙配置**: 限制 8081 端口访问
   ```bash
   # 仅允许特定 IP 访问
   sudo ufw allow from <allowed-ip> to any port 8081
   ```

2. **反向代理**: 使用 Nginx/Caddy 添加 HTTPS
   ```nginx
   server {
       listen 443 ssl;
       server_name whisper.yourdomain.com;

       location / {
           proxy_pass http://localhost:8081;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **API 认证**: 在 server.py 中添加 API Key 验证

---

## 进阶配置

### 切换模型

修改 `server.py` 中的模型加载行：

```python
# tiny: 最小最快，准确率较低 (~80MB)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# base: 平衡速度和准确率 (~150MB)
model = WhisperModel("base", device="cpu", compute_type="int8")

# small: 更准但更慢，推荐 (~300MB)
model = WhisperModel("small", device="cpu", compute_type="int8")

# medium: 最准，需要更多内存 (~600MB)
model = WhisperModel("medium", device="cpu", compute_type="int8")
```

### 切换量化精度

```python
# int8: 最省内存，推荐 CPU 使用
model = WhisperModel("small", device="cpu", compute_type="int8")

# float16: 精度略高，内存多一倍 (需要 GPU 或支持 FP16 的 CPU)
model = WhisperModel("small", device="cpu", compute_type="float16")
```

**注意**: 切换模型后需要重新构建镜像：
```bash
docker compose down
docker compose up --build -d
```

### 多语言支持

Whisper 自动检测语言，但可以通过 `language` 参数指定：

```bash
# 指定中文
curl -X POST http://localhost:8081/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=zh"

# 指定英文
curl -X POST http://localhost:8081/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=en"
```

---

## 总结

本部署方案使用 faster-whisper (CTranslate2 后端) 替代 openai-whisper，大幅降低内存占用并提升推理速度，同时保持几乎相同的转写准确度。

**关键要点**:
1. 使用 `faster-whisper` 替代 `openai-whisper`，内存从 ~2.8GB 降至 ~300MB
2. int8 量化在 CPU 上高效运行，无需 GPU
3. 模型通过 HuggingFace 下载，缓存到 `models/` 目录持久化
4. 准确度与 openai-whisper 几乎相同 (CTranslate2 官方声称 bit-exact)
5. 推理速度提升 3-4 倍

如有问题，请检查日志并参考故障排查部分。