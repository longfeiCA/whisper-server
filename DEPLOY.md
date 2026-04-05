# Whisper Server ARM64 Docker 部署指南

## 环境要求

- **架构**: ARM64 (aarch64)
- **操作系统**: Linux (推荐 Ubuntu/Debian)
- **内存**: 至少 4GB RAM (base 模型需要约 1GB)
- **Docker**: 已安装并运行
- **Docker Compose**: v2.0+

## 项目概述

本部署使用 OpenAI Whisper Python 库构建 REST API 服务，通过 FastAPI 框架提供兼容 OpenAI API 格式的语音转文字服务。

**服务特点**:
- 支持 ARM64 架构
- 自动下载 base 模型
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
# wget: 用于下载模型（如需要）
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
# fastapi: Web 框架
# uvicorn: ASGI 服务器
# python-multipart: 处理文件上传
# openai-whisper: OpenAI Whisper 语音识别库
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    openai-whisper

# 设置工作目录
WORKDIR /app

# 创建 FastAPI 服务脚本
RUN cat > server.py << 'EOF'
import whisper
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Whisper API Server")

# 全局变量存储模型
model = None

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    global model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    print("Model loaded successfully!")

@app.get("/")
def root():
    """健康检查端点"""
    return {"status": "ok", "model": "base", "service": "whisper-api"}

@app.get("/health")
def health():
    """健康检查"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    language: str = Form(None)
):
    """
    语音转文字 API
    
    Args:
        file: 音频文件 (支持 mp3, wav, ogg, m4a 等格式)
        model_name: 模型名称 (可选，默认 base)
        language: 语言代码，如 zh, en (可选，自动检测)
    
    Returns:
        text: 转写文本
        language: 检测到的语言
        segments: 分段数量
    """
    try:
        # 保存上传的文件到临时文件
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        # 使用 Whisper 进行转写
        result = model.transcribe(
            tmp_path,
            language=language,
            fp16=False  # ARM 架构通常不支持 fp16
        )
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        return JSONResponse(content={
            "text": result["text"].strip(),
            "language": result.get("language", language or "auto"),
            "segments": len(result.get("segments", []))
        })
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error: {error_detail}")
        return JSONResponse(
            content={"error": str(e), "detail": error_detail},
            status_code=500
        )

if __name__ == "__main__":
    # 监听所有网络接口，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "server.py"]
```

**重要备注**:
- 使用 `python:3.10-slim` 基础镜像，支持 ARM64
- `fp16=False` 是必须的，因为 ARM 架构通常不支持半精度浮点运算
- 模型在容器启动时加载，首次启动需要等待 1-2 分钟
- 音频文件通过 FFmpeg 自动转换格式

---

### 4. 创建 Docker Compose 文件

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
      # 格式: "主机IP:主机端口:容器端口"
      # 0.0.0.0 表示监听所有网络接口
      - "0.0.0.0:8081:8000"
    volumes:
      # 持久化模型文件，避免重复下载
      - ./models:/root/.cache/whisper
    environment:
      - PYTHONUNBUFFERED=1
      - MODEL_SIZE=base
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    # 限制资源使用（可选）
    # deploy:
    #   resources:
    #     limits:
    #       memory: 4G
    #     reservations:
    #       memory: 1G

networks:
  default:
    name: whisper-network
    driver: bridge
```

**关键配置说明**:
- `0.0.0.0:8081:8000`: 将容器内的 8000 端口映射到主机的 8081 端口，并监听所有网络接口
- `./models:/root/.cache/whisper`: Whisper 库默认将模型下载到 `~/.cache/whisper`，挂载卷可持久化存储
- `restart: unless-stopped`: 除非手动停止，否则总是重启容器
- `healthcheck`: 每 30 秒检查一次服务健康状态

---

### 5. 构建并启动服务

```bash
# 进入工作目录
cd /opt/whisper-server

# 构建镜像并启动（首次构建可能需要 5-10 分钟）
docker compose up --build -d

# 查看构建和启动日志
docker compose logs -f
```

**首次启动过程**:
1. 下载 Python 基础镜像
2. 安装系统依赖 (ffmpeg 等)
3. 安装 Python 包
4. 下载 Whisper base 模型 (~150MB)
5. 加载模型到内存

**备注**: 模型下载可能需要几分钟，取决于网络速度。可以通过 `docker compose logs -f` 查看进度。

---

### 6. 验证服务状态

```bash
# 查看容器状态
docker ps | grep whisper-server

# 预期输出示例:
# d00627a6aaa6   whisper-server-whisper   "python server.py"   Up 2 minutes   0.0.0.0:8081->8000/tcp   whisper-server

# 检查健康状态
docker inspect --format='{{.State.Health.Status}}' whisper-server

# 查看日志
docker compose logs --tail=50
```

---

### 7. 测试 API

#### 7.1 健康检查

```bash
# 测试根端点
curl http://localhost:8081/

# 预期响应:
# {"status":"ok","model":"base","service":"whisper-api"}

# 测试健康检查端点
curl http://localhost:8081/health

# 预期响应:
# {"status":"healthy","model_loaded":true}
```

#### 7.2 语音转文字测试

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
  -F "model_name=base" \
  -F "language=zh"

# 预期响应格式:
# {"text":"转写后的文本内容","language":"zh","segments":5}
```

**备注**: 
- 支持的语言代码: zh (中文), en (英文), ja (日语) 等
- 不传 `language` 参数会自动检测语言
- 支持的音频格式: mp3, wav, ogg, m4a, webm 等

---

### 8. 从外部网络访问

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

### 9. 常用管理命令

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

### 10. 故障排查

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
URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```

**解决方案**:
```bash
# 检查网络连接
ping google.com

# 手动下载模型并放置到 models 目录
# 模型下载地址: https://openaipublic.azureedge.net/main/whisper/models/...
```

#### 问题 3: 内存不足

**错误信息**:
```
RuntimeError: CUDA out of memory
# 或
Killed (OOM)
```

**解决方案**:
```bash
# 使用更小的模型 (tiny 替代 base)
# 修改 Dockerfile 中的模型加载代码:
# model = whisper.load_model("tiny")

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
├── models/               # 模型缓存目录（自动生成）
│   └── base.pt          # Whisper base 模型文件（自动下载）
└── README.md            # 本说明文档
```

---

## 性能参考

在 ARM64 (2 vCPU, 4GB RAM) 环境下：

- **启动时间**: 1-2 分钟（含模型加载）
- **内存占用**: 约 1-1.5GB
- **转写速度**: 
  - base 模型: 约 0.5-1x 实时（1 分钟音频需要 0.5-1 秒处理）
  - tiny 模型: 约 1-2x 实时（更快但准确率较低）

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

修改 `Dockerfile` 中的模型加载行：

```python
# tiny: 最小最快，准确率较低
model = whisper.load_model("tiny")

# base: 平衡速度和准确率（推荐）
model = whisper.load_model("base")

# small: 更准但更慢，需要更多内存
model = whisper.load_model("small")
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

本部署方案提供了一种在 ARM64 服务器上运行 Whisper 语音识别服务的稳定方法。通过 Docker 容器化部署，可以方便地在多台服务器间迁移和扩展。

**关键要点**:
1. 使用 `python:3.10-slim` 作为基础镜像确保 ARM64 兼容性
2. `fp16=False` 是必须的，避免 ARM 架构浮点运算问题
3. 模型持久化到宿主机，避免重复下载
4. 端口映射使用 `0.0.0.0` 允许外部访问

如有问题，请检查日志并参考故障排查部分。
