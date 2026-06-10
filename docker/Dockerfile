FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# 系统依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip

# Python 依赖
COPY requirements.txt /tmp/requirements.txt

# Pytorch (CUDA 12.4)
RUN curl -L -o "/tmp/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" && \
    pip install --no-cache-dir "/tmp/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl" && \
    rm "/tmp/torch-2.6.0+cu124-cp310-cp310-linux_x86_64.whl"

# Python 依赖
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载模型
RUN python -c "from modelscope.hub.snapshot_download import snapshot_download; snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')"

# 项目代码
WORKDIR /app
COPY . .
RUN cp .env.example .env

# 清除损坏的 Python 字节码缓存
RUN find /usr/local/lib/python3.10 -name "*.pyc" -delete

# 容器日志走 stdout，不写文件
ENV LOG_DIR=/tmp/logs

# 服务端口
EXPOSE 8765

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]
