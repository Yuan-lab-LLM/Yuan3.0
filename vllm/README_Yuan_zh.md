##  1.镜像方式（建议）

我们强烈建议使用 Yuan3.0 Flash 的最新版本 docker 镜像。
```bash
docker pull yuanlabai/vllm:v0.11.0
```

##  2.源码编译（可选）

从源码安装vllm
```
git clone https://github.com/Yuan-lab-LLM/Yuan3.0.git
cd Yuan3.0/vllm
pip install -e .
```

##  3. 快速开始

**3.1  环境配置**

您可以使用以下 Docker 命令启动 Yuan3.0 Flash 容器实例：
```bash
docker run --gpus all --privileged --ulimit stack=68719476736 --shm-size=1000G -itd -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanlabai/vllm:v0.11.0
docker exec -it your_name bash
```

**3.2  部署服务**

Yuan3.0 Flash Model 仅支持 vLLm V1架构。
```bash
vllm serve /path/yuanvl3.0-Flash --port 8100 --gpu-memory-utilization 0.9 --tensor-parallel-size 2
```

**3.2  请求调用**

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8010/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

response = client.chat.completions.create(
    model="/path/Yuanvl3.0-Flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello!"},
        ],
    }
    ],
    max_tokens=256,
    temperature=1e-6,
)
print("Chat completion output:", response.choices[0].message.content)
```
