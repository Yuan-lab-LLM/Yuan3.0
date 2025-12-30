##  1. Docker image

We strongly recommend using the latest release of docker images of Yuan3.0 Flash.

```bash
docker pull yuanlabai/vllm:v0.11.0
```


##  2. Install (optional)

Install vLLM from source:
```
git clone https://github.com/Yuan-lab-LLM/Yuan3.0.git
cd Yuan3.0/vllm
pip install -e .
```


##  3. Quick Start


**3.1  Environment Config**

You can launch an instance of the Yuan3.0 Flash container with the following Docker commands:

```bash
docker run --gpus all --privileged --ulimit stack=68719476736 --shm-size=1000G -itd -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanlabai/vllm:v0.11.0
docker exec -it your_name bash
```

**3.2 Deployment service**

Yuan3.0 Flash Model just support vLLm V1.
```bash
python -m vllm.entrypoints.openai.api_server --model=/path/Yuan3__0-Flash --port 8100 --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --trust-remote-code --allowed-local-media-path "/path/images"
```

**3.3 Client request**

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8100/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompt = '请描述这张图片的内容'
image_path = 'Yuan3.0/vllm/docs/images/image.jpg'
image_url = f"file:{image_path}"

response = client.chat.completions.create(
    model="/path/Yuan3__0-Flash",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": f"{prompt}"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }
    ],
    max_tokens=256,
    temperature=1e-6,
)
print("Chat completion output:", response.choices[0].message.content)
```
