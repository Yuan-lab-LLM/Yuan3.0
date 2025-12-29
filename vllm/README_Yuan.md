##  1. Docker image


We strongly recommend using the latest release of docker images of Yuan3.0 Flash.

```bash
docker pull yuanmodel/vllm:v0.11.0
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
docker run --gpus all --privileged --ulimit stack=68719476736 --shm-size=1000G -itd -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanmodel/vllm:v0.11.0
docker exec -it your_name bash
```

**3.2 Deployment service **
Yuan3.0 Model just support vLLm V1.
```bash
vllm serve /path/yuanvl3.0-Flash --port 8100 --gpu-memory-utilization 0.9 --tensor-parallel-size 2
```

**3.3 Client request **
```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8010/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

response = client.chat.completions.create(
    model="/path/yuanvl3.0-40B",
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
