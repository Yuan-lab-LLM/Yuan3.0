<div align="center">
<h1>
  Yuan 3.0 Multimodal Foundation Model
</h1>
</div>

<hr>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/YuanLabAI"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Yuan3.0-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://www.modelscope.cn/profile/Yuanlab"><img alt="ModelScope"
    src="https://img.shields.io/badge/💾%20ModelScope-Yuan3.0-6b4fbb?color=6b4fbb&logoColor=white"/></a>
  <a href="https://x.com/Yuanlabai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-Yuanlabai-white?logo=x&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2601.01718"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-Yuan3.0%20Paper-b31b1b?logo=arxiv&logoColor=white"/></a>
  </a>


  
</div>



<h4 align="center">
    <p>
        <a href="./README_ZH.md">简体中文</a> |
        <b>English</b>
    <p>
</h4>


-----



## Latest Updates 🎉🎉

* **[2025-12-30]** **Released Yuan 3.0-40B Multimodal Large Language Model, a high-performance model for enterprise-grade application scenarios: Yuan3.0 Flash**



## 1. Introduction

Yuan 3.0 Flash, developed by the **YuanLab.ai team**, is a **40B parameter multimodal foundation model** that employs a Mixture of Experts (MoE) architecture, activating only approximately **3.7B parameters** per inference. Through innovative reinforcement learning training methods (RAPO), it significantly reduces inference token consumption while improving reasoning accuracy, exploring the innovative path of "less computation, higher intelligence" for large language models. We have also released the <a href="https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/YUAN3.0_FLASH-paper.pdf" target="_blank">**technical report**</a> for the Yuan3.0 model, where you can find more detailed technical information and evaluation results.

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan3.0 Multimodal Large Language Model Architecture

</div>

### Core Features

- 🚀 **Efficient Inference**: Reduces inference token consumption by up to 75%, significantly lowering costs
- 🎯 **Enterprise-Grade Optimization**: Deeply optimized for enterprise scenarios such as RAG, document understanding, and table analysis
- 🎨 **Multimodal Support**: Supports text, image, table, document and other multimodal inputs
- 📚 **Long Context**: Supports 128K context length, achieving 100% accuracy in "Needle in a Haystack" tests
- ⚡ **Ready-to-Use Intelligence**: Default inference mode meets the needs of most enterprise scenarios

## 2. Performance

Yuan 3.0 Flash outperforms GPT-5.1 in enterprise-grade RAG, multimodal retrieval, table understanding, summary generation and other tasks. With 40B parameters, it achieves the reasoning accuracy of 235B/671B models while reducing token consumption by 50%-75%, providing enterprises with high-performance, low-cost large language model solutions.

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-benchmarks.png width=80% />

Fig.2: Yuan3.0 Flash Evaluation Results

</div>



## 3. Core Technology

### RAPO Reinforcement Learning Algorithm

The innovative **Reflection-aware Adaptive Policy Optimization (RAPO)** algorithm, through the Reflection Inhibition Reward Mechanism (RIRM):

- ✅ Identifies the key point where the correct answer is first obtained
- 🎯 Suppresses subsequent redundant reasoning behavior
- 📉 Improves accuracy while reducing inference token count by approximately 75%

| Training Method | AIME 2024 Accuracy | Avg Output Length | MATH-500 Accuracy | Avg Output Length |
|---------|------------------|--------------|-----------------|--------------|
| Yuan3.0 Flash (40B) SFT | 31.45% | 13,656 tokens | 83.20% | 3,362 tokens |
| RL+DAPO length-penalty | 46.35% | 13,781 tokens | 89.06% | 3,974 tokens |
| **RL+RIRM** | **47.92%** | **7,505 tokens** | **89.47%** | **1,777 tokens** |





## 4. Model Download

**We provide download links for multiple model formats:**

|    Model     |   Parameters  |  Precision  |   Sequence Length  |   Model Format   |         Download Link         |
| :----------: | :------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Flash |    40B    |  16bit    |    128K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/Yuanlab/Yuan3.0-Flash) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Flash) \|  [WiseModel]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Flash)
| Yuan3.0 Flash 4bit |    40B   |  4bit     |    128K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/Yuanlab/Yuan3.0-Flash-int4) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Flash-4bit) \|  [WiseModel]( https://www.wisemodel.cn/models/YuanLab/Yuan3.0-Flash-4bit)





## 5. Evaluation Results

**5.1 Text-based RAG Evaluation: ChatRAG** 🏆

Yuan 3.0 Flash leads DeepSeek-V3, DeepSeek-R1 and other large language models in average accuracy across 10 evaluation tasks in the industry-standard RAG benchmark ChatRAG.

**Model Average Accuracy Comparison**


| Models | Avg All | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
|--------|---------|-----|------|-------|------|------|------|-----|------|-------|--------|
| **DeepSeek-V3** | 50.47 | 31.59 | 28.86 | 49.31 | 76.98 | 26.11 | 83.49 | 82.13 | 46.69 | 47.43 | 32.08 |
| **DeepSeek-V3.23** | 49.67 | 34.30 | 28.09 | 49.97 | 77.29 | 29.46 | 72.85 | 79.48 | 44.64 | 47.99 | 32.64 |  
| **OpenAI GPT-4o** | 50.54 | 32.76 | 26.56 | 49.30 | 76.11 | 28.78 | 81.85 | 81.14 | 49.75 | 41.29 | 26.69 |
| **OpenAI GPT-o3** | 44.06 | 23.05 | 20.82 | 40.42 | 69.42 | 18.56 | 67.75 | 86.71 | 45.85 | 41.29 | 26.69 |
| **DeepSeek-R1** | 39.42 | 21.46 | 22.23 | 42.41 | 62.53 | 24.68 | 81.48 | 82.06 | 30.74 | 37.97 | 28.68 |
| **OpenAI GPT-5.1** | 46.10 | 28.24 | 23.16 | 45.43 | 68.84 | 20.88 | 73.05 | 81.32 | 44.70 | 45.39 | 29.95 |
| **Yuan3.0 Flash** | **64.47** | 49.82 | 53.79 | 57.08 | 90.93 | 59.99 | 74.40 | 87.52 | 66.31 | 68.45 | 36.40 |



*<small>
• **Long Context Tests** (D2D, QuAC, QReCC)   
• **Wikipedia Retrieval Tests** (TCQA, INSCIT)   
• **Short Text & Structured Context Tests** (CoQA, DoQA, CFQA, SQA, HDial)
</small>*

---


**5.2 Multimodal RAG Evaluation: Docmatix** 🏆

Yuan3.0 Flash leads Claude3.5, OpenAI GPT-4o, o3 and other models in the multimodal RAG benchmark Docmatix, with accuracy performance only second to GPT-5.1.

**Model Average Accuracy Comparison**

| Models | Avg. |
|--------|:---------:|
| **Qwen2.5-VL-72B-Instruct** | 59.75 |
| **InternVL3-78B** | 42.99 |
| **Claude3.5-Sonnet** | 42.55 |
| **OpenAI GPT-4o** | 56.79 |
| **OpenAI GPT-o3** | 45.57 |
| **OpenAI GPT-4V** | 60.10 |
| **OpenAI GPT-5.1** | 48.52 |
| **Yuan3.0 Flash** | **65.07** |


*<small>**Docmatix** - Evaluates the model's ability to retrieve information, correlate, and accurately answer questions across text, tables, images and other multimodal content in multi-page complex documents.</small>*

---

**5.3 Multimodal Complex Table Content Analysis Evaluation: MMTab** 🏆

Multimodal table understanding is an important application scenario in enterprise office automation. Yuan3.0 Flash achieves leading average accuracy on 15 evaluation tasks in the industry-standard multimodal complex table understanding benchmark MMTab, surpassing OpenAI's GPT-5.1.

**Model Average Accuracy Comparison**

| Models | Avg. | TABMWP | WTQ | WTQ | HiTab | TAT-QA | FeTaQAU | TabFact | InfoTabs | HiTab_T2T | Rotowire | WikiBIO | TSD_Row | TSD_Col | TCE | TCL | MCD | RCE |
|--------|:----:|:------:|:---:|:---:|:-----:|:------:|:-------:|:-------:|:--------:|:---------:|:--------:|:-------:|:-------:|:-------:|:---:|:---:|:---:|:---:|
| **Zhipu GLM-4.5V** | 52.00 | 88.21 | 77.42 | 51.52 | 62.69 | 5.25 | 89.44 | 79.48 | 5.17 | 4.48 | 2.69 | 47.40 | 89.70 | 52.74 | 50.84 | 43.47 | 50.77 | 82.79 |
| **OpenAI GPT-4V** | 29.90 | 60.50 | 48.00 | 27.50 | 32.50 | 11.04 | 45.50 | 65.60 | 2.98 | 4.23 | 1.94 | 19.00 | 38.00 | 14.36 | 27.91 | 3.50 | 48.52 | 57.14 |
| **OpenAI GPT-5.1** | 55.15 | 64.95 | 60.77 | 77.77 | 61.37 | 8.70 | 52.81 | 64.30 | 44.16 | 17.81 | 11.95 | 96.60 | 62.10 | 86.43 | 44.66 | 72.46 | 53.58 | 57.20 |
| **Yuan3.0 Flash** | 58.29 | 95.09 | 68.23 | 69.80 | 69.17 | 28.42 | 87.32 | 83.50 | 13.30 | 14.74 | 17.26 | 46.60 | 82.80 | 56.77 | 56.98 | 65.20 | 62.07 | 73.67 |

---

**5.4 Text Summarization Generation Evaluation: SummEval** 🏆

Summarization generation is a core requirement for historical information compression in intelligent agent applications. Yuan 3.0 achieves leading average accuracy in the industry-standard summarization generation benchmark SummEval across three major capabilities: lexical overlap, semantic similarity, and factual consistency, surpassing the DeepSeek-V3 large language model.

**Model Average Accuracy Comparison**


| Models | Avg. | Lexical Overlap<br>ROUGE-1 | Lexical Overlap<br>ROUGE-2 | Semantic Similarity<br>BERTScore | Factual Consistency<br>SummaC |
|--------|:---------:|:-----------:|:-----------:|:--------------:|:------------:|
| **DeepSeek-V3** | 59.28 | 25.50 | 9.20 | 86.30 | 68.20 |
| **DeepSeek-V3.2** | 51.36 | 33.30 | 11.92 | 85.61 | 41.76 |
| **Gemini-2.0-Flash** | 45.35 | 24.80 | 8.70 | 85.70 | 29.50 |
| **Claude-3.5-Sonnet** | 45.43 | 24.10 | 8.30 | 85.20 | 30.70 |
| **OpenAI GPT-4o** | 46.53 | 25.00 | 8.90 | 85.90 | 32.50 |
| **OpenAI GPT-5.1** | 49.44 | 27.48 | 10.16 | 84.63 | 40.50 |
| **Yuan3.0 Flash** | **59.31** | 51.32 | 28.32 | 89.99 | 45.34 |


---

## 6. Quick Start

**6.1 Yuan3.0 Flash Inference**

Yuan3.0 Flash supports bfloat16 and int4 quantized models. For specific usage methods, please refer to [QuickStart](vllm/README_Yuan.md)


**6.2 Data Preprocessing**

We provide data preprocessing scripts. Please refer to the data preprocessing [documentation](rlhf/docs/data_process.md).

**6.3 Model Fine-tuning Training**

We provide supervised fine-tuning scripts and reinforcement learning workflows for the Yuan3.0 Flash model. Please refer to the fine-tuning training [documentation](rlhf/docs/instruct_tuning.md) and reinforcement learning [documentation](rlhf/docs/RL_training.md).



## 7. License Agreement
The use of Yuan 3.0 code and models must comply with the [《Yuan 3.0 Model License Agreement》](https://github.com/Yuan-lab-LLM/Yuan3.0?tab=License-1-ov-file). The Yuan 3.0 model supports commercial use without requiring authorization application. Please understand and comply with the agreement, and do not use the open-source model and code, as well as derivatives generated based on the open-source project, for any purpose that may bring harm to the country and society, or for any service that has not undergone security assessment and filing.

Although we have taken measures to ensure the compliance and accuracy of the data during model training, due to the large number of model parameters and the influence of probabilistic randomness, we cannot guarantee the accuracy of the output content, and the model is easily misled by input instructions. This project does not assume responsibility for data security and public opinion risks caused by open-source models and code, or any risks and responsibilities arising from the model being misled, abused, disseminated, or improperly used. You will independently bear all risks and consequences arising from the use, copying, distribution, and modification of the model through using this open-source project.
