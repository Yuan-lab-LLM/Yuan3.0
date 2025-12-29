<div align="center">
<h1>
  Yuan 3.0 Multimodal Large Language Model
</h1>
</div>

<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> • 📎  <a href="https://arxiv.org/abs/2405.17976" target="_blank">Yuan 3.0 Paper</a>
</p>

<div align="center">
    
  <a href="code_license">
    <img alt="Code License" src="https://img.shields.io/badge/Apache%202.0%20-green?style=flat&label=Code%20License&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0-MoE%3Ftab%3DApache-2.0-1-ov-file"/>
  </a>
  <a href="model_license">
    <img alt="Model License" src="https://img.shields.io/badge/Yuan3.0%20License-blue?style=flat&logoColor=blue&label=Model%20License&color=blue&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0%2Fblob%2Fmain%2FLICENSE-Yuan" />
  </a>

</div>

<h4 align="center">
    <p>
        <a href="./README_CN.md">简体中文</a> |
        <b>English</b>
    <p>
</h4>

-----

## 0. Latest News 🎉🎉

* **[2025-12-30]** **Released Yuan 3.0-40B Multimodal Large Language Model, a high-performance model for enterprise-grade citation scenarios: Yuan3.0 Flash**

## 1. Introduction

Inspur Information's **"Yuan 3.0" Multimodal Large Language Model (Yuan 3.0)** is a new generation general-purpose large model designed for real-world application scenarios, focusing on **text retrieval, multimodal table analysis, complex table understanding, and summary generation**, suitable for knowledge-intensive and structured data-heavy application environments.

🚀⚙️ In terms of model architecture, Yuan 3.0 adopts a new generation **Sparse Mixture of Experts (MoE) architecture**, designed around large-scale parameters and efficient inference requirements, and introduces self-developed dynamic routing and expert pruning mechanisms to achieve adaptive collaboration among multiple experts. While ensuring model expressiveness and inference effectiveness, it effectively reduces the actual activated parameter scale and computational overhead, achieving a balance between performance, cost, and scalability.

📊✨ At the capability level, Yuan 3.0 not only performs stably in general language understanding and generation tasks, but has also been specifically enhanced in the following areas:

* **Retrieval-Augmented Generation (RAG) related capabilities**, adapted for long document and knowledge base scenarios;
* **Multimodal and complex table understanding**, supporting joint analysis across text and structured data;
* **Long context modeling and information summarization**, suitable for report generation and data insight tasks.

In multiple mainstream evaluations, Yuan 3.0 demonstrates competitive performance in code generation, mathematical reasoning, scientific Q&A, and comprehensive knowledge understanding. Particularly in enterprise-critical capabilities such as text retrieval, multimodal table analysis, complex table understanding, and summary generation, it has comprehensively surpassed OpenAI GPT-5.1, making it an ideal choice for enterprise intelligent transformation.

## Model Information

| Parameter | Value |
|--------|------|
| Model Parameters | 40 Billion (40B) |
| Activated Parameters | 3.7 Billion |
| Supported Sequence Length | 128K |
| Release Date | December 30, 2025 |

## Application Scenarios

Yuan 3.0-40B (Yuan3.0 Flash) is positioned as a **lightweight, cost-effective enterprise-ready AI engine**, particularly suitable for:

- Enterprise knowledge base retrieval and Q&A
- Complex business report analysis and generation
- Long document understanding and summarization
- Multimodal data joint analysis
- Intelligent office and decision support

At the same time, we have released the <a href="https://arxiv.org/abs/2405.17976" target="_blank">**technical report**</a> for the Yuan 3.0 model, where you can view more detailed technical details and evaluation results through the paper.

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan 3.0 Architecture

</div>

## 2. Model Downloads

**We provide download links for various model formats:**

|    Model     |   Parameters  |  Sequence Length  |   Format   |         Download Links         |
| :----------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Flash |    40B    |    128K    |    HuggingFace    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan3/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan3) \|  [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan3)

## 3. Evaluation Results

**3.1  Text-based RAG Evaluation: ChatRAG**🏆

Yuan 3.0 Flash leads large models such as DeepSeek-V3 and DeepSeek-R1 in average accuracy across 10 evaluation tasks in the authoritative industry RAG evaluation ChatRAG.

**Model Average Accuracy Comparison**

| Models | Avg. | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
|--------|:----------:|:------:|:------:|:-------:|:------:|:------:|:------:|:-----:|:------:|:-------:|:--------:|
| | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % |
| **DeepSeek-V3** | 50.47 | 31.59 | 28.86 | 49.31 | 76.98 | 26.11 | 83.49 | 82.13 | 46.69 | 47.43 | 32.08 |
| **DeepSeek-V3.2** | 49.67 | 34.30 | 28.09 | 49.97 | 77.29 | 29.46 | 72.85 | 79.48 | 44.64 | 47.99 | 32.64 |
| **OpenAI GPT-4o** | 50.54 | 32.76 | 26.56 | 49.3 | 76.11 | 28.78 | 81.85 | 81.14 | 49.75 | 41.29 | 26.69 |
| **OpenAI GPT-o3** | 44.06 | 23.05 | 20.82 | 40.42 | 69.42 | 18.56 | 67.75 | 86.71 | 45.85 | 41.29 | 26.69 |
| **DeepSeek-R1** | 39.42 | 21.46 | 22.23 | 42.41 | 62.53 | 24.68 | 81.48 | 82.06 | 30.74 | 37.97 | 28.68 |
| **Yuan3.0 Flash** | **63.97** | **48.98** | **53.07** | **56.04** | **91.24** | **58.97** | **74.33** | **85.35** | **66.38** | **69.54** | **35.80** |


- **Long Context Tests** (D2D, QuAC, QReCC)   
- **Wikipedia Retrieval Tests** (TCQA, INSCIT)   
- **Short Text and Structured Context Tests** (CoQA, DoQA, CFQA, SQA, HDial)


---

**3.2  Multimodal RAG Evaluation: Docmatix**🏆

Yuan3.0 Flash leads models such as Claude3.5, OpenAI GPT-4o, and o3 in the multimodal RAG evaluation Docmatix, with accuracy performance second only to GPT-5.1.

**Model Average Accuracy Comparison**

| Models | Avg. |
|--------|:---------:|
| **Qwen2.5-VL-72B-Instruct** | 59.75 |
| **InternVL3-78B** | 42.99 |
| **Claude3.5-Sonnet** | 42.55 |
| **OpenAI GPT-4o** | 56.79 |
| **OpenAI GPT-o3** | 45.57 |
| **OpenAI GPT-4V** | 60.10 |
| **OpenAI GPT-5.1** | 66.83 |
| **Yuan3.0 Flash** | **66.58** |

**Docmatix** - Evaluates the model's ability to retrieve, correlate, and accurately answer questions across text, tables, images, and other multimodal content in multi-page complex documents.

---

**3.3  Multimodal Complex Table Content Analysis Evaluation: MMTab**🏆

Multimodal table understanding is an important application scenario for enterprise office work. Yuan 3.0-1T achieves leading average accuracy over OpenAI's GPT-5.1 across 15 evaluation tasks in the authoritative industry multimodal complex table understanding evaluation MMTab.

**Model Average Accuracy Comparison**

|       Models      |  Avg.  | TABMWP | WTQ | WTQ | HiTab | TAT-QA | FeTaQAU | TabFact | InfoTabs | HiTab_T2T | Rotowire | WikiBIO | TSD_Row| TSD_Col | TCE | TCL | MCD | RCE |
| ----------------- | :--------: | :--------------: | :-----------: | :-----------: | :-------------: | :--------------: | :-------------: | :---------------: | :----------------: | :----------------: | :---------------: | :--------------: | :-----------: | :-----------: | :-----------: | :-----------: | :---------: | :-------------: |
| **Zhipu GLM-4.5V**       |     52.00 |           88.21 |        77.42 |        77.42 |          51.52 |           62.69 |           5.25 |            89.44 |             79.48 |              5.17 |             4.48 |            2.69 |         47.4 |         89.7 |        52.74 |        50.84 |      43.47 |          50.77 |
| **OpenAI GPT-4V**     |     29.90 |           60.50 |        48.00 |        48.00 |          27.50 |           32.50 |          11.04 |            45.50 |             65.60 |              2.98 |             4.23 |            1.94 |        19.00 |        38.00 |        14.36 |        27.91 |       3.50 |          48.52 |
| **OpenAI GPT-5.1**    |     76.73 |           98.15 |        95.70 |        93.59 |          96.37 |           17.05 |          88.86 |            85.62 |             40.22 |             17.41 |            11.53 |           96.60 |        98.80 |        86.43 |        85.62 |        99.91 |      95.20 |          97.27 |
| **Yuan3.0 Flash** | **77.96** |       **90.96** |    **97.88** |    **98.86** |      **98.45** |       **15.76** |      **93.78** |        **89.59** |         **13.21** |         **42.80** |        **27.05** |       **91.00** |    **91.00** |    **95.91** |    **99.32** |    **89.16** |  **95.28** |      **95.31** |

---

**3.4  Text Summarization Generation Evaluation: SummEval**🏆

Summarization generation is a core requirement for user history information compression in agent applications. Yuan 3.0 achieves leading average accuracy over the DeepSeek-V3 large model in the three major capability categories of lexical overlap, semantic similarity, and factual consistency in the authoritative industry summarization generation evaluation SummEval.

**Model Average Accuracy Comparison**

| Models | Avg. | Lexical Overlap<br>ROUGE-1 | Lexical Overlap<br>ROUGE-2 | Semantic Similarity<br>BERTScore | Factual Consistency<br>SummaC |
|--------|:---------:|:-----------:|:-----------:|:--------------:|:------------:|
| **DeepSeek-V3** | 59.28 | 25.5 | 9.2 | 86.3 | 68.2 |
| **DeepSeek-V3.2** | 51.36 | 33.30 | 11.92 | 85.61 | 41.76 |
| **Gemini-2.0-Flash** | 45.35 | 24.8 | 8.7 | 85.7 | 29.5 |
| **Claude-3.5-Sonnet** | 45.43 | 24.1 | 8.3 | 85.2 | 30.7 |
| **OpenAI GPT-4o** | 46.53 | 25.0 | 8.9 | 85.9 | 32.5 |
| **OpenAI o1** | 46.50 | 24.0 | 8.1 | 85.1 | 34.0 |
| **Yuan3.0 Flash** | **61.07** | **46.8** | **28.5** | **89.0** | **53.2** |

---

**3.5  Multimodal Table, Document, and OCR Evaluation**🏆

Multimodal table, document, and OCR evaluation is primarily used to measure the model's comprehensive understanding ability of text, charts, and documents in real business scenarios. Yuan3.0 Flash performs excellently in the authoritative industry evaluations OCRBench, ChartQA, ChartMuseum, and DocVQA.

- **OCRBench** - Evaluates the model's text recognition and understanding ability in complex images.   
- **ChartQA** - Evaluates the model's ability to understand and answer questions related to charts (bar charts, line charts, pie charts, etc.).   
- **ChartMuseum** - Evaluates the model's visual understanding and Q&A ability for multiple types of real charts.


**Non-Thinking Average Accuracy Comparison**
| Models | Type | OCRBench | ChartQA | DocVQA |
|--------|:------:|:----------:|:---------:|:--------:|
| **GLM-4.6V-Flash** | - | - | 84.7 | - |
| **Qwen2.5-VL-72B** | - | - | 89.5 | 96.4 |
| **InternVL3-78B** | - | - | 89.7 | 95.4 |
| **OpenAI GPT-5** | minimal | 78.7 | 59.1 | 89.6 |
| **Qwen3-32B-VL** | Non-Thinking | 96.9 | 88.5 | 89.5 |
| **Yuan3.0 Flash** | **Non-Thinking** | **70.6** | **87.7** | **90.1** |

**Thinking Average Accuracy Comparison**
| Models | Type | OCRBench | ChartQA | DocVQA |
|--------|:------:|:----------:|:---------:|:--------:|
| **GLM-4.6V** | Thinking | 86.5 | - | - |
| **Qwen3-VL-235B-A22B-Thinking** | Thinking | 87.5 | 90.3 | 96.5 |
| **Kimi-VL-A3B-Thinking-2506** | Thinking | 86.9 | - | - |
| **OpenAI GPT-5** | high | 81.0 | 59.7 | 91.5 |
| **Qwen3-32B-VL** | Thinking | 96.1 | 89.0 | 85.5 |
| **Yuan3.0 Flash** | **Thinking** | **71.0** | **90.2** | **90.1** |

---
