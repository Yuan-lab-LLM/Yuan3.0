<div align="center">
<h1>
  Yuan 3.0 Large Language Model
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
        <a href="./README_ZH.md">简体中文</a> |
        <b>English</b>
    <p>
</h4>


-----



## 0. Latest News 🎉🎉

* **[2025-12-30]** Released **Yuan3.0 Flash** VLM, a high-performance model for enterprise-level reference scenarios: 




## 1. Introduction

Inspur Information's **"Yuan 3.0" Large Language Model** is a new generation general-purpose large model designed for real-world application scenarios, focusing on **text retrieval, multimodal table analysis, complex table understanding and summary generation**, among other high-value tasks. It is suitable for knowledge-intensive and structured data-heavy application environments.

🚀⚙️ In terms of model architecture, Yuan 3.0 adopts a new generation **Sparse Mixture of Experts (MoE) architecture**, designed holistically around large-scale parameters and efficient inference requirements. It introduces proprietary dynamic routing and expert pruning mechanisms to achieve adaptive collaboration among multiple experts. While ensuring model expressiveness and inference effectiveness, it effectively reduces the scale of actually activated parameters and computational overhead, achieving a balance between performance, cost, and scalability.

📊✨ In terms of capabilities, Yuan 3.0 not only demonstrates stable performance in general language understanding and generation tasks, but has also been significantly enhanced in the following areas:

* **Retrieval-Augmented Generation (RAG) related capabilities**, adapted for long document and knowledge base scenarios;
* **Multimodal and complex table understanding**, supporting joint analysis across text and structured data;
* **Long context modeling and information summarization**, suitable for report generation and data insight tasks.

In multiple mainstream evaluations, Yuan 3.0 has demonstrated competitive performance in code generation, mathematical reasoning, scientific Q&A, and comprehensive knowledge understanding. Particularly in enterprise-critical capabilities such as text retrieval, multimodal table analysis, complex table understanding, and summary generation, it has comprehensively surpassed OpenAI GPT-4.1, making it an ideal choice for enterprise intelligent transformation.

## Model Information

| Parameter | Value |
|--------|------|
| Model Parameters | 40 billion (40B) |
| Activated Parameters | 3.7 billion |
| Training Data Volume | 2000B tokens |
| Supported Sequence Length | 128K |
| Release Date | December 30, 2025 |

## Application Scenarios

Yuan 3.0-40B is positioned as a **lightweight, cost-effective, enterprise-ready AI engine**, particularly suitable for:

- Enterprise knowledge base retrieval and Q&A
- Complex business report analysis and generation
- Long document understanding and summarization
- Multimodal data joint analysis
- Intelligent office and decision support



We have also released the <a href="https://arxiv.org/abs/2405.17976" target="_blank">**technical report**</a> for the Yuan 3.0 model, where you can find more detailed technical information and evaluation results.


<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan 3.0 Architecture Diagram

</div>



## 2. Model Downloads

**We provide download links for multiple model formats:**

|    Model     | Parameters  |   Sequence Length  |   Model Format   |         Download Link         |
| :----------: |:------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Flash |    40B    |    128K    |    HuggingFace    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan3/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan3) \|  [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan3)
