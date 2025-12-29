<div align="center">
<h1>
  源3.0多模态大模型
</h1>
</div>


<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> • 📎  <a href="https://arxiv.org/abs/2405.17976" target="_blank">源3.0论文</a>
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
        <b>简体中文</b> |
        <a href="./README.md">English</a>
    <p>
</h4>


-----



##  0. Latest News 🎉🎉

* **[2025-12-30]** **发布源3.0-40B多模态大模型，面向企业级引用场景的高性能模型：Yuan3.0 Flash**




## 1. Introduction

浪潮信息 **“源3.0”多模态大模型（Yuan 3.0）** 是面向真实应用场景设计的新一代通用大模型，重点覆盖 **文本检索、多模态表格分析、复杂表格理解与摘要生成** 等高价值任务，适用于知识密集型与结构化数据占比高的应用环境。

🚀⚙️ 在模型架构上，源3.0 采用新一代 **稀疏混合专家（MoE）架构**，围绕大规模参数与高效推理需求进行整体设计，并引入自研的动态路由与专家裁剪机制，实现多专家之间的自适应协同。在保证模型表达能力与推理效果的同时，有效降低实际激活参数规模与计算开销，实现性能、成本与可扩展性的平衡。

📊✨在能力层面，源3.0 不仅在通用语言理解与生成任务上表现稳定，还在以下方面进行了重点增强：

* **检索增强生成（RAG）相关能力**，适配长文档与知识库场景；
* **多模态与复杂表格理解**，支持跨文本与结构化数据的联合分析；
* **长上下文建模与信息摘要**，适用于报告生成与数据洞察类任务。

在多项主流评测中，源3.0在代码生成、数学推理、科学问答与综合知识理解等任务上展现出有竞争力的性能表现，特别在文本检索、多模态表格分析、复杂表格理解与摘要生成等企业级关键能力上已全面超越OpenAI GPT-5.1，成为企业智能化转型的理想选择。

## 模型信息

| 参数项 | 数值 |
|--------|------|
| 模型参数量 | 400亿（40B） |
| 激活参数量 | 37亿 |
| 训练数据量 | 2000B tokens |
| 支持序列长度 | 128K |
| 发布时间 | 2025年12月30日 |

## 应用场景

源3.0-40B（Yuan3.0 Flash） 定位为**轻量高性价比的企业即用型AI引擎**，特别适合：

- 企业知识库检索与问答
- 复杂业务报表分析与生成
- 长文档理解与摘要
- 多模态数据联合分析
- 智能办公与决策支持



同时，我们发布了Yuan3.0模型的<a href="https://arxiv.org/abs/2405.17976" target="_blank">**技术报告**</a>，可以通过论文查看更详细的技术细节与测评结果。


<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan 3.0 架构图

</div>



##  2. Model Downloads

**我们提供多种模型格式的下载链接：**

|    模型     |   参数量  |  序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Flash |    400亿    |    128K    |    HuggingFace    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan3/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan3) \|  [始智AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan3)


##  3. Evaluation Results


**3.1  文本类RAG评测：ChatRAG**🏆

源3.0 Flash在业界权威RAG评测ChatRAG的10个评测任务上，平均精度领先DeepSeek-V3、DeepSeek-R1等大模型。

**模型平均精度对比**

| Models | Avg. | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
|--------|:----------:|:------:|:------:|:-------:|:------:|:------:|:------:|:-----:|:------:|:-------:|:--------:|
| | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % | Acc. % |
| **DeepSeek-V3** | 50.47 | 31.59 | 28.86 | 49.31 | 76.98 | 26.11 | 83.49 | 82.13 | 46.69 | 47.43 | 32.08 |
| **DeepSeek-V3.2** | 49.67 | 34.30 | 28.09 | 49.97 | 77.29 | 29.46 | 72.85 | 79.48 | 44.64 | 47.99 | 32.64 |
| **OpenAI GPT-4o** | 50.54 | 32.76 | 26.56 | 49.3 | 76.11 | 28.78 | 81.85 | 81.14 | 49.75 | 41.29 | 26.69 |
| **OpenAI GPT-o3** | 44.06 | 23.05 | 20.82 | 40.42 | 69.42 | 18.56 | 67.75 | 86.71 | 45.85 | 41.29 | 26.69 |
| **DeepSeek-R1** | 39.42 | 21.46 | 22.23 | 42.41 | 62.53 | 24.68 | 81.48 | 82.06 | 30.74 | 37.97 | 28.68 |
| **Yuan3.0 Flash** | **63.97** | **48.98** | **53.07** | **56.04** | **91.24** | **58.97** | **74.33** | **85.35** | **66.38** | **69.54** | **35.80** |


*<small>
• **长上下文测试** (D2D、QuAC、QReCC)   
• **维基百科检索测试** (TCQA、INSCIT)   
• **短文、结构化上下文测试** (CoQA、DoQA、CFQA、SQA、HDial)
</small>*

---


**3.2  多模态RAG评测：Docmatix**🏆

Yuan3.0 Flash 在多模态RAG评测Docmatix中领先Claude3.5、OpenAI GPT-4o 、o3等模型，精度表现仅次于GPT-5.1。

**模型平均精度对比**

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


*<small>**Docmatix** - 评测模型在多页复杂文档中跨文本、表格、图像等多模态内容进行信息检索、关联与准确问答的能力。</small>*

---

**3.3  多模态复杂表格内容分析评测：MMTab**🏆

多模态表格理解是企业办公重要应用场景，源3.0-1T在业界权威多模态复杂表格理解评测MMTab的15个评测任务上，实现平均精度领先OpenAI的GPT-5.1。

**模型平均精度对比**

|       Models      |  Avg.  | TABMWP | WTQ | WTQ | HiTab | TAT-QA | FeTaQAU | TabFact | InfoTabs | HiTab_T2T | Rotowire | WikiBIO | TSD_Row| TSD_Col | TCE | TCL | MCD | RCE |
| ----------------- | :--------: | :--------------: | :-----------: | :-----------: | :-------------: | :--------------: | :-------------: | :---------------: | :----------------: | :----------------: | :---------------: | :--------------: | :-----------: | :-----------: | :-----------: | :-----------: | :---------: | :-------------: |
| **智谱 GLM-4.5V**       |     52.00 |           88.21 |        77.42 |        77.42 |          51.52 |           62.69 |           5.25 |            89.44 |             79.48 |              5.17 |             4.48 |            2.69 |         47.4 |         89.7 |        52.74 |        50.84 |      43.47 |          50.77 |
| **OpenAI GPT-4V**     |     29.90 |           60.50 |        48.00 |        48.00 |          27.50 |           32.50 |          11.04 |            45.50 |             65.60 |              2.98 |             4.23 |            1.94 |        19.00 |        38.00 |        14.36 |        27.91 |       3.50 |          48.52 |
| **OpenAI GPT-5.1**    |     76.73 |           98.15 |        95.70 |        93.59 |          96.37 |           17.05 |          88.86 |            85.62 |             40.22 |             17.41 |            11.53 |           96.60 |        98.80 |        86.43 |        85.62 |        99.91 |      95.20 |          97.27 |
| **Yuan3.0 Flash** | **77.96** |       **90.96** |    **97.88** |    **98.86** |      **98.45** |       **15.76** |      **93.78** |        **89.59** |         **13.21** |         **42.80** |        **27.05** |       **91.00** |    **91.00** |    **95.91** |    **99.32** |    **89.16** |  **95.28** |      **95.31** |

---

**3.4  文本摘要生成评测：SummEval**🏆

摘要生成是智能体应用中用户历史信息压缩的核心需求，源3.0在业界权威摘要生成评测SummEval的词汇重叠、语义相似度、事实一致性3大类能力上，实现平均精度领先DeepSeek-V3大模型。

**模型平均精度对比**

| Models | Avg. | 词汇重叠<br>ROUGE-1 | 词汇重叠<br>ROUGE-2 | 语义相似度<br>BERTScore | 事实一致性<br>SummaC |
|--------|:---------:|:-----------:|:-----------:|:--------------:|:------------:|
| **DeepSeek-V3** | 59.28 | 25.5 | 9.2 | 86.3 | 68.2 |
| **DeepSeek-V3.2** | 51.36 | 33.30 | 11.92 | 85.61 | 41.76 |
| **Gemini-2.0-Flash** | 45.35 | 24.8 | 8.7 | 85.7 | 29.5 |
| **Claude-3.5-Sonnet** | 45.43 | 24.1 | 8.3 | 85.2 | 30.7 |
| **OpenAI GPT-4o** | 46.53 | 25.0 | 8.9 | 85.9 | 32.5 |
| **OpenAI o1** | 46.50 | 24.0 | 8.1 | 85.1 | 34.0 |
| **Yuan3.0 Flash** | **61.07** | **46.8** | **28.5** | **89.0** | **53.2** |

---


**3.5  多模态表格、文档与OCR评测b**🏆

多模态表格、文档与OCR评测，主要用于衡量模型在真实业务场景中对文字、图表和文档的综合理解能力。Yuan3.0 Flash在业界权威评测OCRBench、ChartQA、ChartMuseum与DocVQA中表现出色。

*<small>
• **OCRBench** - 评测模型在复杂图像中的文字识别与理解能力。   
• **ChartQA** - 评测模型理解和回答图表（柱状图、折线图、饼图等）相关问题的能力。   
• **ChartMuseum** - 评测模型对多类型真实图表的视觉理解与问答能力。
</small>*




**Non-Thinking 平均精度对比**
| Models | 类型 | OCRBench | ChartQA | DocVQA |
|--------|:------:|:----------:|:---------:|:--------:|
| **GLM-4.6V-Flash** | - | - | 84.7 | - |
| **Qwen2.5-VL-72B** | - | - | 89.5 | 96.4 |
| **InternVL3-78B** | - | - | 89.7 | 95.4 |
| **OpenAI GPT-5** | minimal | 78.7 | 59.1 | 89.6 |
| **Qwen3-32B-VL** | Non-Thinking | 96.9 | 88.5 | 89.5 |
| **Yuan3.0 Flash** | **Non-Thinking** | **70.6** | **87.7** | **90.1** |


**Thinking平均精度对比**
| Models | 类型 | OCRBench | ChartQA | DocVQA |
|--------|:------:|:----------:|:---------:|:--------:|
| **GLM-4.6V** | Thinking | 86.5 | - | - |
| **Qwen3-VL-235B-A22B-Thinking** | Thinking | 87.5 | 90.3 | 96.5 |
| **Kimi-VL-A3B-Thinking-2506** | Thinking | 86.9 | - | - |
| **OpenAI GPT-5** | high | 81.0 | 59.7 | 91.5 |
| **Qwen3-32B-VL** | Thinking | 96.1 | 89.0 | 85.5 |
| **Yuan3.0 Flash** | **Thinking** | **71.0** | **90.2** | **90.1** |


---
