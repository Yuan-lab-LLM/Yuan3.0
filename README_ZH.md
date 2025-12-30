<img width="432" height="52" alt="image" src="https://github.com/user-attachments/assets/bbfdaa5f-2b59-43ab-b94a-95de91eaaf07" /><img width="432" height="24" alt="image" src="https://github.com/user-attachments/assets/c04e5581-a1f3-43dd-8e66-d87129296efa" /><div align="center">
<h1>
  源3.0多模态大模型
</h1>
</div>


<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> • 📎  <a href="https://arxiv.org/abs/2405.17976" target="_blank">源3.0论文</a>
</p>

<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> • 📎 <a href="https://arxiv.org/abs/2405.17976" target="_blank">源3.0论文</a> • 𝕏 <a href="https://x.com/Yuanlabai" target="_blank">X</a>
</p>



<div align="center">
  <img src="https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/Yuan2.0_logo.png?raw=true" width="60%" alt="Yuan-3.0" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/IEITYuan"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Yuan%20LLM-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://www.modelscope.cn/profile/YuanLLM"><img alt="ModelScope"
    src="https://img.shields.io/badge/👾%20ModelScope-Yuan%20LLM-6b4fbb?color=6b4fbb&logoColor=white"/></a>
  <br>
  <a href="https://x.com/Yuanlabai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-Yuanlabai-white?logo=x&logoColor=white"/></a>
  <a href="https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/wechat.jpg?raw=true"><img alt="Wechat"
    src="https://img.shields.io/badge/WeChat-Yuan%20LLM-brightgreen?logo=wechat&logoColor=white"/></a>
  <br>
  <a href="https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/LICENSE-Yuan"><img alt="Model License"
    src="https://img.shields.io/badge/Model_License-Yuan_Agreement-f5de53?&color=f5de53"/></a>
  <a href="https://arxiv.org/abs/2405.17976"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-Yuan%203.0%20Paper-b31b1b?logo=arxiv&logoColor=white"/></a>
</div>




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

Yuan 3.0 Flash 由 **YuanLab.ai 团队**开发，是一款 **40B 参数规模的多模态基础大模型**，采用稀疏混合专家（MoE）架构，单次推理仅激活约 **3.7B 参数**。通过创新的强化学习训练方法（RAPO），在提升推理准确性的同时显著降低推理 token 消耗，探索 "更少算力、更高智能" 的大模型创新路径。

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan 3.0 架构图

</div>

### 核心特性

- 🚀 **高效推理**：推理 token 消耗降低高达 75%，显著节省成本
- 🎯 **企业级优化**：针对 RAG、文档理解、表格分析等企业场景深度优化
- 🎨 **多模态支持**：支持文本、图像、表格、文档等多模态输入
- 📚 **长上下文**：支持 128K 上下文，在 "大海捞针" 测试中实现 100% 准确率
- ⚡ **即用即智能**：默认推理模式即可满足绝大多数企业场景需求

## 📊 性能表现

Yuan 3.0 Flash 在企业级 RAG、多模态检索、表格理解、摘要生成等任务上优于 GPT-5.1，同时以 40B 参数量达到 235B/671B 模型的推理精度，Token 消耗降低 50%-75%，为企业提供高性能、低成本的大模型解决方案。

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan 3.0 架构图

</div>



## 🔬 核心技术

### RAPO 强化学习算法

创新的 **Reflection-aware Adaptive Policy Optimization (RAPO)** 算法，通过反思抑制奖励机制（RIRM）：

- ✅ 识别首次得到正确答案的关键节点
- 🎯 抑制后续冗余推理行为
- 📉 准确率提升的同时，推理 token 数量减少约 75%

| 训练方法 | AIME 2024 准确率 | 平均输出长度 | MATH-500 准确率 | 平均输出长度 |
|---------|------------------|--------------|-----------------|--------------|
| Yuan3.0 Flash (40B) SFT | 31.45% | 13,656 tokens | 83.20% | 3,362 tokens |
| RL+DAPO length-penalty | 46.35% | 13,781 tokens | 89.06% | 3,974 tokens |
| **RL+RIRM** | **47.92%** | **7,505 tokens** | **89.47%** | **1,777 tokens** |







浪潮信息 **“源3.0”多模态大模型（Yuan 3.0）** 是面向真实应用场景设计的新一代通用大模型，重点覆盖 **文本检索、多模态表格分析、复杂表格理解与摘要生成** 等高价值任务，适用于知识密集型与结构化数据占比高的应用环境。

🚀⚙️ 在模型架构上，源3.0 采用新一代 **稀疏混合专家（MoE）架构**，围绕大规模参数与高效推理需求进行整体设计，并引入自研的动态路由与专家裁剪机制，实现多专家之间的自适应协同。在保证模型表达能力与推理效果的同时，有效降低实际激活参数规模与计算开销，实现性能、成本与可扩展性的平衡。

📊✨在能力层面，源3.0 不仅在通用语言理解与生成任务上表现稳定，还在以下方面进行了重点增强：

* **检索增强生成（RAG）相关能力**，适配长文档与知识库场景；
* **多模态与复杂表格理解**，支持跨文本与结构化数据的联合分析；
* **长上下文建模与信息摘要**，适用于报告生成与数据洞察类任务。

在多项主流评测中，源3.0在代码生成、数学推理、科学问答与综合知识理解等任务上展现出有竞争力的性能表现，特别在文本检索、多模态表格分析、复杂表格理解与摘要生成等企业级关键能力上已全面超越OpenAI GPT-5.1，成为企业智能化转型的理想选择。



##########3
# Yuan3.0 Flash

## 🌟 模型介绍

**Yuan3.0 Flash** 是由 **YuanLab.ai 团队** 开发的开源混合专家（MoE）多模态大语言模型，具有 **37 亿激活参数和 400 亿总参数**。该模型专门针对企业导向任务进行优化，重点覆盖 **文本检索、多模态表格分析、复杂表格理解与摘要生成** 等高价值任务，同时在通用任务上保持强大的竞争力。

## 🏗️ 架构创新

采用新一代 **稀疏混合专家（MoE）架构**，通过动态路由机制实现多专家自适应协同，并引入专家裁剪技术降低计算开销。同时提出 **RAPO**（反思感知自适应策略优化）算法，有效调节大型推理模型中的过度思考现象，提升推理效率。

## 🎯 核心能力

**企业级能力**：检索增强生成（RAG）、多模态表格理解、长上下文建模、智能摘要生成  
**通用推理能力**：数学推理、科学问答、代码生成、综合知识理解

## 📊 性能表现

在 RAG（ChatRAG）、多模态检索（Docmatix）、多模态表格理解（MMTab）、摘要生成（SummEval）等任务上 **全面超越 GPT-5.1**。精度接近 Qwen3-VL235B-A22B（235B）和 DeepSeek-R1-0528（671B），但 token 消耗仅约为其 **1/4 ~ 1/2**。





## 模型信息

| 参数项 | 数值 |
|--------|------|
| 模型参数量 | 400亿（40B） |
| 激活参数量 | 37亿 |
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


| Models | Avg All | D2D | QuAC | QReCC | CoQA | DoQA | CFQA | SQA | TCQA | HDial | INSCIT |
|--------|---------|-----|------|-------|------|------|------|-----|------|-------|--------|
| **DeepSeek-V3** | 50.5 | 31.6 | 28.9 | 49.3 | 77.0 | 26.1 | 83.5 | 82.1 | 46.7 | 47.4 | 32.1 |
| **DeepSeek-V3.23** | 49.7 | 34.3 | 28.1 | 50.0 | 77.3 | 29.5 | 72.9 | 79.5 | 44.6 | 48.0 | 32.6 |  
| **OpenAI GPT-4o** | 50.5 | 32.8 | 26.6 | 49.3 | 76.1 | 28.8 | 81.9 | 81.1 | 49.8 | 41.3 | 26.7 |
| **OpenAI GPT-o3** | 44.1 | 23.1 | 20.8 | 40.4 | 69.4 | 18.6 | 67.8 | 86.7 | 45.9 | 41.3 | 26.7 |
| **DeepSeek-R1** | 39.4 | 21.5 | 22.2 | 42.4 | 62.5 | 24.7 | 81.5 | 82.1 | 30.7 | 38.0 | 28.7 |
| **OpenAI GPT-5.1** | 46.1 | 28.2 | 23.2 | 45.4 | 68.8 | 20.9 | 73.1 | 81.3 | 44.7 | 45.4 | 30.0 |
| **Yuan3.0 Flash** | **64.5** | 49.8 | 53.8 | 57.1 | 90.9 | 60.0 | 74.4 | 87.5 | 66.3 | 68.4 | 36.4 |




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
| **Qwen2.5-VL-72B-Instruct** | 59.8 |
| **InternVL3-78B** | 43.0 |
| **Claude3.5-Sonnet** | 42.6 |
| **OpenAI GPT-4o** | 56.8 |
| **OpenAI GPT-o3** | 45.6 |
| **OpenAI GPT-4V** | 60.1 |
| **OpenAI GPT-5.1** | 48.5 |
| **Yuan3.0 Flash** | **65.1** |


*<small>**Docmatix** - 评测模型在多页复杂文档中跨文本、表格、图像等多模态内容进行信息检索、关联与准确问答的能力。</small>*

---

**3.3  多模态复杂表格内容分析评测：MMTab**🏆

多模态表格理解是企业办公重要应用场景，源3.0-1T在业界权威多模态复杂表格理解评测MMTab的15个评测任务上，实现平均精度领先OpenAI的GPT-5.1。

**模型平均精度对比**


| Models | Avg. | TABMWP | WTQ | WTQ | HiTab | TAT-QA | FeTaQAU | TabFact | InfoTabs | HiTab_T2T | Rotowire | WikiBIO | TSD_Row | TSD_Col | TCE | TCL | MCD | RCE |
|--------|:----:|:------:|:---:|:---:|:-----:|:------:|:-------:|:-------:|:--------:|:---------:|:--------:|:-------:|:-------:|:-------:|:---:|:---:|:---:|:---:|
| **智谱 GLM-4.5V** | 52.0 | 88.2 | 77.4 | 51.5 | 62.7 | 5.3 | 89.4 | 79.5 | 5.2 | 4.5 | 2.7 | 47.4 | 89.7 | 52.7 | 50.8 | 43.5 | 50.8 | 82.8 |
| **OpenAI GPT-4V** | 29.9 | 60.5 | 48.0 | 27.5 | 32.5 | 11.0 | 45.5 | 65.6 | 3.0 | 4.2 | 1.9 | 19.0 | 38.0 | 14.4 | 27.9 | 3.5 | 48.5 | 57.1 |
| **OpenAI GPT-5.1** | 55.2 | 64.9 | 60.8 | 77.8 | 61.4 | 8.7 | 52.8 | 64.3 | 44.2 | 17.8 | 11.9 | 96.6 | 62.1 | 86.4 | 44.7 | 72.5 | 53.6 | 57.2 |
| **Yuan3.0 Flash** | 58.3 | 95.1 | 68.2 | 69.8 | 69.2 | 28.4 | 87.3 | 83.5 | 13.3 | 14.7 | 17.3 | 46.6 | 82.8 | 56.8 | 57.0 | 65.2 | 62.1 | 73.7 |


---

**3.4  文本摘要生成评测：SummEval**🏆

摘要生成是智能体应用中用户历史信息压缩的核心需求，源3.0在业界权威摘要生成评测SummEval的词汇重叠、语义相似度、事实一致性3大类能力上，实现平均精度领先DeepSeek-V3大模型。

**模型平均精度对比**

| Models | Avg. | 词汇重叠<br>ROUGE-1 | 词汇重叠<br>ROUGE-2 | 语义相似度<br>BERTScore | 事实一致性<br>SummaC |
|--------|:---------:|:-----------:|:-----------:|:--------------:|:------------:|
| **DeepSeek-V3** | **59.3** | 25.5 | 9.2 | 86.3 | 68.2 |
| **DeepSeek-V3.2** | 51.4 | 33.3 | 11.9 | 85.6 | 41.8 |
| **Gemini-2.0-Flash** | 45.4 | 24.8 | 8.7 | 85.7 | 29.5 |
| **Claude-3.5-Sonnet** | 45.4 | 24.1 | 8.3 | 85.2 | 30.7 |
| **OpenAI GPT-4o** | 46.5 | 25.0 | 8.9 | 85.9 | 32.5 |
| **OpenAI GPT-5.1** | 49.4 | 27.5 | 10.2 | 84.6 | 40.5 |
| **Yuan3.0 Flash** | **59.3** | 51.3 | 28.3 | 90.0 | 45.3 |


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
