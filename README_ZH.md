<div align="center">
<h1>
  源3.0多模态基础大模型
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
  <a href="https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/YUAN3.0_FLASH-paper.pdf"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-Yuan3.0%20Paper-b31b1b?logo=arxiv&logoColor=white"/></a>


  
</div>



<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="./README.md">English</a>
    <p>
</h4>


-----



##  最近更新 🎉🎉

* **[2025-12-30]** **发布源3.0-40B多模态大模型，面向企业级应用场景的高性能模型：Yuan3.0 Flash**




## 1. 简介

Yuan 3.0 Flash 由 **YuanLab.ai 团队**开发，是一款 **40B 参数规模的多模态基础大模型**，采用稀疏混合专家（MoE）架构，单次推理仅激活约 **3.7B 参数**。通过创新的强化学习训练方法（RAPO），在提升推理准确性的同时显著降低推理 token 消耗，探索 "更少算力、更高智能" 的大模型创新路径。同时，我们发布了Yuan3.0模型的<a href="https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/YUAN3.0_FLASH-paper.pdf" target="_blank">**技术报告**</a>，可以通过论文查看更详细的技术细节与测评结果。

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-architecture.png width=80% />

Fig.1: Yuan3.0多模态大模型架构图

</div>

### 核心特性

- 🚀 **高效推理**：推理 token 消耗降低高达 75%，显著节省成本
- 🎯 **企业级优化**：针对 RAG、文档理解、表格分析等企业场景深度优化
- 🎨 **多模态支持**：支持文本、图像、表格、文档等多模态输入
- 📚 **长上下文**：支持 128K 上下文，在 "大海捞针" 测试中实现 100% 准确率
- ⚡ **即用即智能**：默认推理模式即可满足绝大多数企业场景需求

## 2. 性能表现

Yuan 3.0 Flash 在企业级 RAG、多模态检索、表格理解、摘要生成等任务上优于 GPT-5.1，同时以 40B 参数量达到 235B/671B 模型的推理精度，Token 消耗降低 50%-75%，为企业提供高性能、低成本的大模型解决方案。

<div align=center> <img src=https://github.com/Yuan-lab-LLM/Yuan3.0/blob/main/docs/Yuan3.0-benchmarks.png width=80% />

Fig.2: Yuan3.0 Flash评测结果

</div>



## 3. 核心技术

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





## 4. 模型下载

**我们提供多种模型格式的下载链接：**

|    模型     |   参数量  |  精度  |   序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :------: | :------: | :-------: |:---------------------------: |
| Yuan3.0 Flash |    400亿    |  16bit    |    128K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/Yuanlab/Yuan3.0-Flash) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Flash) \|  [始智AI]( https://www.wisemodel.cn/models/YuanLabAI/Yuan3.0-Flash)
| Yuan3.0 Flash 4bit |    400亿   |  4bit     |    128K    |    HuggingFace    | [ModelScope]( https://modelscope.cn/models/Yuanlab/Yuan3.0-Flash-int4) \| [HuggingFace]( https://huggingface.co/YuanLabAI/Yuan3.0-Flash-4bit) \|  [始智AI]( https://www.wisemodel.cn/models/YuanLab/Yuan3.0-Flash-4bit)





## 5. 测评结果

**5.1  文本类RAG评测：ChatRAG**🏆

源3.0 Flash在业界权威RAG评测ChatRAG的10个评测任务上，平均精度领先DeepSeek-V3、DeepSeek-R1等大模型。

**模型平均精度对比**


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
• **长上下文测试** (D2D、QuAC、QReCC)   
• **维基百科检索测试** (TCQA、INSCIT)   
• **短文、结构化上下文测试** (CoQA、DoQA、CFQA、SQA、HDial)
</small>*

---


**5.2  多模态RAG评测：Docmatix**🏆

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
| **OpenAI GPT-5.1** | 48.52 |
| **Yuan3.0 Flash** | **65.07** |


*<small>**Docmatix** - 评测模型在多页复杂文档中跨文本、表格、图像等多模态内容进行信息检索、关联与准确问答的能力。</small>*

---

**5.3  多模态复杂表格内容分析评测：MMTab**🏆

多模态表格理解是企业办公重要应用场景，源3.0-1T在业界权威多模态复杂表格理解评测MMTab的15个评测任务上，实现平均精度领先OpenAI的GPT-5.1。

**模型平均精度对比**


| Models | Avg. | TABMWP | WTQ | WTQ | HiTab | TAT-QA | FeTaQAU | TabFact | InfoTabs | HiTab_T2T | Rotowire | WikiBIO | TSD_Row | TSD_Col | TCE | TCL | MCD | RCE |
|--------|:----:|:------:|:---:|:---:|:-----:|:------:|:-------:|:-------:|:--------:|:---------:|:--------:|:-------:|:-------:|:-------:|:---:|:---:|:---:|:---:|
| **Zhipu GLM-4.5V** | 52.00 | 88.21 | 77.42 | 51.52 | 62.69 | 5.25 | 89.44 | 79.48 | 5.17 | 4.48 | 2.69 | 47.40 | 89.70 | 52.74 | 50.84 | 43.47 | 50.77 | 82.79 |
| **OpenAI GPT-4V** | 29.90 | 60.50 | 48.00 | 27.50 | 32.50 | 11.04 | 45.50 | 65.60 | 2.98 | 4.23 | 1.94 | 19.00 | 38.00 | 14.36 | 27.91 | 3.50 | 48.52 | 57.14 |
| **OpenAI GPT-5.1** | 55.15 | 64.95 | 60.77 | 77.77 | 61.37 | 8.70 | 52.81 | 64.30 | 44.16 | 17.81 | 11.95 | 96.60 | 62.10 | 86.43 | 44.66 | 72.46 | 53.58 | 57.20 |
| **Yuan3.0 Flash** | 58.29 | 95.09 | 68.23 | 69.80 | 69.17 | 28.42 | 87.32 | 83.50 | 13.30 | 14.74 | 17.26 | 46.60 | 82.80 | 56.77 | 56.98 | 65.20 | 62.07 | 73.67 |


---

**3.4  文本摘要生成评测：SummEval**🏆

摘要生成是智能体应用中用户历史信息压缩的核心需求，源3.0在业界权威摘要生成评测SummEval的词汇重叠、语义相似度、事实一致性3大类能力上，实现平均精度领先DeepSeek-V3大模型。

**模型平均精度对比**

| Models | Avg. | 词汇重叠<br>ROUGE-1 | 词汇重叠<br>ROUGE-2 | 语义相似度<br>BERTScore | 事实一致性<br>SummaC |
|--------|:---------:|:-----------:|:-----------:|:--------------:|:------------:|
| **DeepSeek-V3** | 59.28 | 25.50 | 9.20 | 86.30 | 68.20 |
| **DeepSeek-V3.2** | 51.36 | 33.30 | 11.92 | 85.61 | 41.76 |
| **Gemini-2.0-Flash** | 45.35 | 24.80 | 8.70 | 85.70 | 29.50 |
| **Claude-3.5-Sonnet** | 45.43 | 24.10 | 8.30 | 85.20 | 30.70 |
| **OpenAI GPT-4o** | 46.53 | 25.00 | 8.90 | 85.90 | 32.50 |
| **OpenAI GPT-5.1** | 49.44 | 27.48 | 10.16 | 84.63 | 40.50 |
| **Yuan3.0 Flash** | **59.31** | 51.32 | 28.32 | 89.99 | 45.34 |

---

## 6. 快速开始

**6.1 Yuan3.0 Flash推理**

Yuan3.0 Flash 支持 bfloat16 和 int4 量化模型，具体使用方法，详见[QuickStart](vllm/README_Yuan.md)


**6.2 数据预处理**

我们提供了数据预处理的脚本，参考数据预处理[说明文档](脚本rlhf/docs/data_process.md)。

**6.3 模型微调训练**

我们提供了Yuan3.0 Flash模型的监督微调脚本与强化学习流程，详见微调训练[文档](rlhf/docs/instruct_tuning.md)与强化学习说明[文档](rlhf/docs/RL_training.md)。



##  5. 协议声明
使用源3.0代码及模型需遵循[《源3.0模型许可协议》](https://github.com/Yuan-lab-LLM/Yuan3.0?tab=License-1-ov-file)，源3.0模型支持商用，不需要申请授权，请您了解并遵循，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。


