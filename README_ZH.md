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





