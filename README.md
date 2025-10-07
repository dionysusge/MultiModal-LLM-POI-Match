# POI匹配系统

**Author:** Dionysus  
**Contact:** WeChat: gzw1546484791 | Email: dionysusge58@gmail.com

## 项目简介

POI匹配系统是一个基于深度学习的地理位置兴趣点（Point of Interest）匹配解决方案。该系统能够智能匹配来自不同地图服务商（百度地图、高德地图）的POI数据，通过多模态特征融合和大语言模型增强技术，实现高精度的POI实体对齐。

## 核心特性

### 🚀 多模态特征融合
- **文本特征**：使用Sentence-BERT提取POI名称和描述的语义特征
- **地理特征**：基于Geohash编码的位置信息表示
- **图神经网络**：GAT（Graph Attention Network）捕获POI间的空间关系
- **LLM增强**：支持多种大语言模型进行文本特征增强

### 🎯 智能匹配算法
- **交互特征提取**：计算POI对之间的相似度、距离等交互特征
- **深度学习分类器**：基于MLP和LightGBM的混合分类模型
- **批量处理**：支持大规模POI数据的高效处理

### 🔧 灵活配置
- **多种LLM支持**：HuggingFace模型、OLLAMA本地模型
- **调试模式**：支持小数据集快速测试
- **缓存机制**：智能缓存中间结果，提升处理效率

## 项目结构

```
POI_delivery/
├── data/                           # 数据目录
│   ├── bd_chinese_data.csv        # 百度地图POI数据
│   ├── gd_chinese_data.csv        # 高德地图POI数据
│   └── match*.csv                 # 标注匹配数据
├── models/                         # 模型目录
│   ├── DialoGPT-small/            # 对话生成模型
│   ├── Qwen3-0.6B/               # 千问模型
│   ├── paraphrase-multilingual-MiniLM-L12-v2/  # 多语言句子编码器
│   ├── llm_text_enhancer.py      # LLM文本增强模块
│   └── multi_modal_fusion.py     # 多模态融合模块
├── outputs/                        # 输出目录
│   ├── cache/                     # 缓存文件
│   └── *.pkl                      # 特征和模型文件
├── scripts/                        # 辅助脚本
├── utils/                          # 工具模块
├── main.py           # 主要训练脚本
├── modal_fusion_mlp.py           # 模态融合MLP模型
└── requirements.txt               # 依赖包列表
```

## 主要脚本说明

### main.py
**功能**：POI匹配系统的主要训练脚本
- 支持多种LLM模型（HuggingFace、OLLAMA）
- 集成文本嵌入、GAT图神经网络、地理编码
- 提供完整的训练和评估流程
- 支持调试模式和缓存机制

**主要特性**：
- 多模态特征提取（文本、地理、图结构）
- LLM文本增强
- GAT图注意力网络
- LightGBM分类器训练
- 智能缓存管理

### modal_fusion_mlp.py
**功能**：优化的交互特征融合MLP模型
- 实现高效的特征交互计算
- 使用残差连接和批归一化
- 支持余弦退火学习率调度
- 提供详细的模型评估指标

**主要特性**：
- 优化的交互特征提取器
- 残差块设计
- 批归一化稳定训练
- 多层感知机分类器

## 环境要求

### 系统要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 内存：建议16GB+
- 存储：建议10GB+可用空间

### 核心依赖
```
torch>=2.0.0
torch-geometric>=2.3.0
sentence-transformers>=2.2.0
transformers>=4.20.0
lightgbm>=3.3.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
```

## 安装指南

### 1. 克隆项目
```bash
git clone <repository-url>
cd POI_delivery
```

### 2. 创建虚拟环境
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖
```bash
# 使用清华源加速安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 4. 准备数据
将POI数据文件放置在 `data/` 目录下：
- `bd_chinese_data.csv`：百度地图POI数据
- `gd_chinese_data.csv`：高德地图POI数据  
- `match2.csv`：标注的匹配对数据

### 5. 下载预训练模型
确保 `models/` 目录下包含所需的预训练模型文件。

## 使用方法

### 快速开始

#### 1. 调试模式训练
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 使用HuggingFace模型进行调试
python main.py --debug --debug-limit 1000 --model-type huggingface --model-name models/DialoGPT-small

# 使用OLLAMA模型进行调试
python main.py --debug --debug-limit 1000 --model-type ollama --model-name qwen2.5:0.5b
```

#### 2. 完整数据训练
```bash
# 使用HuggingFace模型
python main.py --model-type huggingface --model-name models/Qwen3-0.6B

# 使用OLLAMA模型
python main.py --model-type ollama --model-name qwen2.5:3b
```

#### 3. 模态融合MLP训练
```bash
python modal_fusion_mlp.py
```

### 命令行参数

#### main.py 参数
```bash
--model-type        # LLM模型类型 [huggingface|ollama] (默认: huggingface)
--model-name        # 模型名称或路径 (默认: models/DialoGPT-small)
--ollama-url        # OLLAMA服务URL (默认: http://localhost:11434)
--debug             # 启用调试模式
--debug-limit       # 调试模式数据限制 (默认: 1000)
--device            # 计算设备 [cuda|cpu] (默认: cuda)
```

#### 使用示例
```bash
# 基础训练
python main.py

# 调试模式，使用OLLAMA模型
python main.py --model-type ollama --model-name qwen2.5:0.5b --debug --debug-limit 500

# 使用CPU训练
python main.py --device cpu --debug

# 使用本地HuggingFace模型
python main.py --model-type huggingface --model-name models/Qwen3-0.6B
```

## 输出文件

训练完成后，系统会在 `outputs/` 目录生成以下文件：

### 模型文件
- `gat_classifier_model.pkl`：训练好的分类器模型
- `final_matched_pairs.csv`：最终的POI匹配结果

### 特征文件
- `text_emb_*.pkl`：文本嵌入特征
- `gat_emb_*.pkl`：GAT图嵌入特征
- `enhanced_features_*.pkl`：增强特征向量

### 评估报告
- `model_evaluation_report.txt`：详细的模型评估报告

## 性能优化

### 1. 内存优化
- 使用批处理减少内存占用
- 智能缓存机制避免重复计算
- 支持调试模式快速验证

### 2. 计算优化
- GPU加速训练（支持CUDA）
- 多进程数据加载
- 高效的特征提取算法

### 3. 模型优化
- 残差连接提升梯度流
- 批归一化稳定训练
- 学习率调度优化收敛

## 故障排除

### 常见问题

#### 1. CUDA内存不足
```bash
# 使用CPU训练
python main.py --device cpu

# 减少批处理大小
# 修改脚本中的 TEXT_EMBEDDING_BATCH_SIZE 参数
```

#### 2. 模型文件缺失
```bash
# 检查models目录结构
ls models/
# 确保包含所需的预训练模型文件
```

#### 3. 依赖包冲突
```bash
# 重新创建虚拟环境
rm -rf .venv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 4. OLLAMA连接失败
```bash
# 检查OLLAMA服务状态
curl http://localhost:11434/api/tags

# 修改OLLAMA URL
python main.py --ollama-url http://your-ollama-server:11434
```

## 开发指南

### 扩展开发
- 新增LLM模型支持：修改 `models/llm_text_enhancer.py`
- 新增特征提取器：扩展 `models/multi_modal_fusion.py`
- 新增分类器：修改训练脚本的分类器部分


## 更新日志

### v1.0.0
- 初始版本发布
- 支持多模态POI匹配
- 集成LLM文本增强
- 提供完整的训练和评估流程

---

如有问题或建议，请联系作者：
- WeChat: gzw1546484791
- Email: dionysusge58@gmail.com
