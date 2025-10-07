"""
DEF编码器: 将POI的def数据列转换为特征向量

作者: Dionysus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefEncoder(nn.Module):
    """
    DEF数据编码器，将def列的信息转换为密集的特征向量
    
    功能:
        1. 处理分类型def数据
        2. 处理数值型def数据
        3. 处理混合型def数据
        4. 生成统一的特征表示
    """
    
    def __init__(self, 
                 def_vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 device: str = 'cuda'):
        """
        初始化DEF编码器
        
        参数:
            def_vocab_size: def词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            device: 计算设备
        """
        super(DefEncoder, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # DEF嵌入层
        self.def_embedding = nn.Embedding(def_vocab_size, embedding_dim).to(device)
        
        # 特征处理网络
        self.feature_processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        ).to(device)
        
        # 注意力机制 - 用于处理多个def值
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        # 位置编码 - 用于序列型def数据
        self.positional_encoding = nn.Parameter(
            torch.randn(100, embedding_dim).to(device)  # 最多支持100个def值
        )
    
    def forward(self, def_indices: torch.Tensor, def_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            def_indices: def索引张量 [batch_size, max_def_length]
            def_mask: 掩码张量，标识有效的def值 [batch_size, max_def_length]
            
        返回:
            编码后的特征 [batch_size, output_dim]
        """
        batch_size, seq_len = def_indices.shape
        
        # 1. 嵌入层
        embedded = self.def_embedding(def_indices)  # [batch_size, seq_len, embedding_dim]
        
        # 2. 添加位置编码
        if seq_len <= self.positional_encoding.size(0):
            pos_encoding = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            embedded = embedded + pos_encoding
        
        # 3. 应用注意力机制
        if def_mask is not None:
            # 将mask转换为注意力掩码格式
            attn_mask = ~def_mask.bool()  # True表示需要被掩码的位置
        else:
            attn_mask = None
        
        attended, _ = self.attention(embedded, embedded, embedded, key_padding_mask=attn_mask)
        
        # 4. 池化操作 - 使用平均池化
        if def_mask is not None:
            # 掩码平均池化
            mask_expanded = def_mask.unsqueeze(-1).expand_as(attended)
            masked_attended = attended * mask_expanded
            pooled = masked_attended.sum(dim=1) / def_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # 简单平均池化
            pooled = attended.mean(dim=1)
        
        # 5. 特征处理
        output = self.feature_processor(pooled)
        
        return output


class DefDataProcessor:
    """
    DEF数据预处理器，负责将原始def数据转换为模型可用的格式
    
    作者: Dionysus
    """
    
    def __init__(self, max_def_length: int = 30):
        """
        初始化DEF数据处理器
        
        参数:
            max_def_length: 最大def序列长度
        """
        self.max_def_length = max_def_length
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.def_vocab = {}
        self.vocab_size = 0
        self.is_fitted = False
    
    def _parse_def_value(self, def_value: Union[str, float, int]) -> List[str]:
        """
        解析def值，支持多种格式，对文本进行智能分词处理
        使用jieba进行中文分词，正则表达式进行英文分词
        
        参数:
            def_value: def值
            
        返回:
            解析后的词汇列表
        """
        if pd.isna(def_value):
            return ['<UNK>']
        
        if isinstance(def_value, (int, float)):
            return [str(def_value)]
        
        if isinstance(def_value, str):
            import re
            import jieba
            
            # 清理文本
            text = def_value.strip()
            if not text:
                return ['<UNK>']
            
            # 如果是短文本（可能是简单标签），直接返回
            if len(text) <= 10 and '，' not in text and '。' not in text and ',' not in text and '.' not in text:
                return [text]
            
            tokens = []
            
            # 检测文本中的中文比例
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
            chinese_ratio = len(chinese_chars) / len(text) if text else 0
            
            if chinese_ratio > 0.3:  # 如果中文字符占比超过30%，使用jieba分词
                # 使用jieba进行中文分词
                jieba_tokens = list(jieba.cut(text, cut_all=False))
                
                for token in jieba_tokens:
                    token = token.strip()
                    if not token:
                        continue
                    
                    # 过滤掉单字符和标点符号
                    if len(token) >= 2 and not re.match(r'^[，。、；：！？""''（）【】\s\.,;:!?()\[\]]+$', token):
                        tokens.append(token)
                    
                    # 如果jieba分出的词还包含英文，进一步处理
                    if re.search(r'[a-zA-Z]', token):
                        english_words = re.findall(r'[a-zA-Z]{2,}', token)
                        tokens.extend([word.lower() for word in english_words])
            
            else:  # 英文为主的文本，使用原有的正则表达式方法
                # 1. 按标点符号分割
                sentences = re.split(r'[，。,.\s]+', text)
                
                # 2. 提取关键词
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    # 提取中文词汇（2-4个字符）
                    chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', sentence)
                    tokens.extend(chinese_words)
                    
                    # 提取英文词汇
                    english_words = re.findall(r'[a-zA-Z]{2,}', sentence)
                    tokens.extend([word.lower() for word in english_words])
            
            # 去重并过滤
            unique_tokens = []
            seen = set()
            for token in tokens:
                if token not in seen and len(token) >= 2:
                    unique_tokens.append(token)
                    seen.add(token)
            
            # 限制词汇数量，保留前20个最重要的词
            if len(unique_tokens) > 20:
                unique_tokens = unique_tokens[:20]
            
            return unique_tokens if unique_tokens else ['<UNK>']
        
        return ['<UNK>']
    
    def fit(self, def_data: pd.Series) -> 'DefDataProcessor':
        """
        拟合def数据，构建词汇表
        
        参数:
            def_data: def数据序列
            
        返回:
            自身实例
        """
        logger.info("开始构建DEF词汇表...")
        
        # 收集所有def值
        all_def_values = []
        for def_value in def_data:
            parsed_values = self._parse_def_value(def_value)
            all_def_values.extend(parsed_values)
        
        # 构建词汇表
        unique_values = list(set(all_def_values))
        self.def_vocab = {value: idx for idx, value in enumerate(unique_values)}
        self.vocab_size = len(self.def_vocab)
        
        # 添加特殊标记
        if '<PAD>' not in self.def_vocab:
            self.def_vocab['<PAD>'] = self.vocab_size
            self.vocab_size += 1
        
        if '<UNK>' not in self.def_vocab:
            self.def_vocab['<UNK>'] = self.vocab_size
            self.vocab_size += 1
        
        self.is_fitted = True
        logger.info(f"DEF词汇表构建完成，共 {self.vocab_size} 个唯一值")
        
        return self
    
    def transform(self, def_data: pd.Series) -> Dict[str, np.ndarray]:
        """
        转换def数据为模型输入格式
        
        参数:
            def_data: def数据序列
            
        返回:
            包含indices和mask的字典
        """
        if not self.is_fitted:
            raise ValueError("DefDataProcessor必须先调用fit方法")
        
        batch_size = len(def_data)
        indices = np.full((batch_size, self.max_def_length), self.def_vocab['<PAD>'], dtype=np.int64)
        masks = np.zeros((batch_size, self.max_def_length), dtype=np.float32)
        
        for i, def_value in enumerate(def_data):
            parsed_values = self._parse_def_value(def_value)
            
            # 转换为索引
            for j, value in enumerate(parsed_values[:self.max_def_length]):
                if value in self.def_vocab:
                    indices[i, j] = self.def_vocab[value]
                else:
                    indices[i, j] = self.def_vocab['<UNK>']
                masks[i, j] = 1.0
        
        return {
            'indices': indices,
            'masks': masks
        }
    
    def fit_transform(self, def_data: pd.Series) -> Dict[str, np.ndarray]:
        """
        拟合并转换def数据
        
        参数:
            def_data: def数据序列
            
        返回:
            包含indices和mask的字典
        """
        return self.fit(def_data).transform(def_data)


class DefFeatureExtractor:
    """
    DEF特征提取器，整合数据处理和编码功能
    
    作者: Dionysus
    """
    
    def __init__(self, 
                 max_def_length: int = 30,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 device: str = 'cuda'):
        """
        初始化DEF特征提取器
        
        参数:
            max_def_length: 最大def序列长度
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            device: 计算设备
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.processor = DefDataProcessor(max_def_length)
        self.encoder = None
        self.is_fitted = False
    
    def fit(self, def_data: pd.Series) -> 'DefFeatureExtractor':
        """
        拟合def数据
        
        参数:
            def_data: def数据序列
            
        返回:
            自身实例
        """
        # 拟合数据处理器
        self.processor.fit(def_data)
        
        # 初始化编码器
        self.encoder = DefEncoder(
            def_vocab_size=self.processor.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            device=self.device
        )
        
        self.is_fitted = True
        return self
    
    def extract_features(self, def_data: pd.Series, batch_size: int = 256) -> np.ndarray:
        """
        提取def特征
        
        参数:
            def_data: def数据序列
            batch_size: 批处理大小
            
        返回:
            特征数组 [num_samples, output_dim]
        """
        if not self.is_fitted:
            raise ValueError("DefFeatureExtractor必须先调用fit方法")
        
        # 转换数据
        processed_data = self.processor.transform(def_data)
        indices = processed_data['indices']
        masks = processed_data['masks']
        
        # 批量提取特征
        all_features = []
        self.encoder.eval()
        
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_indices = torch.tensor(
                    indices[i:i+batch_size], 
                    dtype=torch.long, 
                    device=self.device
                )
                batch_masks = torch.tensor(
                    masks[i:i+batch_size], 
                    dtype=torch.float32, 
                    device=self.device
                )
                
                batch_features = self.encoder(batch_indices, batch_masks)
                all_features.append(batch_features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def fit_extract(self, def_data: pd.Series, batch_size: int = 256) -> np.ndarray:
        """
        拟合并提取def特征
        
        参数:
            def_data: def数据序列
            batch_size: 批处理大小
            
        返回:
            特征数组 [num_samples, output_dim]
        """
        return self.fit(def_data).extract_features(def_data, batch_size)


def create_def_encoder(device: str = 'cuda', 
                      output_dim: int = 128) -> DefFeatureExtractor:
    """
    创建DEF编码器实例
    
    参数:
        device: 计算设备
        output_dim: 输出维度
        
    返回:
        DEF特征提取器实例
    """
    return DefFeatureExtractor(
        max_def_length=30,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=output_dim,
        device=device
    )