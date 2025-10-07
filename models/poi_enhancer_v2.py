"""
POI-Enhancer v2: 优化版本的兴趣点表征学习语义增强框架

主要优化:
1. 简化架构，减少重复计算
2. 直接利用SentenceTransformer的输出
3. 轻量级语义增强策略
4. 更高效的批处理机制

作者: Dionysus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

# 设置日志 - 简化输出
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class LightweightPOIEnhancer(nn.Module):
    """
    轻量级POI语义增强器，直接基于SentenceTransformer进行增强
    
    优化策略:
        1. 直接使用SentenceTransformer的输出，避免重复特征提取
        2. 简化增强网络架构
        3. 使用残差连接保持原始语义
        4. 支持批量处理提高效率
    """
    
    def __init__(self, 
                 input_dim: int = 384,  # SentenceTransformer输出维度
                 enhancement_dim: int = 384,
                 device: str = 'cuda'):
        """
        初始化轻量级POI增强器
        
        参数:
            input_dim: 输入特征维度（SentenceTransformer输出维度）
            enhancement_dim: 增强后的特征维度
            device: 计算设备
        """
        super(LightweightPOIEnhancer, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.enhancement_dim = enhancement_dim
        
        # 轻量级语义增强网络
        self.semantic_enhancer = nn.Sequential(
            nn.Linear(input_dim, enhancement_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(enhancement_dim, enhancement_dim),
            nn.LayerNorm(enhancement_dim)
        ).to(device)
        
        # 上下文注意力层（简化版）
        self.context_attention = nn.MultiheadAttention(
            embed_dim=enhancement_dim,
            num_heads=4,  # 减少注意力头数
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(enhancement_dim * 2, enhancement_dim),
            nn.GELU(),
            nn.Linear(enhancement_dim, enhancement_dim)
        ).to(device)
    
    def _create_context_prompts(self, poi_texts: List[str]) -> List[str]:
        """
        创建轻量级上下文提示，避免过度复杂化
        
        参数:
            poi_texts: POI文本列表
            
        返回:
            上下文增强的文本列表
        """
        enhanced_prompts = []
        
        for text in poi_texts:
            # 解析POI文本 (格式: name | category | address)
            parts = text.split(' | ')
            if len(parts) >= 3:
                name, category, address = parts[0], parts[1], parts[2]
                # 简化的上下文增强
                enhanced_prompt = f"{name} 是一个位于 {address} 的 {category}"
            else:
                enhanced_prompt = text
            
            enhanced_prompts.append(enhanced_prompt)
        
        return enhanced_prompts
    
    def forward(self, original_embeddings: torch.Tensor, poi_texts: List[str] = None) -> torch.Tensor:
        """
        对POI嵌入进行语义增强
        
        参数:
            original_embeddings: 原始SentenceTransformer嵌入 [batch_size, input_dim]
            poi_texts: POI文本列表（可选，用于上下文增强）
            
        返回:
            增强后的语义特征 [batch_size, enhancement_dim]
        """
        # 确保输入在正确的设备上
        original_embeddings = original_embeddings.to(self.device)
        
        # 1. 基础语义增强
        enhanced_features = self.semantic_enhancer(original_embeddings)
        
        # 2. 上下文注意力增强（如果提供了文本）
        if poi_texts is not None and len(poi_texts) == original_embeddings.size(0):
            # 为注意力机制添加序列维度
            features_seq = enhanced_features.unsqueeze(1)  # [batch_size, 1, enhancement_dim]
            
            # 应用自注意力
            attended_features, _ = self.context_attention(
                features_seq, features_seq, features_seq
            )
            attended_features = attended_features.squeeze(1)  # [batch_size, enhancement_dim]
            
            # 特征融合
            combined_features = torch.cat([enhanced_features, attended_features], dim=1)
            final_features = self.feature_fusion(combined_features)
        else:
            final_features = enhanced_features
        
        # 3. 残差连接（如果维度匹配）
        if original_embeddings.size(1) == final_features.size(1):
            output = final_features + original_embeddings
        else:
            output = final_features
        
        return output


class OptimizedPOISemanticAugmenter:
    """
    优化版POI语义数据增强器
    
    主要优化:
        1. 直接使用SentenceTransformer，避免重复模型加载
        2. 简化增强策略，提高处理效率
        3. 智能缓存机制
        4. 批量处理优化
    """
    
    def __init__(self, 
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 device: str = 'cuda'):
        """
        初始化优化版语义增强器
        
        参数:
            model_name: SentenceTransformer模型名称
            device: 计算设备
        """
        self.device = device
        self.model_name = model_name
        
        # 加载SentenceTransformer模型
        logger.info(f"加载SentenceTransformer模型: {model_name}")
        self.sentence_model = SentenceTransformer(model_name, device=device)
        
        # 初始化轻量级增强器
        self.enhancer = LightweightPOIEnhancer(
            input_dim=self.sentence_model.get_sentence_embedding_dimension(),
            device=device
        )
        
        logger.info("优化版POI语义增强器初始化完成")
    
    def _apply_lightweight_augmentation(self, embeddings: np.ndarray, 
                                      augmentation_factor: float = 0.05) -> np.ndarray:
        """
        应用轻量级数据增强（在特征空间进行）
        
        参数:
            embeddings: 原始嵌入
            augmentation_factor: 增强因子
            
        返回:
            增强后的嵌入
        """
        # 添加少量高斯噪声
        noise = np.random.normal(0, augmentation_factor, embeddings.shape)
        augmented_embeddings = embeddings + noise
        
        # 归一化处理
        norms = np.linalg.norm(augmented_embeddings, axis=1, keepdims=True)
        augmented_embeddings = augmented_embeddings / (norms + 1e-8)
        
        return augmented_embeddings
    
    def enhance_poi_representations(self, 
                                  poi_texts: List[str], 
                                  use_neural_enhancement: bool = True,
                                  use_augmentation: bool = False,
                                  batch_size: int = 64) -> np.ndarray:
        """
        增强POI表示，优化版本
        
        参数:
            poi_texts: POI文本列表
            use_neural_enhancement: 是否使用神经网络增强
            use_augmentation: 是否使用数据增强
            batch_size: 批处理大小
            
        返回:
            增强后的特征表示
        """
        # 1. 使用SentenceTransformer生成基础嵌入
        base_embeddings = self.sentence_model.encode(
            poi_texts,
            batch_size=batch_size,
            show_progress_bar=False,  # 关闭内部进度条
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化嵌入
        )
        
        # 2. 神经网络增强（可选）
        if use_neural_enhancement:
            self.enhancer.eval()
            enhanced_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(base_embeddings), batch_size):
                    end_idx = min(i + batch_size, len(base_embeddings))
                    batch_embeddings = torch.tensor(
                        base_embeddings[i:end_idx], 
                        dtype=torch.float32
                    ).to(self.device)
                    
                    batch_texts = poi_texts[i:end_idx]
                    
                    # 应用增强器
                    enhanced_batch = self.enhancer(batch_embeddings, batch_texts)
                    enhanced_embeddings.append(enhanced_batch.cpu().numpy())
            
            final_embeddings = np.vstack(enhanced_embeddings)
        else:
            final_embeddings = base_embeddings
        
        # 3. 轻量级数据增强（可选）
        if use_augmentation:
            augmented_embeddings = self._apply_lightweight_augmentation(final_embeddings)
            
            # 融合原始和增强特征
            final_embeddings = 0.7 * final_embeddings + 0.3 * augmented_embeddings
        return final_embeddings
    
    def encode_texts_efficiently(self, 
                                poi_texts: List[str], 
                                batch_size: int = 256) -> np.ndarray:
        """
        高效编码文本，直接返回SentenceTransformer嵌入
        
        参数:
            poi_texts: POI文本列表
            batch_size: 批处理大小
            
        返回:
            文本嵌入
        """
        logger.info(f"高效编码 {len(poi_texts)} 个POI文本")
        
        embeddings = self.sentence_model.encode(
            poi_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings


def create_optimized_poi_enhancer(model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                                 device: str = 'cuda') -> OptimizedPOISemanticAugmenter:
    """
    创建优化版POI语义增强器实例
    
    参数:
        model_name: SentenceTransformer模型名称
        device: 计算设备
        
    返回:
        优化版POI语义增强器实例
    """
    return OptimizedPOISemanticAugmenter(model_name=model_name, device=device)


# 兼容性函数，保持与原版本的接口一致
def create_poi_enhancer(device: str = 'cuda') -> OptimizedPOISemanticAugmenter:
    """
    创建POI语义增强器实例（兼容性函数）
    
    参数:
        device: 计算设备
        
    返回:
        优化版POI语义增强器实例
    """
    return create_optimized_poi_enhancer(device=device)