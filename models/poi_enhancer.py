"""
POI-Enhancer: 基于大语言模型的兴趣点表征学习语义增强框架

作者: Dionysus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class POIEnhancer(nn.Module):
    """
    POI语义增强器，使用预训练语言模型对POI文本进行语义增强
    
    功能:
        1. 对POI文本进行上下文理解和语义扩展
        2. 生成更丰富的语义表示
        3. 提供多种增强策略
    """
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                 enhancement_dim: int = 384,
                 device: str = 'cuda'):
        """
        初始化POI增强器
        
        参数:
            model_name: 预训练模型名称
            enhancement_dim: 增强后的特征维度
            device: 计算设备
        """
        super(POIEnhancer, self).__init__()
        self.device = device
        self.enhancement_dim = enhancement_dim
        
        # 确定模型路径 - 优先使用本地模型
        local_model_path = os.path.join(os.path.dirname(__file__), 'paraphrase-multilingual-MiniLM-L12-v2')
        if os.path.exists(local_model_path):
            model_path = local_model_path
            logger.info(f"使用本地模型: {model_path}")
        else:
            model_path = model_name
            logger.info(f"使用在线模型: {model_path}")
        
        # 加载预训练模型和分词器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=os.path.exists(local_model_path)
            )
            self.model = AutoModel.from_pretrained(
                model_path, 
                local_files_only=os.path.exists(local_model_path)
            ).to(device)
            logger.info(f"成功加载预训练模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
        
        # 语义增强层
        self.semantic_enhancer = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, enhancement_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(enhancement_dim * 2, enhancement_dim),
            nn.LayerNorm(enhancement_dim)
        ).to(device)
        
        # 上下文感知层
        self.context_attention = nn.MultiheadAttention(
            embed_dim=enhancement_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(enhancement_dim * 2, enhancement_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(enhancement_dim, enhancement_dim)
        ).to(device)
    
    def _create_enhanced_prompts(self, poi_texts: List[str]) -> List[str]:
        """
        为POI文本创建增强提示，提供更丰富的上下文信息
        
        参数:
            poi_texts: POI文本列表
            
        返回:
            增强后的提示文本列表
        """
        enhanced_prompts = []
        
        for text in poi_texts:
            # 解析POI文本 (格式: name | category | address)
            parts = text.split(' | ')
            if len(parts) >= 3:
                name, category, address = parts[0], parts[1], parts[2]
                
                # 创建语义增强提示
                enhanced_prompt = f"""
                这是一个兴趣点(POI)的详细描述：
                名称：{name}
                类别：{category}
                地址：{address}
                
                请理解这个地点的特征、功能和语义含义。
                """.strip()
            else:
                # 如果格式不标准，使用原始文本
                enhanced_prompt = f"兴趣点描述：{text}"
            
            enhanced_prompts.append(enhanced_prompt)
        
        return enhanced_prompts
    
    def _extract_features(self, texts: List[str]) -> torch.Tensor:
        """
        从文本中提取深层语义特征
        
        参数:
            texts: 文本列表
            
        返回:
            特征张量 [batch_size, hidden_size]
        """
        # 分词和编码
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # 通过预训练模型提取特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]标记的表示作为句子级特征
            features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return features
    
    def _apply_context_attention(self, features: torch.Tensor) -> torch.Tensor:
        """
        应用上下文注意力机制，增强特征表示
        
        参数:
            features: 输入特征 [batch_size, feature_dim]
            
        返回:
            增强后的特征 [batch_size, feature_dim]
        """
        # 为注意力机制添加序列维度
        features_seq = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 应用自注意力
        attended_features, _ = self.context_attention(
            features_seq, features_seq, features_seq
        )
        
        # 移除序列维度
        attended_features = attended_features.squeeze(1)  # [batch_size, feature_dim]
        
        return attended_features
    
    def forward(self, poi_texts: List[str]) -> torch.Tensor:
        """
        对POI文本进行语义增强
        
        参数:
            poi_texts: POI文本列表
            
        返回:
            增强后的语义特征 [batch_size, enhancement_dim]
        """
        # 1. 创建增强提示
        enhanced_prompts = self._create_enhanced_prompts(poi_texts)
        
        # 2. 提取原始特征
        original_features = self._extract_features(poi_texts)
        enhanced_features = self._extract_features(enhanced_prompts)
        
        # 3. 通过语义增强层处理
        original_enhanced = self.semantic_enhancer(original_features)
        prompt_enhanced = self.semantic_enhancer(enhanced_features)
        
        # 4. 应用上下文注意力
        original_attended = self._apply_context_attention(original_enhanced)
        prompt_attended = self._apply_context_attention(prompt_enhanced)
        
        # 5. 特征融合
        combined_features = torch.cat([original_attended, prompt_attended], dim=1)
        final_features = self.feature_fusion(combined_features)
        
        # 6. 残差连接
        output = final_features + original_enhanced
        
        return output
    
    def enhance_batch(self, poi_texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量处理POI文本增强
        
        参数:
            poi_texts: POI文本列表
            batch_size: 批处理大小
            
        返回:
            增强后的特征数组
        """
        self.eval()
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(poi_texts), batch_size):
                batch_texts = poi_texts[i:i + batch_size]
                batch_features = self.forward(batch_texts)
                all_features.append(batch_features.cpu().numpy())
        
        return np.vstack(all_features)


class POISemanticAugmenter:
    """
    POI语义数据增强器，提供多种数据增强策略
    
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        初始化语义增强器
        
        参数:
            device: 计算设备
        """
        self.device = device
        self.enhancer = POIEnhancer(device=device)
    
    def augment_with_synonyms(self, poi_texts: List[str]) -> List[str]:
        """
        使用同义词进行数据增强
        
        参数:
            poi_texts: 原始POI文本列表
            
        返回:
            增强后的文本列表
        """
        # 简单的同义词替换策略
        synonym_map = {
            '餐厅': ['饭店', '食堂', '酒楼', '餐馆'],
            '酒店': ['宾馆', '旅馆', '客栈', '招待所'],
            '商场': ['购物中心', '百货商店', '商业中心', '超市'],
            '医院': ['诊所', '卫生院', '医疗中心', '保健院'],
            '学校': ['教育机构', '学院', '培训中心', '教学点'],
            '银行': ['金融机构', '信用社', '储蓄所', '营业厅']
        }
        
        augmented_texts = []
        for text in poi_texts:
            augmented_text = text
            for original, synonyms in synonym_map.items():
                if original in text:
                    # 随机选择一个同义词替换
                    import random
                    synonym = random.choice(synonyms)
                    augmented_text = augmented_text.replace(original, synonym, 1)
            augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def augment_with_context(self, poi_texts: List[str]) -> List[str]:
        """
        使用上下文信息进行数据增强
        
        参数:
            poi_texts: 原始POI文本列表
            
        返回:
            增强后的文本列表
        """
        augmented_texts = []
        
        for text in poi_texts:
            parts = text.split(' | ')
            if len(parts) >= 3:
                name, category, address = parts[0], parts[1], parts[2]
                
                # 添加上下文信息
                context_info = f"位于{address}的{category}类型场所{name}"
                augmented_texts.append(context_info)
            else:
                augmented_texts.append(text)
        
        return augmented_texts
    
    def enhance_poi_representations(self, 
                                  poi_texts: List[str], 
                                  use_augmentation: bool = True,
                                  batch_size: int = 32) -> np.ndarray:
        """
        增强POI表示，结合语义增强和数据增强
        
        参数:
            poi_texts: POI文本列表
            use_augmentation: 是否使用数据增强
            batch_size: 批处理大小
            
        返回:
            增强后的特征表示
        """
        logger.info(f"开始增强 {len(poi_texts)} 个POI的语义表示")
        
        # 1. 原始文本增强
        original_features = self.enhancer.enhance_batch(poi_texts, batch_size)
        
        if not use_augmentation:
            return original_features
        
        # 2. 数据增强
        augmented_texts_1 = self.augment_with_synonyms(poi_texts)
        augmented_texts_2 = self.augment_with_context(poi_texts)
        
        # 3. 增强文本的特征提取
        aug_features_1 = self.enhancer.enhance_batch(augmented_texts_1, batch_size)
        aug_features_2 = self.enhancer.enhance_batch(augmented_texts_2, batch_size)
        
        # 4. 特征融合 - 使用加权平均
        weights = [0.5, 0.25, 0.25]  # 原始文本权重更高
        final_features = (weights[0] * original_features + 
                         weights[1] * aug_features_1 + 
                         weights[2] * aug_features_2)
        
        logger.info("POI语义增强完成")
        return final_features


def create_poi_enhancer(device: str = 'cuda') -> POISemanticAugmenter:
    """
    创建POI语义增强器实例
    
    参数:
        device: 计算设备
        
    返回:
        POI语义增强器实例
    """
    return POISemanticAugmenter(device=device)