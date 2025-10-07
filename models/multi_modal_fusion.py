"""
多模态特征融合模块: 使用多头自注意力机制融合文本、图、地理、def四种特征向量

作者: Dionysus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureProjector(nn.Module):
    """
    特征投影器，将不同维度的特征向量投影到统一的维度空间
    
    功能:
        1. 将四种特征向量投影到相同维度
        2. 添加模态特定的位置编码
        3. 支持残差连接和层归一化
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int],
                 unified_dim: int = 256,
                 dropout: float = 0.1,
                 device: str = 'cuda'):
        """
        初始化特征投影器
        
        参数:
            input_dims: 各模态的输入维度字典 {'text': dim1, 'graph': dim2, 'geo': dim3, 'def': dim4}
            unified_dim: 统一的输出维度
            dropout: dropout比率
            device: 计算设备
        """
        super(FeatureProjector, self).__init__()
        self.device = device
        self.unified_dim = unified_dim
        self.modalities = list(input_dims.keys())
        
        # 为每种模态创建投影层
        self.projectors = nn.ModuleDict()
        for modality, input_dim in input_dims.items():
            self.projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, unified_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(unified_dim * 2, unified_dim),
                nn.LayerNorm(unified_dim)
            ).to(device)
        
        # 模态特定的位置编码
        self.modality_embeddings = nn.ParameterDict()
        for i, modality in enumerate(self.modalities):
            self.modality_embeddings[modality] = nn.Parameter(
                torch.randn(1, unified_dim).to(device)
            )
        
        # 门控机制 - 控制每种模态的重要性
        self.gate_network = nn.Sequential(
            nn.Linear(unified_dim * len(self.modalities), unified_dim),
            nn.ReLU(),
            nn.Linear(unified_dim, len(self.modalities)),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            features: 各模态特征字典
            
        返回:
            projected_features: 投影后的特征 [batch_size, num_modalities, unified_dim]
            gate_weights: 门控权重 [batch_size, num_modalities]
        """
        batch_size = list(features.values())[0].size(0)
        projected_list = []
        
        # 投影每种模态的特征
        for modality in self.modalities:
            if modality in features:
                # 投影到统一维度
                projected = self.projectors[modality](features[modality])
                # 添加模态特定编码
                projected = projected + self.modality_embeddings[modality]
                projected_list.append(projected)
            else:
                # 如果某个模态缺失，使用零向量
                zero_feature = torch.zeros(batch_size, self.unified_dim, device=self.device)
                projected_list.append(zero_feature)
        
        # 堆叠所有模态特征
        projected_features = torch.stack(projected_list, dim=1)  # [batch_size, num_modalities, unified_dim]
        
        # 计算门控权重
        concatenated = torch.cat(projected_list, dim=1)  # [batch_size, num_modalities * unified_dim]
        gate_weights = self.gate_network(concatenated)  # [batch_size, num_modalities]
        
        return projected_features, gate_weights


class MultiHeadCrossModalAttention(nn.Module):
    """
    多头跨模态注意力机制
    
    功能:
        1. 计算不同模态之间的注意力
        2. 支持多头注意力机制
        3. 融合跨模态信息
    """
    
    def __init__(self, 
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 device: str = 'cuda'):
        """
        初始化多头跨模态注意力
        
        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: dropout比率
            device: 计算设备
        """
        super(MultiHeadCrossModalAttention, self).__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 查询、键、值投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim).to(device)
        self.k_proj = nn.Linear(embed_dim, embed_dim).to(device)
        self.v_proj = nn.Linear(embed_dim, embed_dim).to(device)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim).to(device)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(embed_dim).to(device)
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query: 查询张量 [batch_size, seq_len_q, embed_dim]
            key: 键张量 [batch_size, seq_len_k, embed_dim]
            value: 值张量 [batch_size, seq_len_v, embed_dim]
            mask: 注意力掩码
            
        返回:
            输出张量 [batch_size, seq_len_q, embed_dim]
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # 保存残差连接的输入
        residual = query
        
        # 计算Q, K, V
        Q = self.q_proj(query)  # [batch_size, seq_len_q, embed_dim]
        K = self.k_proj(key)    # [batch_size, seq_len_k, embed_dim]
        V = self.v_proj(value)  # [batch_size, seq_len_v, embed_dim]
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output


class AdaptiveAttentionWeighting(nn.Module):
    """
    自适应注意力权重分配模块
    
    功能:
        1. 动态评估各模态特征的重要性
        2. 基于特征质量和任务相关性分配权重
        3. 支持特征间的相互作用建模
    """
    
    def __init__(self,
                 unified_dim: int = 256,
                 num_modalities: int = 4,
                 geo_weight_factor: float = 2.0,
                 device: str = 'cuda'):
        """
        初始化自适应注意力权重模块
        
        参数:
            unified_dim: 统一特征维度
            num_modalities: 模态数量
            geo_weight_factor: 地理特征权重增强因子
            device: 计算设备
        """
        super(AdaptiveAttentionWeighting, self).__init__()
        self.device = device
        self.unified_dim = unified_dim
        self.num_modalities = num_modalities
        self.geo_weight_factor = geo_weight_factor
        
        # 定义模态顺序，确保地理特征在固定位置
        self.modality_order = ['text', 'graph', 'geo', 'def']
        self.geo_index = self.modality_order.index('geo') if 'geo' in self.modality_order else 2
        
        # 特征质量评估网络
        self.quality_assessor = nn.Sequential(
            nn.Linear(unified_dim, unified_dim // 2).to(device),
            nn.ReLU(),
            nn.Linear(unified_dim // 2, 1).to(device),
            nn.Sigmoid()
        )
        
        # 模态间相互作用建模
        self.interaction_matrix = nn.Parameter(
            torch.randn(num_modalities, num_modalities).to(device)
        )
        
        # 任务相关性评估
        self.task_relevance = nn.Sequential(
            nn.Linear(unified_dim * num_modalities, unified_dim).to(device),
            nn.ReLU(),
            nn.Linear(unified_dim, num_modalities).to(device),
            nn.Softmax(dim=-1)
        )
        
        # 最终权重融合
        self.weight_fusion = nn.Sequential(
            nn.Linear(num_modalities * 3, num_modalities * 2).to(device),
            nn.ReLU(),
            nn.Linear(num_modalities * 2, num_modalities).to(device),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算自适应注意力权重
        
        参数:
            features: [batch_size, num_modalities, unified_dim]
            
        返回:
            attention_weights: [batch_size, num_modalities]
        """
        batch_size = features.size(0)
        
        # 1. 特征质量评估
        quality_scores = []
        for i in range(self.num_modalities):
            quality = self.quality_assessor(features[:, i, :])  # [batch_size, 1]
            quality_scores.append(quality.squeeze(-1))
        quality_weights = torch.stack(quality_scores, dim=1)  # [batch_size, num_modalities]
        
        # 2. 模态间相互作用
        # 计算特征间的相似性
        interaction_scores = torch.zeros(batch_size, self.num_modalities).to(self.device)
        for i in range(self.num_modalities):
            for j in range(self.num_modalities):
                if i != j:
                    # 计算模态i和模态j的相似性
                    similarity = F.cosine_similarity(
                        features[:, i, :], features[:, j, :], dim=-1
                    )
                    interaction_scores[:, i] += self.interaction_matrix[i, j] * similarity
        
        # 归一化相互作用分数
        interaction_weights = F.softmax(interaction_scores, dim=-1)
        
        # 3. 任务相关性评估
        concatenated_features = features.view(batch_size, -1)  # [batch_size, num_modalities * unified_dim]
        task_weights = self.task_relevance(concatenated_features)  # [batch_size, num_modalities]
        
        # 4. 融合所有权重信息
        all_weights = torch.cat([
            quality_weights,
            interaction_weights,
            task_weights
        ], dim=-1)  # [batch_size, num_modalities * 3]
        
        # 5. 最终权重计算
        final_weights = self.weight_fusion(all_weights)  # [batch_size, num_modalities]
        
        # 6. 为地理特征增加权重增强
        geo_enhanced_final_weights = final_weights.clone()
        if self.geo_index < geo_enhanced_final_weights.size(1):
            # 对地理特征权重进行增强
            geo_enhanced_final_weights[:, self.geo_index] *= self.geo_weight_factor
            
            # 重新归一化，确保权重和为1
            geo_enhanced_final_weights = F.softmax(geo_enhanced_final_weights, dim=1)
        
        return geo_enhanced_final_weights


class MultiModalFusionNetwork(nn.Module):
    """
    多模态融合网络
    
    功能:
        1. 将不同模态的特征投影到统一空间
        2. 使用增强的注意力机制进行特征融合
        3. 输出融合后的特征表示
    """
    
    def __init__(self,
                 input_dims: Dict[str, int],
                 unified_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 output_dim: int = 512,
                 dropout: float = 0.1,
                 geo_weight_factor: float = 2.0,
                 device: str = 'cuda'):
        """
        初始化多模态融合网络
        
        参数:
            input_dims: 各模态输入维度字典
            unified_dim: 统一特征维度
            num_heads: 注意力头数
            num_layers: 注意力层数
            output_dim: 输出维度
            dropout: dropout比率
            geo_weight_factor: 地理特征权重增强因子，默认2.0表示地理特征权重是其他特征的2倍
            device: 计算设备
        """
        super(MultiModalFusionNetwork, self).__init__()
        self.device = device
        self.unified_dim = unified_dim
        self.num_modalities = len(input_dims)
        self.output_dim = output_dim
        self.geo_weight_factor = geo_weight_factor
        
        # 定义模态顺序，确保地理特征在固定位置
        self.modality_order = ['text', 'graph', 'geo', 'def']
        self.geo_index = self.modality_order.index('geo') if 'geo' in self.modality_order else 2
        
        # 特征投影器
        self.projector = FeatureProjector(
            input_dims=input_dims,
            unified_dim=unified_dim,
            dropout=dropout,
            device=device
        )
        
        # 自适应注意力权重分配
        self.adaptive_attention = AdaptiveAttentionWeighting(
            unified_dim=unified_dim,
            num_modalities=self.num_modalities,
            geo_weight_factor=geo_weight_factor,
            device=device
        )
        
        # 多层自注意力
        self.self_attention_layers = nn.ModuleList([
            MultiHeadCrossModalAttention(
                embed_dim=unified_dim,
                num_heads=num_heads,
                dropout=dropout,
                device=device
            ) for _ in range(num_layers)
        ])
        
        # 跨模态注意力
        self.cross_modal_attention = MultiHeadCrossModalAttention(
            embed_dim=unified_dim,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        
        # 特征聚合层
        self.aggregator = nn.Sequential(
            nn.Linear(self.num_modalities * unified_dim, unified_dim).to(device),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(unified_dim, unified_dim).to(device),
            nn.ReLU()
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(unified_dim, output_dim).to(device),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim).to(device)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            features: 各模态特征字典
            
        返回:
            融合结果字典，包含最终特征和注意力权重
        """
        # 1. 特征投影
        projected_features, gate_weights = self.projector(features)
        # projected_features: [batch_size, num_modalities, unified_dim]
        
        # 2. 计算自适应注意力权重
        adaptive_weights = self.adaptive_attention(projected_features)
        
        # 3. 为地理特征增加权重增强
        geo_enhanced_weights = adaptive_weights.clone()
        if self.geo_index < geo_enhanced_weights.size(1):
            geo_enhanced_weights[:, self.geo_index] *= self.geo_weight_factor
        
        # 重新归一化权重，确保权重和为1
        geo_enhanced_weights = F.softmax(geo_enhanced_weights, dim=1)
        
        # 4. 应用门控权重和增强后的自适应权重
        gate_weights_expanded = gate_weights.unsqueeze(-1)  # [batch_size, num_modalities, 1]
        geo_enhanced_weights_expanded = geo_enhanced_weights.unsqueeze(-1)  # [batch_size, num_modalities, 1]
        gated_features = projected_features * gate_weights_expanded * geo_enhanced_weights_expanded
        
        # 4. 多层自注意力
        attended_features = gated_features
        for attention_layer in self.self_attention_layers:
            attended_features = attention_layer(
                attended_features, attended_features, attended_features
            )
        
        # 5. 跨模态注意力 - 每个模态作为query，其他模态作为key和value
        cross_modal_outputs = []
        for i in range(self.num_modalities):
            query = attended_features[:, i:i+1, :]  # [batch_size, 1, unified_dim]
            # 其他模态作为key和value
            other_modalities = torch.cat([
                attended_features[:, :i, :],
                attended_features[:, i+1:, :]
            ], dim=1)  # [batch_size, num_modalities-1, unified_dim]
            
            if other_modalities.size(1) > 0:
                cross_output = self.cross_modal_attention(query, other_modalities, other_modalities)
                cross_modal_outputs.append(cross_output.squeeze(1))  # [batch_size, unified_dim]
            else:
                cross_modal_outputs.append(query.squeeze(1))
        
        # 6. 特征聚合
        # 结合原始特征和跨模态特征
        original_flat = attended_features.view(attended_features.size(0), -1)  # [batch_size, num_modalities * unified_dim]
        cross_modal_flat = torch.stack(cross_modal_outputs, dim=1).view(attended_features.size(0), -1)
        
        # 加权融合
        alpha = 0.7  # 原始特征权重
        beta = 0.3   # 跨模态特征权重
        combined_features = alpha * original_flat + beta * cross_modal_flat
        
        aggregated = self.aggregator(combined_features)  # [batch_size, unified_dim]
        
        # 7. 最终输出
        final_output = self.output_layer(aggregated)  # [batch_size, output_dim]
        
        return {
            'fused_features': final_output,
            'gate_weights': gate_weights,
            'adaptive_weights': adaptive_weights,
            'geo_enhanced_weights': geo_enhanced_weights,
            'projected_features': projected_features,
            'attended_features': attended_features
        }


class POIMultiModalProcessor:
    """
    POI多模态处理器，整合所有特征提取和融合功能
    
    作者: Dionysus
    """
    
    def __init__(self,
                 text_dim: int = 384,
                 graph_dim: int = 128,
                 geo_dim: int = 128,
                 def_dim: int = 128,
                 unified_dim: int = 256,
                 output_dim: int = 512,
                 geo_weight_factor: float = 2.0,
                 device: str = 'cuda'):
        """
        初始化多模态处理器
        
        参数:
            text_dim: 文本特征维度
            graph_dim: 图特征维度
            geo_dim: 地理特征维度
            def_dim: def特征维度
            unified_dim: 统一维度
            output_dim: 输出维度
            geo_weight_factor: 地理特征权重增强因子
            device: 计算设备
        """
        self.device = device
        
        # 定义输入维度
        self.input_dims = {
            'text': text_dim,
            'graph': graph_dim,
            'geo': geo_dim,
            'def': def_dim
        }
        
        # 创建融合网络
        self.fusion_network = MultiModalFusionNetwork(
            input_dims=self.input_dims,
            unified_dim=unified_dim,
            num_heads=8,
            num_layers=3,
            output_dim=output_dim,
            dropout=0.1,
            geo_weight_factor=geo_weight_factor,
            device=device
        )
        
        logger.info(f"多模态处理器初始化完成，输入维度: {self.input_dims}")
    
    def process_batch(self, 
                     text_features: np.ndarray,
                     graph_features: np.ndarray,
                     geo_features: np.ndarray,
                     def_features: np.ndarray,
                     batch_size: int = 256) -> Dict[str, np.ndarray]:
        """
        批量处理多模态特征
        
        参数:
            text_features: 文本特征数组
            graph_features: 图特征数组
            geo_features: 地理特征数组
            def_features: def特征数组
            batch_size: 批处理大小
            
        返回:
            处理结果字典
        """
        self.fusion_network.eval()
        
        num_samples = len(text_features)
        all_fused_features = []
        all_gate_weights = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                
                # 准备批次数据
                batch_features = {
                    'text': torch.tensor(text_features[i:end_idx], dtype=torch.float32, device=self.device),
                    'graph': torch.tensor(graph_features[i:end_idx], dtype=torch.float32, device=self.device),
                    'geo': torch.tensor(geo_features[i:end_idx], dtype=torch.float32, device=self.device),
                    'def': torch.tensor(def_features[i:end_idx], dtype=torch.float32, device=self.device)
                }
                
                # 前向传播
                results = self.fusion_network(batch_features)
                
                # 收集结果
                all_fused_features.append(results['fused_features'].cpu().numpy())
                all_gate_weights.append(results['gate_weights'].cpu().numpy())
        
        return {
            'fused_features': np.vstack(all_fused_features),
            'gate_weights': np.vstack(all_gate_weights)
        }
    
    def fuse_features(self, 
                     text_features: np.ndarray,
                     image_features: np.ndarray,  # 实际上是graph_features
                     geo_features: np.ndarray,
                     def_features: np.ndarray) -> np.ndarray:
        """
        融合单个样本的多模态特征
        
        参数:
            text_features: 文本特征向量
            image_features: 图特征向量（这里实际是GAT图嵌入）
            geo_features: 地理特征向量
            def_features: def特征向量
            
        返回:
            融合后的特征向量
        """
        # 将单个样本转换为批次格式
        text_batch = text_features.reshape(1, -1)
        graph_batch = image_features.reshape(1, -1)
        geo_batch = geo_features.reshape(1, -1)
        def_batch = def_features.reshape(1, -1)
        
        # 调用批处理方法
        results = self.process_batch(
            text_features=text_batch,
            graph_features=graph_batch,
            geo_features=geo_batch,
            def_features=def_batch,
            batch_size=1
        )
        
        # 返回单个样本的融合特征
        return results['fused_features'][0]


def create_multimodal_fusion_processor(text_dim: int = 384,
                                     graph_dim: int = 128,
                                     geo_dim: int = 128,
                                     def_dim: int = 128,
                                     output_dim: int = 512,
                                     geo_weight_factor: float = 2.0,
                                     device: str = 'cuda') -> POIMultiModalProcessor:
    """
    创建多模态融合处理器的工厂函数
    
    参数:
        text_dim: 文本特征维度
        graph_dim: 图特征维度
        geo_dim: 地理特征维度
        def_dim: def特征维度
        output_dim: 输出维度
        geo_weight_factor: 地理特征权重增强因子
        device: 计算设备
        
    返回:
        POIMultiModalProcessor: 多模态处理器实例
    """
    return POIMultiModalProcessor(
        text_dim=text_dim,
        graph_dim=graph_dim,
        geo_dim=geo_dim,
        def_dim=def_dim,
        output_dim=output_dim,
        geo_weight_factor=geo_weight_factor,
        device=device
    )