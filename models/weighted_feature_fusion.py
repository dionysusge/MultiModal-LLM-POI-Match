#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加权特征融合模块
Author: Dionysus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedFeatureFusion(nn.Module):
    """
    加权特征融合模块
    
    实现多种特征的加权融合，支持：
    - Geohash特征（经纬度编码，权重最高）
    - LLM特征（语义增强，权重次之）
    - 文本特征（基础语义，权重最低）
    - GAT特征（图结构信息，权重中等）
    """
    
    def __init__(self, 
                 text_dim=384, 
                 geohash_dim=128, 
                 llm_dim=384, 
                 gat_dim=128,
                 output_dim=512,
                 fusion_method='weighted_attention'):
        """
        初始化加权特征融合模块
        
        Args:
            text_dim: 文本特征维度
            geohash_dim: Geohash特征维度
            llm_dim: LLM特征维度
            gat_dim: GAT特征维度
            output_dim: 输出特征维度
            fusion_method: 融合方法 ('weighted_attention', 'weighted_concat', 'hierarchical')
        """
        super(WeightedFeatureFusion, self).__init__()
        
        self.text_dim = text_dim
        self.geohash_dim = geohash_dim
        self.llm_dim = llm_dim
        self.gat_dim = gat_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        # 特征投影层，将不同维度的特征投影到统一维度
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.geohash_proj = nn.Linear(geohash_dim, output_dim)
        self.llm_proj = nn.Linear(llm_dim, output_dim)
        self.gat_proj = nn.Linear(gat_dim, output_dim)
        
        # 预定义权重（可学习）
        # Geohash > LLM > GAT > Text
        self.feature_weights = nn.Parameter(torch.tensor([
            0.15,  # text_weight (最低)
            0.40,  # geohash_weight (最高)
            0.30,  # llm_weight (次之)
            0.15   # gat_weight (中等)
        ]))
        
        if fusion_method == 'weighted_attention':
            # 注意力机制
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            )
            self.norm = nn.LayerNorm(output_dim)
            
        elif fusion_method == 'hierarchical':
            # 分层融合
            self.geo_llm_fusion = nn.Linear(output_dim * 2, output_dim)
            self.all_fusion = nn.Linear(output_dim * 2, output_dim)
            
        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, text_feat, geohash_feat, llm_feat=None, gat_feat=None):
        """
        前向传播
        
        Args:
            text_feat: 文本特征 [batch_size, text_dim]
            geohash_feat: Geohash特征 [batch_size, geohash_dim]
            llm_feat: LLM特征 [batch_size, llm_dim] (可选)
            gat_feat: GAT特征 [batch_size, gat_dim] (可选)
            
        Returns:
            融合后的特征 [batch_size, output_dim]
        """
        batch_size = text_feat.size(0)
        
        # 特征投影到统一维度
        text_proj = self.text_proj(text_feat)
        geohash_proj = self.geohash_proj(geohash_feat)
        
        # 收集所有可用特征
        features = [text_proj, geohash_proj]
        weights = [self.feature_weights[0], self.feature_weights[1]]  # text, geohash
        
        if llm_feat is not None:
            llm_proj = self.llm_proj(llm_feat)
            features.append(llm_proj)
            weights.append(self.feature_weights[2])  # llm
            
        if gat_feat is not None:
            gat_proj = self.gat_proj(gat_feat)
            features.append(gat_proj)
            weights.append(self.feature_weights[3])  # gat
        
        # 权重归一化
        weights = torch.stack(weights)
        weights = F.softmax(weights, dim=0)
        
        if self.fusion_method == 'weighted_attention':
            return self._weighted_attention_fusion(features, weights)
        elif self.fusion_method == 'weighted_concat':
            return self._weighted_concat_fusion(features, weights)
        elif self.fusion_method == 'hierarchical':
            return self._hierarchical_fusion(text_proj, geohash_proj, llm_proj if llm_feat is not None else None, gat_proj if gat_feat is not None else None)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _weighted_attention_fusion(self, features, weights):
        """
        基于注意力机制的加权融合
        """
        # 将特征堆叠为序列 [batch_size, num_features, output_dim]
        feature_stack = torch.stack(features, dim=1)
        
        # 应用权重
        weighted_features = feature_stack * weights.view(1, -1, 1)
        
        # 自注意力
        attended_features, _ = self.attention(
            weighted_features, weighted_features, weighted_features
        )
        
        # 加权平均
        fused_feature = torch.sum(attended_features * weights.view(1, -1, 1), dim=1)
        fused_feature = self.norm(fused_feature)
        
        return self.output_layer(fused_feature)
    
    def _weighted_concat_fusion(self, features, weights):
        """
        加权拼接融合
        """
        # 应用权重并求和
        weighted_sum = sum(feat * weight for feat, weight in zip(features, weights))
        return self.output_layer(weighted_sum)
    
    def _hierarchical_fusion(self, text_feat, geohash_feat, llm_feat=None, gat_feat=None):
        """
        分层融合：先融合高权重特征，再逐步加入低权重特征
        """
        # 第一层：融合Geohash和LLM（高权重特征）
        if llm_feat is not None:
            high_priority = torch.cat([geohash_feat, llm_feat], dim=-1)
            high_priority = self.geo_llm_fusion(high_priority)
        else:
            high_priority = geohash_feat
        
        # 第二层：加入GAT和文本特征
        if gat_feat is not None:
            # 简单加权平均GAT和文本
            low_priority = 0.5 * gat_feat + 0.5 * text_feat
        else:
            low_priority = text_feat
        
        # 最终融合
        final_features = torch.cat([high_priority, low_priority], dim=-1)
        final_output = self.all_fusion(final_features)
        
        return self.output_layer(final_output)


class FeatureFusionManager:
    """
    特征融合管理器
    
    负责管理不同特征的加载、预处理和融合
    """
    
    def __init__(self, fusion_model, device='cpu'):
        """
        初始化特征融合管理器
        
        Args:
            fusion_model: 融合模型实例
            device: 计算设备
        """
        self.fusion_model = fusion_model.to(device)
        self.device = device
        
    def fuse_features(self, text_features, geohash_features, llm_features=None, gat_features=None):
        """
        融合多种特征
        
        Args:
            text_features: 文本特征字典 {poi_id: feature_vector}
            geohash_features: Geohash特征字典 {poi_id: feature_vector}
            llm_features: LLM特征字典 {poi_id: feature_vector} (可选)
            gat_features: GAT特征字典 {poi_id: feature_vector} (可选)
            
        Returns:
            融合后的特征字典 {poi_id: fused_feature_vector}
        """
        fused_features = {}
        
        # 获取所有POI ID
        all_poi_ids = set(text_features.keys()) & set(geohash_features.keys())
        
        if llm_features:
            all_poi_ids &= set(llm_features.keys())
        if gat_features:
            all_poi_ids &= set(gat_features.keys())
        
        self.fusion_model.eval()
        with torch.no_grad():
            for poi_id in all_poi_ids:
                # 准备特征
                text_feat = torch.tensor(text_features[poi_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                geohash_feat = torch.tensor(geohash_features[poi_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                llm_feat = None
                if llm_features and poi_id in llm_features:
                    llm_feat = torch.tensor(llm_features[poi_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                gat_feat = None
                if gat_features and poi_id in gat_features:
                    gat_feat = torch.tensor(gat_features[poi_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 融合特征
                fused_feat = self.fusion_model(text_feat, geohash_feat, llm_feat, gat_feat)
                fused_features[poi_id] = fused_feat.squeeze().cpu().numpy()
        
        return fused_features
    
    def save_model(self, save_path):
        """保存融合模型"""
        torch.save(self.fusion_model.state_dict(), save_path)
        print(f"融合模型已保存到: {save_path}")
    
    def load_model(self, load_path):
        """加载融合模型"""
        self.fusion_model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"融合模型已从 {load_path} 加载")


def create_fusion_model(text_dim=384, geohash_dim=128, llm_dim=384, gat_dim=128, 
                       output_dim=512, fusion_method='weighted_attention'):
    """
    创建特征融合模型的工厂函数
    
    Args:
        text_dim: 文本特征维度
        geohash_dim: Geohash特征维度  
        llm_dim: LLM特征维度
        gat_dim: GAT特征维度
        output_dim: 输出特征维度
        fusion_method: 融合方法
        
    Returns:
        融合模型实例
    """
    return WeightedFeatureFusion(
        text_dim=text_dim,
        geohash_dim=geohash_dim,
        llm_dim=llm_dim,
        gat_dim=gat_dim,
        output_dim=output_dim,
        fusion_method=fusion_method
    )


if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建融合模型
    fusion_model = create_fusion_model(
        text_dim=384,
        geohash_dim=128,
        llm_dim=384,
        gat_dim=128,
        output_dim=512,
        fusion_method='weighted_attention'
    )
    
    # 创建管理器
    manager = FeatureFusionManager(fusion_model, device)
    
    # 模拟特征数据
    batch_size = 10
    text_feat = torch.randn(batch_size, 384)
    geohash_feat = torch.randn(batch_size, 128)
    llm_feat = torch.randn(batch_size, 384)
    gat_feat = torch.randn(batch_size, 128)
    
    # 测试融合
    fused_feat = fusion_model(text_feat, geohash_feat, llm_feat, gat_feat)
    print(f"融合后特征形状: {fused_feat.shape}")
    print(f"特征权重: {F.softmax(fusion_model.feature_weights, dim=0)}")