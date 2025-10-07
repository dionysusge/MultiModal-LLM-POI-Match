"""
交互特征融合MLP优化方案 add gat
Author: Dionysus

优化策略：
1. 降维处理：对交互特征进行降维，减少过拟合风险
2. 批归一化：添加BatchNorm稳定训练
3. 残差连接：增加残差连接提升梯度流
4. 学习率调度：使用余弦退火学习率
5. 特征选择：只保留最有效的交互特征
"""

import pandas as pd
import numpy as np
import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import argparse
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")


class OptimizedInteractionExtractor(nn.Module):
    """优化的交互特征提取器"""
    
    def __init__(self):
        """
        初始化优化的交互特征提取器
        """
        super(OptimizedInteractionExtractor, self).__init__()
        
    def forward(self, feat_a, feat_b):
        """
        计算两个特征向量的优化交互特征
        
        Args:
            feat_a: 特征向量A
            feat_b: 特征向量B
            
        Returns:
            交互特征张量
        """
        # 1. 元素级差值（绝对值）
        diff = torch.abs(feat_a - feat_b)
        
        # 2. 元素级乘积
        product = feat_a * feat_b
        
        # 3. 余弦相似度（标量）
        cos_sim = F.cosine_similarity(feat_a, feat_b, dim=1, eps=1e-8).unsqueeze(1)
        
        # 4. L2距离（标量）
        l2_dist = torch.norm(feat_a - feat_b, p=2, dim=1).unsqueeze(1)
        
        # 5. 特征均值差异（标量）
        mean_diff = torch.abs(torch.mean(feat_a, dim=1) - torch.mean(feat_b, dim=1)).unsqueeze(1)
        
        # 6. 特征方差比（标量）
        var_a = torch.var(feat_a, dim=1).unsqueeze(1) + 1e-8
        var_b = torch.var(feat_b, dim=1).unsqueeze(1) + 1e-8
        var_ratio = torch.min(var_a, var_b) / torch.max(var_a, var_b)
        
        # 拼接交互特征：差值 + 乘积 + 4个标量特征
        interaction_features = torch.cat([
            diff, product, cos_sim, l2_dist, mean_diff, var_ratio
        ], dim=1)
        
        return interaction_features


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # 残差连接
        return F.relu(out)


class OptimizedFusionMLP(nn.Module):
    """优化的交互特征融合MLP"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        """
        初始化优化的交互特征融合MLP
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout概率
        """
        super(OptimizedFusionMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        self.fusion = nn.Sequential(*layers)
        
        # 添加残差块
        self.residual_block = ResidualBlock(hidden_dims[-1], hidden_dims[-1] // 2, dropout)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 拼接后的交互特征张量
            
        Returns:
            融合后的特征张量
        """
        x = self.fusion(x)
        x = self.residual_block(x)
        return x


class OptimizedPOIMatchingMLP(nn.Module):
    """优化的POI匹配交互特征融合MLP模型"""
    
    def __init__(self, text_dim=384, llm_dim=384, geo_dim=32, gat_dim=128):
        """
        初始化优化的POI匹配模型
        
        Args:
            text_dim: 文本特征维度
            llm_dim: LLM特征维度
            geo_dim: 地理特征维度
            gat_dim: GAT特征维度
        """
        super(OptimizedPOIMatchingMLP, self).__init__()
        
        # 交互特征提取器
        self.interaction_extractor = OptimizedInteractionExtractor()
        
        # 计算交互特征维度
        # 每个模态交互: 差值(dim) + 乘积(dim) + 4个标量特征
        text_interaction_dim = text_dim * 2 + 4
        llm_interaction_dim = llm_dim * 2 + 4
        geo_interaction_dim = geo_dim * 2 + 4
        gat_interaction_dim = gat_dim * 2 + 4
        
        # Text-LLM增强交互维度
        enhanced_text_llm_dim = (text_dim + llm_dim) * 2 + 4
        
        total_interaction_dim = text_interaction_dim + llm_interaction_dim + geo_interaction_dim + gat_interaction_dim + enhanced_text_llm_dim
        
        # 特征降维层（减少过拟合风险）
        self.feature_reduction = nn.Sequential(
            nn.Linear(total_interaction_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 交互特征融合层
        self.fusion_layer = OptimizedFusionMLP(1024, [512, 256, 128], dropout=0.3)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b):
        """
        前向传播
        
        Args:
            text_a: POI A的文本特征
            text_b: POI B的文本特征
            llm_a: POI A的LLM特征
            llm_b: POI B的LLM特征
            geo_a: POI A的地理特征
            geo_b: POI B的地理特征
            gat_a: POI A的GAT特征
            gat_b: POI B的GAT特征
            
        Returns:
            匹配概率
        """
        # 1. 基础交互特征
        text_interaction = self.interaction_extractor(text_a, text_b)
        llm_interaction = self.interaction_extractor(llm_a, llm_b)
        geo_interaction = self.interaction_extractor(geo_a, geo_b)
        gat_interaction = self.interaction_extractor(gat_a, gat_b)
        
        # 2. Text-LLM增强交互
        text_llm_concat_a = torch.cat([text_a, llm_a], dim=1)
        text_llm_concat_b = torch.cat([text_b, llm_b], dim=1)
        enhanced_text_llm_interaction = self.interaction_extractor(text_llm_concat_a, text_llm_concat_b)
        
        # 3. 拼接所有交互特征
        all_interactions = torch.cat([
            text_interaction,
            llm_interaction, 
            geo_interaction,
            gat_interaction,
            enhanced_text_llm_interaction
        ], dim=1)
        
        # 4. 特征降维
        reduced_features = self.feature_reduction(all_interactions)
        
        # 5. 融合处理
        fusion_output = self.fusion_layer(reduced_features)
        
        # 6. 分类预测
        match_prob = self.classifier(fusion_output)
        
        return match_prob


class POIDataset(Dataset):
    """POI匹配数据集"""
    
    def __init__(self, text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b, labels):
        """
        初始化数据集
        
        Args:
            text_a: POI A的文本特征数组
            text_b: POI B的文本特征数组
            llm_a: POI A的LLM特征数组
            llm_b: POI B的LLM特征数组
            geo_a: POI A的地理特征数组
            geo_b: POI B的地理特征数组
            gat_a: POI A的GAT特征数组
            gat_b: POI B的GAT特征数组
            labels: 标签数组
        """
        self.text_a = torch.FloatTensor(text_a)
        self.text_b = torch.FloatTensor(text_b)
        self.llm_a = torch.FloatTensor(llm_a)
        self.llm_b = torch.FloatTensor(llm_b)
        self.geo_a = torch.FloatTensor(geo_a)
        self.geo_b = torch.FloatTensor(geo_b)
        self.gat_a = torch.FloatTensor(gat_a)
        self.gat_b = torch.FloatTensor(gat_b)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return (self.text_a[idx], self.text_b[idx], 
                self.llm_a[idx], self.llm_b[idx],
                self.geo_a[idx], self.geo_b[idx],
                self.gat_a[idx], self.gat_b[idx],
                self.labels[idx])


def load_cache_features():
    """
    加载缓存的特征数据
    
    Returns:
        cache_features: 缓存特征字典
    """
    print("\n🔄 加载缓存特征...")
    
    cache_dir = "outputs/cache"
    cache_features = {}
    
    # 加载GAT嵌入
    gat_zh_path = os.path.join(cache_dir, "gat_emb_zh.pkl")
    gat_en_path = os.path.join(cache_dir, "gat_emb_en.pkl")
    
    if os.path.exists(gat_zh_path):
        with open(gat_zh_path, 'rb') as f:
            cache_features['gat_emb_zh'] = pickle.load(f)
        print(f"✅ 加载GAT中文嵌入: {len(cache_features['gat_emb_zh'])} 个")
    
    if os.path.exists(gat_en_path):
        with open(gat_en_path, 'rb') as f:
            cache_features['gat_emb_en'] = pickle.load(f)
        print(f"✅ 加载GAT英文嵌入: {len(cache_features['gat_emb_en'])} 个")
    
    # 加载Qwen3增强特征
    enhanced_path = os.path.join(cache_dir, "enhanced_features_huggingface_models_Qwen3-0.6B.pkl")
    if os.path.exists(enhanced_path):
        with open(enhanced_path, 'rb') as f:
            cache_features['enhanced_features_qwen3'] = pickle.load(f)
        print("✅ 加载Qwen3增强特征")
    
    return cache_features


def load_poi_data():
    """
    加载POI数据
    
    Returns:
        bd_df, gd_df: 百度和高德POI数据框
    """
    print("\n🔄 加载POI数据...")
    
    bd_df = pd.read_csv("data/bd_chinese_data.csv")
    gd_df = pd.read_csv("data/gd_english_data.csv")
    
    print(f"✅ 百度中文POI: {len(bd_df)} 个")
    print(f"✅ 高德英文POI: {len(gd_df)} 个")
    
    return bd_df, gd_df


def load_annotations():
    """
    加载标注数据
    
    Returns:
        annotations_df: 合并后的标注数据框
    """
    print("\n🔄 加载标注数据...")
    
    # 加载三个标注文件
    match1_df = pd.read_csv("data/match1.csv")
    match2_df = pd.read_csv("data/match2.csv") 
    match3_df = pd.read_csv("data/match3.csv")
    
    print(f"✅ 加载 data/match1.csv: {len(match1_df)} 个标注")
    print(f"✅ 加载 data/match2.csv: {len(match2_df)} 个标注")
    print(f"✅ 加载 data/match3.csv: {len(match3_df)} 个标注")
    
    # 合并所有标注数据
    annotations_df = pd.concat([match1_df, match2_df, match3_df], ignore_index=True)
    
    print(f"✅ 总标注数据: {len(annotations_df)} 个")
    print(f"   - 正样本: {annotations_df['label'].sum()} 个")
    print(f"   - 负样本: {len(annotations_df) - annotations_df['label'].sum()} 个")
    
    return annotations_df


def split_enhanced_features(enhanced_vector):
    """
    分割增强特征向量
    
    Args:
        enhanced_vector: 增强特征向量 (800维)
        
    Returns:
        text_features: 文本特征 (384维)
        llm_features: LLM特征 (384维)  
        geo_features: 地理特征 (32维)
    """
    if len(enhanced_vector) != 800:
        raise ValueError(f"增强特征向量长度应为800，实际为{len(enhanced_vector)}")
    
    text_features = enhanced_vector[:384]      # 前384维是文本特征
    llm_features = enhanced_vector[384:768]    # 中间384维是LLM特征
    geo_features = enhanced_vector[768:800]    # 最后32维是地理特征
    
    return text_features, llm_features, geo_features


def create_interaction_features(cache_features, bd_df, gd_df, annotations_df):
    """
    创建交互特征数据
    
    Args:
        cache_features: 缓存特征字典
        bd_df: 百度POI数据框
        gd_df: 高德POI数据框
        annotations_df: 标注数据框
        
    Returns:
        text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b, labels: POI对的特征数组和标签
    """
    print("\n🔄 创建交互特征...")
    
    # 获取缓存特征
    enhanced_qwen3 = cache_features.get('enhanced_features_qwen3', {})
    gat_zh_features = cache_features.get('gat_emb_zh', {})  # 百度POI的GAT特征
    gat_en_features = cache_features.get('gat_emb_en', {})  # 高德POI的GAT特征
    
    # 分离百度和高德的增强特征
    qwen3_zh_features = enhanced_qwen3.get('zh', {})  # 百度POI的Qwen3特征
    qwen3_en_features = enhanced_qwen3.get('en', {})  # 高德POI的Qwen3特征
    
    print(f"Qwen3百度特征: {len(qwen3_zh_features)} 个")
    print(f"Qwen3高德特征: {len(qwen3_en_features)} 个")
    print(f"GAT百度特征: {len(gat_zh_features)} 个")
    print(f"GAT高德特征: {len(gat_en_features)} 个")
    
    if len(qwen3_zh_features) == 0 or len(qwen3_en_features) == 0:
        print("❌ 增强特征数据为空，无法创建特征")
        return None, None, None, None, None, None, None, None, None
    
    if len(gat_zh_features) == 0 or len(gat_en_features) == 0:
        print("❌ GAT特征数据为空，无法创建特征")
        return None, None, None, None, None, None, None, None, None
    
    text_a_list = []
    text_b_list = []
    llm_a_list = []
    llm_b_list = []
    geo_a_list = []
    geo_b_list = []
    gat_a_list = []
    gat_b_list = []
    labels = []
    
    missing_bd_features = 0
    missing_gd_features = 0
    missing_bd_gat = 0
    missing_gd_gat = 0
    successful_matches = 0
    
    # 处理每个标注样本
    for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df), desc="创建交互特征"):
        zh_id = int(row['zh_id'])
        en_id = int(row['en_id'])
        label = int(row['label'])
        
        # 获取增强特征
        if zh_id not in qwen3_zh_features:
            missing_bd_features += 1
            continue
        if en_id not in qwen3_en_features:
            missing_gd_features += 1
            continue
        
        # 获取GAT特征
        if zh_id not in gat_zh_features:
            missing_bd_gat += 1
            continue
        if en_id not in gat_en_features:
            missing_gd_gat += 1
            continue
            
        bd_enhanced = qwen3_zh_features[zh_id]
        gd_enhanced = qwen3_en_features[en_id]
        bd_gat = gat_zh_features[zh_id]
        gd_gat = gat_en_features[en_id]
        
        # 分割特征
        bd_text, bd_llm, bd_geo = split_enhanced_features(bd_enhanced)
        gd_text, gd_llm, gd_geo = split_enhanced_features(gd_enhanced)
        
        successful_matches += 1
        
        # 保存原始特征对
        text_a_list.append(bd_text)
        text_b_list.append(gd_text)
        llm_a_list.append(bd_llm)
        llm_b_list.append(gd_llm)
        geo_a_list.append(bd_geo)
        geo_b_list.append(gd_geo)
        gat_a_list.append(bd_gat)
        gat_b_list.append(gd_gat)
        labels.append(label)
    
    print(f"\n📊 特征创建统计:")
    print(f"   - 缺失百度增强特征: {missing_bd_features}")
    print(f"   - 缺失高德增强特征: {missing_gd_features}")
    print(f"   - 缺失百度GAT特征: {missing_bd_gat}")
    print(f"   - 缺失高德GAT特征: {missing_gd_gat}")
    print(f"   - 成功创建的样本: {successful_matches}")
    
    if len(text_a_list) == 0:
        print("❌ 未能创建任何特征向量")
        return None, None, None, None, None, None, None, None, None
        
    text_a = np.array(text_a_list)
    text_b = np.array(text_b_list)
    llm_a = np.array(llm_a_list)
    llm_b = np.array(llm_b_list)
    geo_a = np.array(geo_a_list)
    geo_b = np.array(geo_b_list)
    gat_a = np.array(gat_a_list)
    gat_b = np.array(gat_b_list)
    labels = np.array(labels, dtype=int)
    
    print("✅ 成功创建交互特征")
    print(f"   - 文本特征A形状: {text_a.shape}")
    print(f"   - 文本特征B形状: {text_b.shape}")
    print(f"   - LLM特征A形状: {llm_a.shape}")
    print(f"   - LLM特征B形状: {llm_b.shape}")
    print(f"   - 地理特征A形状: {geo_a.shape}")
    print(f"   - 地理特征B形状: {geo_b.shape}")
    print(f"   - GAT特征A形状: {gat_a.shape}")
    print(f"   - GAT特征B形状: {gat_b.shape}")
    print(f"   - 标签分布: {np.bincount(labels)}")
    
    return text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b, labels


def train_model(text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b, labels, device='cpu'):
    """
    训练POI匹配模型
    
    Args:
        text_a, text_b: POI对的文本特征数组
        llm_a, llm_b: POI对的LLM特征数组
        geo_a, geo_b: POI对的地理特征数组
        gat_a, gat_b: POI对的GAT特征数组
        labels: 标签数组
        device: 计算设备
        
    Returns:
        model: 训练好的模型
        results: 评估结果字典
    """
    print("\n🔄 训练POI匹配模型...")
    
    # 划分训练测试集
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 获取训练测试数据
    text_a_train, text_a_test = text_a[train_indices], text_a[test_indices]
    text_b_train, text_b_test = text_b[train_indices], text_b[test_indices]
    llm_a_train, llm_a_test = llm_a[train_indices], llm_a[test_indices]
    llm_b_train, llm_b_test = llm_b[train_indices], llm_b[test_indices]
    geo_a_train, geo_a_test = geo_a[train_indices], geo_a[test_indices]
    geo_b_train, geo_b_test = geo_b[train_indices], geo_b[test_indices]
    gat_a_train, gat_a_test = gat_a[train_indices], gat_a[test_indices]
    gat_b_train, gat_b_test = gat_b[train_indices], gat_b[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    
    print(f"训练集大小: {len(y_train)}")
    print(f"测试集大小: {len(y_test)}")
    
    # 创建数据集和数据加载器 - 性能优化版本
    train_dataset = POIDataset(text_a_train, text_b_train, llm_a_train, llm_b_train, geo_a_train, geo_b_train, gat_a_train, gat_b_train, y_train)
    test_dataset = POIDataset(text_a_test, text_b_test, llm_a_test, llm_b_test, geo_a_test, geo_b_test, gat_a_test, gat_b_test, y_test)
    
    # 大幅增加批次大小以充分利用GPU内存
    train_loader = DataLoader(
        train_dataset, 
        batch_size=512,  # 进一步增加到512
        shuffle=True, 
        num_workers=4,   # 多进程数据加载
        pin_memory=True, # 固定内存，加速GPU传输
        persistent_workers=True  # 保持worker进程，减少启动开销
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1024,  # 测试时用更大的批次
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # 初始化模型
    model = OptimizedPOIMatchingMLP(
        text_dim=text_a.shape[1],
        llm_dim=llm_a.shape[1], 
        geo_dim=geo_a.shape[1],
        gat_dim=gat_a.shape[1]
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    
    # 训练参数
    num_epochs = 500  # 保持较多的训练轮数
    best_f1 = 0
    patience = 15
    patience_counter = 0
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        
        # 添加训练进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for text_a_batch, text_b_batch, llm_a_batch, llm_b_batch, geo_a_batch, geo_b_batch, gat_a_batch, gat_b_batch, labels_batch in train_pbar:
            text_a_batch = text_a_batch.to(device)
            text_b_batch = text_b_batch.to(device)
            llm_a_batch = llm_a_batch.to(device)
            llm_b_batch = llm_b_batch.to(device)
            geo_a_batch = geo_a_batch.to(device)
            geo_b_batch = geo_b_batch.to(device)
            gat_a_batch = gat_a_batch.to(device)
            gat_b_batch = gat_b_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(text_a_batch, text_b_batch, llm_a_batch, llm_b_batch, geo_a_batch, geo_b_batch, gat_a_batch, gat_b_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        
        # 更新学习率
        scheduler.step()
        
        # 每5个epoch进行一次验证
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_predictions = []
            val_probabilities = []
            val_labels = []
            
            with torch.no_grad():
                for text_a_batch, text_b_batch, llm_a_batch, llm_b_batch, geo_a_batch, geo_b_batch, gat_a_batch, gat_b_batch, labels_batch in test_loader:
                    text_a_batch = text_a_batch.to(device)
                    text_b_batch = text_b_batch.to(device)
                    llm_a_batch = llm_a_batch.to(device)
                    llm_b_batch = llm_b_batch.to(device)
                    geo_a_batch = geo_a_batch.to(device)
                    geo_b_batch = geo_b_batch.to(device)
                    gat_a_batch = gat_a_batch.to(device)
                    gat_b_batch = gat_b_batch.to(device)
                    
                    outputs = model(text_a_batch, text_b_batch, llm_a_batch, llm_b_batch, geo_a_batch, geo_b_batch, gat_a_batch, gat_b_batch)
                    predictions = (outputs > 0.5).float().cpu().numpy()
                    probabilities = outputs.cpu().numpy()
                    
                    val_predictions.extend(predictions.flatten())
                    val_probabilities.extend(probabilities.flatten())
                    val_labels.extend(labels_batch.cpu().numpy())
            
            # 计算评估指标
            val_f1 = f1_score(val_labels, val_predictions)
            val_auc = roc_auc_score(val_labels, val_probabilities)
            val_acc = accuracy_score(val_labels, val_predictions)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f} - Val F1: {val_f1:.4f} - Val AUC: {val_auc:.4f} - Val Acc: {val_acc:.4f}")
            
            # 早停检查
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'outputs/best_optimized_interaction_fusion_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停：验证F1分数连续{patience}个epoch未提升")
                break
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('outputs/best_optimized_interaction_fusion_model.pth'))
    model.eval()
    
    final_predictions = []
    final_probabilities = []
    final_labels = []
    
    with torch.no_grad():
        for text_a_batch, text_b_batch, llm_a_batch, llm_b_batch, geo_a_batch, geo_b_batch, gat_a_batch, gat_b_batch, labels_batch in test_loader:
            text_a_batch = text_a_batch.to(device)
            text_b_batch = text_b_batch.to(device)
            llm_a_batch = llm_a_batch.to(device)
            llm_b_batch = llm_b_batch.to(device)
            geo_a_batch = geo_a_batch.to(device)
            geo_b_batch = geo_b_batch.to(device)
            gat_a_batch = gat_a_batch.to(device)
            gat_b_batch = gat_b_batch.to(device)
            
            outputs = model(text_a_batch, text_b_batch, llm_a_batch, llm_b_batch, geo_a_batch, geo_b_batch, gat_a_batch, gat_b_batch)
            predictions = (outputs > 0.5).float().cpu().numpy()
            probabilities = outputs.cpu().numpy()
            
            final_predictions.extend(predictions.flatten())
            final_probabilities.extend(probabilities.flatten())
            final_labels.extend(labels_batch.cpu().numpy())
    
    # 最终评估
    final_f1 = f1_score(final_labels, final_predictions)
    final_auc = roc_auc_score(final_labels, final_probabilities)
    final_acc = accuracy_score(final_labels, final_predictions)
    final_precision = precision_score(final_labels, final_predictions)
    final_recall = recall_score(final_labels, final_predictions)
    
    print(f"\n📊 最终评估结果:")
    print(f"   - F1分数: {final_f1:.4f}")
    print(f"   - AUC分数: {final_auc:.4f}")
    print(f"   - 准确率: {final_acc:.4f}")
    print(f"   - 精确率: {final_precision:.4f}")
    print(f"   - 召回率: {final_recall:.4f}")
    
    results = {
        'f1_score': final_f1,
        'auc_score': final_auc,
        'accuracy': final_acc,
        'precision': final_precision,
        'recall': final_recall
    }
    
    return model, results


def main():
    """主函数"""
    print("🚀 开始优化交互特征融合MLP实验 V2")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    cache_features = load_cache_features()
    bd_df, gd_df = load_poi_data()
    annotations_df = load_annotations()
    
    if not cache_features:
        print("❌ 无法加载缓存特征，退出程序")
        return
    
    # 创建交互特征
    text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b, labels = create_interaction_features(
        cache_features, bd_df, gd_df, annotations_df
    )
    
    if text_a is None:
        print("❌ 无法创建交互特征，退出程序")
        return
    
    # 训练模型
    model, results = train_model(text_a, text_b, llm_a, llm_b, geo_a, geo_b, gat_a, gat_b, labels, device)
    
    print("\n✅ 优化交互特征融合MLP实验 V2 完成")
    print(f"最终结果: F1={results['f1_score']:.4f}, AUC={results['auc_score']:.4f}, Acc={results['accuracy']:.4f}")


if __name__ == "__main__":
    main()