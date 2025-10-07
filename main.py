import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pickle
import os
import geohash2
from tqdm import tqdm
import numpy as np
import warnings
from haversine import haversine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import lightgbm as lgb
# 使用BallTree进行高效范围查询
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestClassifier
import sys
import argparse
from datetime import datetime

# 导入LLM文本增强模块
from models.llm_text_enhancer import LLMTextEnhancer

warnings.filterwarnings("ignore", category=UserWarning, module='torch_geometric.nn.conv.gatv2_conv')

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='POI匹配系统 - 支持多种LLM模型')
    
    # LLM模型相关参数
    parser.add_argument('--model-type', type=str, default='huggingface', 
                       choices=['huggingface', 'ollama'],
                       help='LLM模型类型 (默认: huggingface)')
    
    parser.add_argument('--model-name', type=str, default='models/DialoGPT-small',
                       help='模型名称或路径 (默认: models/DialoGPT-small)')
    
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='OLLAMA服务URL (默认: http://localhost:11434)')
    
    # 其他参数
    parser.add_argument('--debug', action='store_true', default=False,
                       help='启用debug模式，只处理少量数据')
    
    parser.add_argument('--debug-limit', type=int, default=1000,
                       help='debug模式下的数据限制 (默认: 1000)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备 (默认: cuda)')
    
    return parser.parse_args()

# 解析命令行参数
args = parse_arguments()

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 输入数据
BD_CHINESE_POI_PATH = os.path.join(DATA_DIR, 'bd_chinese_data.csv')  # zh对应百度
GD_CHINESE_POI_PATH = os.path.join(DATA_DIR, 'gd_chinese_data.csv')  # en对应高德
LABELED_DATA_PATH = os.path.join(DATA_DIR, 'match2.csv')  # 包含en_id(高德), zh_id(百度), label

# 配置参数 - 从命令行参数获取
DEBUG_MODE = args.debug  # 从命令行参数获取
DEBUG_DATA_LIMIT = args.debug_limit  # 从命令行参数获取

# LLM模型配置
LLM_MODEL_TYPE = args.model_type
LLM_MODEL_NAME = args.model_name
OLLAMA_BASE_URL = args.ollama_url
DEVICE = args.device

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# 输出文件路径
FINAL_MATCHES_PATH = os.path.join(OUTPUT_DIR, 'final_matched_pairs.csv')

CLASSIFIER_MODEL_PATH = os.path.join(OUTPUT_DIR, 'gat_classifier_model.pkl')
EVALUATION_REPORT_PATH = os.path.join(OUTPUT_DIR, 'model_evaluation_report.txt')

# 缓存路径配置
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
TEXT_EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'text_embeddings.pkl')
GAT_EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, 'gat_embeddings.pkl')
ENHANCED_FEATURES_CACHE = os.path.join(CACHE_DIR, 'enhanced_features.pkl')

# --- 模型与处理参数 ---
TEXT_EMBEDDING_BATCH_SIZE = 256
GEOHASH_PRECISION = 7  # 约等于 153m x 153m 的格子
GAT_OUTPUT_DIM = 128   # GAT输出的嵌入维度
MATCHING_BATCH_SIZE = 1024

# --- GAT 模型定义 ---
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 32, heads=8, dropout=0.2)
        self.conv2 = GATConv(32 * 8, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# --- 1. 生成高质量文本嵌入 ---
def generate_text_embeddings(file_path, model, device):
    print(f"\n--- 步骤 1: 开始为 {os.path.basename(file_path)} 生成文本嵌入 ---")
    df = pd.read_csv(file_path)
    
    if 'id' not in df.columns:
        print(f"警告: 在 {os.path.basename(file_path)} 中未找到 'id' 列。将自动生成基于行号的ID。")
        df['id'] = range(len(df))
    
    required_cols = ['name', 'category', 'latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: 文件 {os.path.basename(file_path)} 中缺少以下必要列: {missing_cols}")
        return pd.DataFrame(), {}
    
    df.dropna(subset=['id', 'name', 'address', 'category', 'latitude', 'longitude'], inplace=True)
    df['id'] = df['id'].astype(int)
    
    # Debug模式：智能选择能够形成完整匹配对的数据
    if DEBUG_MODE:
        original_len = len(df)
        
        # 读取标注数据
        labeled_data = pd.read_csv(LABELED_DATA_PATH)
        
        # 根据文件名判断是百度还是高德数据
        if 'bd_chinese' in file_path:
            # 百度数据，获取zh_id
            # 首先选择前DEBUG_DATA_LIMIT个标注对，然后提取对应的百度ID
            selected_pairs = labeled_data.head(DEBUG_DATA_LIMIT)
            required_ids = set(selected_pairs['zh_id'].unique())
            data_type = "百度"
        else:
            # 高德数据，获取en_id
            # 选择相同的标注对，提取对应的高德ID
            selected_pairs = labeled_data.head(DEBUG_DATA_LIMIT)
            required_ids = set(selected_pairs['en_id'].unique())
            data_type = "高德"
        
        # 筛选包含标注数据ID的行
        df_required = df[df['id'].isin(required_ids)]
        
        # 如果找到的POI数量不足，补充一些其他数据
        if len(df_required) < DEBUG_DATA_LIMIT:
            remaining_limit = DEBUG_DATA_LIMIT - len(df_required)
            df_other = df[~df['id'].isin(required_ids)].head(remaining_limit)
            df = pd.concat([df_required, df_other], ignore_index=True)
            print(f"🐛 DEBUG模式({data_type}): 包含 {len(df_required)} 个匹配对相关POI + {len(df_other)} 个其他POI，共 {len(df)} 条")
        else:
            df = df_required.head(DEBUG_DATA_LIMIT)
            print(f"🐛 DEBUG模式({data_type}): 选择 {len(df)} 个匹配对相关POI")
        
        print(f"   原始数据: {original_len} 条 -> DEBUG处理: {len(df)} 条")
        print(f"   选择的{data_type}ID示例: {list(df['id'].head(10))}")
    
    # 拼接文本
    df['text_to_embed'] = df['name'].astype(str) + " | " + df['category'].astype(str) + " | " + df['address'].astype(str)
    
    print(f"共 {len(df)} 条有效POI。")
    embeddings = model.encode(
        df['text_to_embed'].tolist(),
        batch_size=TEXT_EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        device=device,
        convert_to_numpy=True
    )
    
    embeddings_dict = {row['id']: emb for _, row, emb in zip(range(len(df)), df.to_dict('records'), embeddings)}
    
    return df, embeddings_dict

import torch
import torch.nn as nn

# --- 2. 添加Geohash编码 ---
def add_geohash(df):
    print("\n--- 步骤 2: 添加Geohash编码 ---")

     # 检查DataFrame是否为空
    if df.empty:
        print("警告: DataFrame为空，无法添加Geohash")
        return df
    
    # 检查是否包含经纬度列
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        print("警告: DataFrame缺少经纬度列，无法添加Geohash")
        return df
    
    df['geohash'] = df.apply(
        lambda row: geohash2.encode(row['latitude'], row['longitude'], precision=GEOHASH_PRECISION),
        axis=1
    )
    return df

# --- Geohash嵌入 --- 
class GeohashEncoder(nn.Module):
    def __init__(self, charset_size, embedding_dim, hidden_dim, output_dim, device):
        """
        使用BiGRU编码Geohash字符串
        :param charset_size: Geohash字符集大小（Base32共32个字符）
        :param embedding_dim: 字符嵌入维度
        :param hidden_dim: GRU隐藏层维度
        :param output_dim: 最终输出维度
        :param device: 计算设备
        """
        super(GeohashEncoder, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 字符嵌入层
        self.embedding = nn.Embedding(charset_size, embedding_dim).to(device)
        
        # 双向GRU
        self.bigru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        ).to(device)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim * 2, output_dim).to(device)  # 双向所以是2倍
        
    def forward(self, geohash_indices):
        """
        :param geohash_indices: 已转换为索引序列的Geohash Tensor
        :return: Geohash的嵌入表示
        """
        # 字符嵌入
        embedded = self.embedding(geohash_indices)
        
        # 通过双向GRU
        gru_output, hidden = self.bigru(embedded)
        
        # 取最后时间步的前向和后向隐藏状态，拼接后通过全连接层
        forward_last = hidden[-2, :, :]  # 前向最后隐藏状态
        backward_last = hidden[-1, :, :]  # 后向最后隐藏状态
        combined = torch.cat((forward_last, backward_last), dim=1)
        
        output = self.fc(combined)
        return output


# --- 3. 构建图并生成GAT嵌入(使用类别的onehot编码作为节点) ---
def generate_gat_embeddings(df, device):
    # 使用K—nn构图
    print("\n--- 步骤 3: 构建图并生成GAT嵌入 (使用K邻近构图) ---")

    # 准备节点特征 - 使用类别信息
    print("准备节点特征 (使用类别信息)...")

    # 创建类别到索引的映射
    categories = df['category'].unique()
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # 创建节点特征 - 使用类别索引
    ids = df['id'].tolist()
    node_features = np.array([cat_to_idx[row['category']] for _, row in df.iterrows()])

    # 将类别索引转换为one-hot编码
    num_categories = len(categories)
    node_features = np.eye(num_categories)[node_features].astype('float32')

    x = torch.tensor(node_features).to(device)

    # 构建图：基于K邻近连接节点
    print("构建K邻近图...")
    df['node_idx'] = range(len(df))

    # 提取坐标并转换为numpy数组
    coords = df[['latitude', 'longitude']].values

    # 使用BallTree进行高效K邻近查询
    from sklearn.neighbors import BallTree
    tree = BallTree(coords, metric='haversine')

    # 查询每个点的10个最近邻居（包括自身）
    K = 5
    distances, indices = tree.query(coords, k=K+1)  # +1 因为包括自身

    source_nodes, target_nodes = [], []

    # 遍历每个节点，添加与邻居的边
    for i in tqdm(range(len(indices)), desc="构建边"):
        # 跳过自身（索引0是自身）
        for j in indices[i][1:]:
            source_nodes.append(i)
            target_nodes.append(j)
            # 添加反向边（无向图）
            source_nodes.append(j)
            target_nodes.append(i)

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    print(f"图构建完成，共 {len(ids)} 个节点, {edge_index.size(1)} 条边。")

    # 运行GAT模型 - 使用CPU避免GPU内存不足
    print("由于节点数量较大，使用CPU处理GAT模型以避免GPU内存不足...")
    x_cpu = x.cpu()
    edge_index_cpu = edge_index
    data = Data(x=x_cpu, edge_index=edge_index_cpu)
    model = GAT(in_channels=x_cpu.size(1), out_channels=GAT_OUTPUT_DIM)  # 保持在CPU
    model.eval()

    print("通过GAT模型生成最终嵌入...")
    with torch.no_grad():
        gat_embeddings_tensor = model(data.x, data.edge_index)

    gat_embeddings = gat_embeddings_tensor.cpu().numpy()
    gat_embeddings_dict = {id: emb for id, emb in zip(ids, gat_embeddings)}

    return gat_embeddings_dict


# --- 4. 获得所有嵌入 ---
def create_poi_enhanced_features_from_cache(df, cache_dir, debug_mode=False, debug_limit=1000, model_name="DialoGPT-small"):
    """
    从缓存文件中加载并创建加权融合的增强特征向量
    
    Args:
        df: POI数据框
        cache_dir: 缓存目录路径
        debug_mode: 是否为调试模式
        debug_limit: 调试模式下的数据限制
        model_name: LLM模型名称
    
    Returns:
        enhanced_features: 增强特征字典
    """
    enhanced_features = {}
    
    # 构建缓存文件路径
    debug_suffix = f"_debug_{debug_limit}" if debug_mode else ""
    model_suffix = f"_huggingface_models_{model_name.replace('/', '_')}"
    
    # 文件路径
    text_emb_file = os.path.join(cache_dir, f"text_emb_zh{debug_suffix}.pkl")
    gat_emb_file = os.path.join(cache_dir, f"gat_emb_zh{debug_suffix}.pkl") 
    enhanced_file = os.path.join(cache_dir, f"enhanced_features{debug_suffix}{model_suffix}.pkl")
    
    print(f"从缓存加载特征文件...")
    print(f"文本嵌入: {text_emb_file}")
    print(f"GAT嵌入: {gat_emb_file}")
    print(f"增强特征: {enhanced_file}")
    
    # 加载各种特征
    try:
        with open(text_emb_file, 'rb') as f:
            text_emb_dict = pickle.load(f)
        print(f"✓ 文本嵌入加载完成: {len(text_emb_dict)} 个特征")
        
        with open(gat_emb_file, 'rb') as f:
            gat_emb_dict = pickle.load(f)
        print(f"✓ GAT嵌入加载完成: {len(gat_emb_dict)} 个特征")
        
        with open(enhanced_file, 'rb') as f:
            enhanced_features_cache = pickle.load(f)
        print(f"✓ 增强特征加载完成: {len(enhanced_features_cache)} 个特征")
        
    except FileNotFoundError as e:
        print(f"缓存文件未找到: {e}")
        return None
    
    # 解耦特征并实现加权融合
    print("\n开始特征解耦和加权融合...")
    
    # 特征权重设置（经纬度 > LLM > 文本）
    geohash_weight = 0.5    # 经纬度特征权重最高
    llm_weight = 0.3        # LLM特征权重次之
    text_weight = 0.15      # 文本特征权重较低
    gat_weight = 0.05       # GAT特征权重最低
    
    print(f"特征权重设置:")
    print(f"  - Geohash (经纬度): {geohash_weight}")
    print(f"  - LLM特征: {llm_weight}")
    print(f"  - 文本特征: {text_weight}")
    print(f"  - GAT特征: {gat_weight}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建加权融合特征"):
        poi_id = row['id']
        
        if poi_id not in enhanced_features_cache:
            continue
            
        # 从缓存的增强特征中解耦各部分
        cached_feature = enhanced_features_cache[poi_id]
        
        # 获取各个特征组件
        text_vector = text_emb_dict.get(poi_id)
        gat_vector = gat_emb_dict.get(poi_id)
        
        if text_vector is None or gat_vector is None:
            continue
        
        # 从缓存特征中提取各部分
        text_dim = len(text_vector)
        gat_dim = len(gat_vector)
        
        # 假设缓存特征的结构是: [text, geohash, llm]
        geohash_start = text_dim
        geohash_end = geohash_start + 64  # Geohash编码器输出64维
        llm_start = geohash_end
        
        # 提取各部分特征
        cached_text = cached_feature[:text_dim]
        cached_geohash = cached_feature[geohash_start:geohash_end]
        cached_llm = cached_feature[llm_start:] if llm_start < len(cached_feature) else np.array([])
        
        # 标准化各特征到相同尺度
        def normalize_feature(feat):
            if len(feat) == 0:
                return feat
            norm = np.linalg.norm(feat)
            return feat / norm if norm > 0 else feat
        
        norm_text = normalize_feature(cached_text)
        norm_geohash = normalize_feature(cached_geohash)
        norm_gat = normalize_feature(gat_vector)
        norm_llm = normalize_feature(cached_llm) if len(cached_llm) > 0 else np.array([])
        
        # 加权融合
        weighted_features = []
        
        # 添加加权的各个特征
        if len(norm_geohash) > 0:
            weighted_features.append(norm_geohash * geohash_weight)
        
        if len(norm_llm) > 0:
            weighted_features.append(norm_llm * llm_weight)
            
        if len(norm_text) > 0:
            weighted_features.append(norm_text * text_weight)
            
        if len(norm_gat) > 0:
            weighted_features.append(norm_gat * gat_weight)
        
        # 拼接所有加权特征
        if weighted_features:
            combined_vector = np.concatenate(weighted_features)
            enhanced_features[poi_id] = combined_vector
    
    print(f"✓ 加权融合特征构建完成: {len(enhanced_features)} 个特征")
    return enhanced_features


def create_poi_enhanced_features(df, text_emb_dict, gat_emb_dict, geohash_encoder, device, llm_enhancer=None):
    """
    为每个POI创建增强特征向量：拼接文本、Geohash（BiGRU编码）、图向量和LLM增强特征
    
    Args:
        df: POI数据框
        text_emb_dict: 文本嵌入字典
        gat_emb_dict: GAT嵌入字典
        geohash_encoder: Geohash编码器
        device: 计算设备
        llm_enhancer: LLM文本增强器（可选）
    
    Returns:
        enhanced_features: 增强特征字典
    """
    enhanced_features = {}
    
    # Base32字符集：Geohash使用的32个字符
    base32_chars = "0123456789bcdefghjkmnpqrstuvwxyz"
    char_to_idx = {char: idx for idx, char in enumerate(base32_chars)}
    
    # 如果提供了LLM增强器，批量生成LLM增强特征
    llm_enhanced_features = {}
    if llm_enhancer is not None:
        print("生成LLM增强特征...")
        poi_texts = []
        poi_ids = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="准备POI数据"):
            poi_id = row['id']
            poi_name = str(row.get('name', ''))
            poi_address = str(row.get('address', ''))
            poi_category = str(row.get('category', ''))
            
            poi_texts.append({
                'name': poi_name,
                'address': poi_address,
                'category': poi_category
            })
            poi_ids.append(poi_id)
        
        # 批量生成LLM增强特征
        try:
            # 将POI字典转换为文本列表
            poi_text_list = []
            for poi_dict in poi_texts:
                poi_text = f"{poi_dict['name']} | {poi_dict['category']} | {poi_dict['address']}"
                poi_text_list.append(poi_text)
            
            print(f"开始生成 {len(poi_text_list)} 个POI的LLM增强特征...")
            llm_features = llm_enhancer.batch_generate_enhanced_features(poi_text_list)
            for poi_id, llm_feat in zip(poi_ids, llm_features):
                llm_enhanced_features[poi_id] = llm_feat
            print(f"LLM增强特征生成完成，共 {len(llm_enhanced_features)} 个特征")
        except Exception as e:
            print(f"LLM增强特征生成失败: {e}")
            print("将使用原始特征继续...")
    
    for _, row in df.iterrows():
        poi_id = row['id']
        
        # 1. 获取文本向量
        text_vector = text_emb_dict[poi_id]
        
        # 2. 获取GAT图向量
        gat_vector = gat_emb_dict[poi_id]
        
        # 3. 生成Geohash向量（使用BiGRU编码）
        geohash_str = row['geohash']
        
        # 将Geohash字符串转换为索引序列
        geohash_indices = [char_to_idx.get(char, 0) for char in geohash_str]
        geohash_indices = torch.tensor(geohash_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # 通过BiGRU编码器获取Geohash向量
        with torch.no_grad():
            geohash_vector = geohash_encoder(geohash_indices).squeeze().cpu().numpy()
        
        # 4. 获取LLM增强特征（如果可用）
        if poi_id in llm_enhanced_features:
            llm_vector = llm_enhanced_features[poi_id]
            # 拼接所有特征：原始文本 + Geohash + LLM增强特征
            combined_vector = np.concatenate([text_vector, geohash_vector, llm_vector])
        else:
            # 如果没有LLM增强特征，使用原始特征
            combined_vector = np.concatenate([text_vector, geohash_vector])
 
        enhanced_features[poi_id] = combined_vector
    
    return enhanced_features


def train_classifier_with_enhanced_features(enhanced_feat_en, enhanced_feat_zh, labeled_data_path):
    print("\n--- 步骤 4: 使用增强特征训练分类器 ---")
    
    # 加载标注数据
    labeled_data = pd.read_csv(labeled_data_path)
    
    # 准备特征和标签
    features = []
    labels = []
    
    print("为标注对构建联合特征...")
    for _, row in tqdm(labeled_data.iterrows(), total=len(labeled_data)):
        # en_id对应高德，zh_id对应百度
        en_id = row['en_id']  # 高德ID
        zh_id = row['zh_id']  # 百度ID
        label = row['label']
        
        # 获取两个POI的增强特征
        en_feat = enhanced_feat_en.get(en_id)  # 高德特征
        zh_feat = enhanced_feat_zh.get(zh_id)  # 百度特征
        
        if en_feat is None or zh_feat is None:
            continue
        
        # 拼接两个POI的特征作为分类器输入
        combined_feature = np.concatenate([en_feat, zh_feat])
        features.append(combined_feature)
        labels.append(label)
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"特征维度: {features.shape}")
    print(f"正样本数: {sum(labels)}, 负样本数: {len(labels) - sum(labels)}")
    
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 训练LightGBM
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        verbosity=1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )
    
    # 评估模型
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n分类报告:")
    print(report)
    print(f"F1分数: {f1:.4f}")
    print(f"AUC分数: {auc:.4f}")
    
    return model

# --- 主流程 ---
# --- 主流程 ---
def main():
    print("====== 开始执行POI匹配模型训练 ======")
    
    if DEBUG_MODE:
        print(f"🐛 DEBUG模式已启用 - 只处理前 {DEBUG_DATA_LIMIT} 条数据")
        print("   这将大大加快处理速度，适合测试和调试")
    
    # 设置设备
    device = DEVICE if torch.cuda.is_available() or DEVICE == 'cpu' else 'cpu'
    print(f"使用的设备: {device}")
    print(f"LLM模型类型: {LLM_MODEL_TYPE}")
    print(f"LLM模型名称: {LLM_MODEL_NAME}")
    
    # 加载文本编码模型
    text_model = SentenceTransformer(MODEL_PATH, device=device)
    
    # 初始化Geohash编码器
    base32_chars = "0123456789bcdefghjkmnpqrstuvwxyz"
    charset_size = len(base32_chars)
    geohash_encoder = GeohashEncoder(
        charset_size=charset_size,
        embedding_dim=8,      # 字符嵌入维度
        hidden_dim=16,       # GRU隐藏层维度
        output_dim=32,       # 输出维度
        device=device
    )
    geohash_encoder.eval()   # 设置为评估模式
    
    # 初始化LLM文本增强器（可选）
    llm_enhancer = None
    
    try:
        print("\n初始化LLM文本增强器...")
        
        if LLM_MODEL_TYPE == "ollama":
            # 使用OLLAMA模型
            print(f"使用OLLAMA模型: {LLM_MODEL_NAME}")
            llm_enhancer = LLMTextEnhancer(
                llm_model_path=LLM_MODEL_NAME,
                text_encoder_name="paraphrase-multilingual-MiniLM-L12-v2",
                use_llm=True,
                device=device,
                target_dim=384,
                model_type="ollama",
                ollama_base_url=OLLAMA_BASE_URL
            )
        else:
            # 使用HuggingFace模型
            if LLM_MODEL_NAME.startswith("microsoft/") or LLM_MODEL_NAME.startswith("gpt"):
                # 在线模型
                model_path = LLM_MODEL_NAME
            elif os.path.isabs(LLM_MODEL_NAME):
                # 绝对路径
                model_path = LLM_MODEL_NAME
            else:
                # 相对路径，需要拼接
                if LLM_MODEL_NAME.startswith("models/"):
                    # 如果已经包含models/前缀，直接使用PROJECT_ROOT拼接
                    model_path = os.path.join(PROJECT_ROOT, LLM_MODEL_NAME)
                else:
                    # 否则使用MODEL_DIR拼接
                    model_path = os.path.join(MODEL_DIR, LLM_MODEL_NAME)
            
            print(f"使用HuggingFace模型: {model_path}")
            llm_enhancer = LLMTextEnhancer(
                llm_model_path=model_path,
                text_encoder_name="paraphrase-multilingual-MiniLM-L12-v2",
                use_llm=True,
                device=device,
                target_dim=384,
                model_type="huggingface"
            )
        
        print("LLM文本增强器初始化成功")
    except Exception as e:
        print(f"LLM文本增强器初始化失败: {e}")
        print("将使用原始特征继续训练...")
    
    # --- 处理百度中文数据 (zh对应百度) ---
    print("\n处理百度中文POI数据...")
    
    # 尝试加载缓存的文本嵌入
    cache_suffix = f"_debug_{DEBUG_DATA_LIMIT}" if DEBUG_MODE else ""
    zh_cache_path = os.path.join(CACHE_DIR, f'text_emb_zh{cache_suffix}.pkl')
    if os.path.exists(zh_cache_path):
        print(f"加载缓存的百度文本嵌入{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(zh_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            df_zh, text_emb_zh = cache_data['df'], cache_data['embeddings']
    else:
        df_zh, text_emb_zh = generate_text_embeddings(BD_CHINESE_POI_PATH, text_model, device)
        # 保存到缓存
        print(f"保存百度文本嵌入到缓存{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(zh_cache_path, 'wb') as f:
            pickle.dump({'df': df_zh, 'embeddings': text_emb_zh}, f)
    
    df_zh = add_geohash(df_zh)  # 使用 add_geohash 函数添加 geohash 列
    
    # 尝试加载缓存的GAT嵌入
    zh_gat_cache_path = os.path.join(CACHE_DIR, f'gat_emb_zh{cache_suffix}.pkl')
    if os.path.exists(zh_gat_cache_path):
        print(f"加载缓存的百度GAT嵌入{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(zh_gat_cache_path, 'rb') as f:
            gat_emb_zh = pickle.load(f)
    else:
        gat_emb_zh = generate_gat_embeddings(df_zh, device)
        # 保存到缓存
        print(f"保存百度GAT嵌入到缓存{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(zh_gat_cache_path, 'wb') as f:
            pickle.dump(gat_emb_zh, f)
    
    # --- 处理高德中文数据 (en对应高德) ---
    print("\n处理高德中文POI数据...")
    
    # 尝试加载缓存的文本嵌入
    en_cache_path = os.path.join(CACHE_DIR, f'text_emb_en{cache_suffix}.pkl')
    if os.path.exists(en_cache_path):
        print(f"加载缓存的高德文本嵌入{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(en_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            df_en, text_emb_en = cache_data['df'], cache_data['embeddings']
    else:
        df_en, text_emb_en = generate_text_embeddings(GD_CHINESE_POI_PATH, text_model, device)
        # 保存到缓存
        print(f"保存高德文本嵌入到缓存{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(en_cache_path, 'wb') as f:
            pickle.dump({'df': df_en, 'embeddings': text_emb_en}, f)
    
    df_en = add_geohash(df_en)  # 使用 add_geohash 函数添加 geohash 列
    
    # 尝试加载缓存的GAT嵌入
    en_gat_cache_path = os.path.join(CACHE_DIR, f'gat_emb_en{cache_suffix}.pkl')
    if os.path.exists(en_gat_cache_path):
        print(f"加载缓存的高德GAT嵌入{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(en_gat_cache_path, 'rb') as f:
            gat_emb_en = pickle.load(f)
    else:
        gat_emb_en = generate_gat_embeddings(df_en, device)
        # 保存到缓存
        print(f"保存高德GAT嵌入到缓存{' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(en_gat_cache_path, 'wb') as f:
            pickle.dump(gat_emb_en, f)

    
    # --- 创建增强特征 ---
    print("\n创建增强特征...")
    
    # 根据模型类型和名称生成缓存文件名
    model_cache_suffix = f"_{LLM_MODEL_TYPE}_{LLM_MODEL_NAME.replace('/', '_').replace(':', '_')}"
    enhanced_features_cache_path = os.path.join(CACHE_DIR, f'enhanced_features{cache_suffix}{model_cache_suffix}.pkl')
    
    # 尝试加载缓存的增强特征
    if os.path.exists(enhanced_features_cache_path):
        print(f"加载缓存的增强特征 (模型: {LLM_MODEL_TYPE}/{LLM_MODEL_NAME}){' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(enhanced_features_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            enhanced_features_zh = cache_data['zh']
            enhanced_features_en = cache_data['en']
        print(f"从缓存加载了 {len(enhanced_features_zh)} 个百度特征和 {len(enhanced_features_en)} 个高德特征")
    else:
        enhanced_features_zh = create_poi_enhanced_features(
            df_zh, text_emb_zh, gat_emb_zh, geohash_encoder, device, llm_enhancer
        )
        enhanced_features_en = create_poi_enhanced_features(
            df_en, text_emb_en, gat_emb_en, geohash_encoder, device, llm_enhancer
        )
        
        # 保存增强特征到缓存
        print(f"保存增强特征到缓存 (模型: {LLM_MODEL_TYPE}/{LLM_MODEL_NAME}){' (DEBUG模式)' if DEBUG_MODE else ''}...")
        with open(enhanced_features_cache_path, 'wb') as f:
            pickle.dump({
                'zh': enhanced_features_zh,
                'en': enhanced_features_en,
                'model_type': LLM_MODEL_TYPE,
                'model_name': LLM_MODEL_NAME,
                'timestamp': datetime.now().isoformat()
            }, f)
        print(f"已缓存 {len(enhanced_features_zh)} 个百度特征和 {len(enhanced_features_en)} 个高德特征")
    
    # --- 使用标注数据训练分类器 ---
    classifier = train_classifier_with_enhanced_features(
        enhanced_features_en, enhanced_features_zh, LABELED_DATA_PATH
    )
    
    print("\n====== 模型训练完成! ======")

if __name__ == '__main__':
    main()