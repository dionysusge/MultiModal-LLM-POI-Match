#!/usr/bin/env python3
"""
LLM文本增强器 - 双线路并行架构
作者: Dionysus

基于论文实现的双线路并行架构：
线路1: 原始POI文本 → MiniLM编码 → 原始嵌入特征
线路2: POI信息 → 大模型生成三种文本 → 特征对齐 → 语义融合 → 交叉注意力融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import requests
import json
from typing import Dict, List, Optional, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

class MultiQueryAttention(nn.Module):
    """多查询注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.head_dim)
        self.v_linear = nn.Linear(d_model, self.head_dim)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 计算Q, K, V
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, 1, 1, self.head_dim).expand(-1, self.num_heads, -1, -1)
        V = self.v_linear(value).view(batch_size, 1, 1, self.head_dim).expand(-1, self.num_heads, -1, -1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(attended)

class ParallelAttentionFeedforward(nn.Module):
    """并行注意力前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class DualFeatureAlignment(nn.Module):
    """双特征对齐模块"""
    
    def __init__(self, llm_dim: int, target_dim: int, num_heads: int = 8):
        super().__init__()
        self.target_dim = target_dim
        
        # 线性映射层
        self.W_V = nn.Linear(llm_dim, target_dim)
        self.W_A = nn.Linear(llm_dim, target_dim)
        self.W_S = nn.Linear(llm_dim, target_dim)
        
        # Transformer编码器用于特征对齐
        self.attention_VA = nn.MultiheadAttention(target_dim, num_heads, batch_first=True)
        self.attention_SA = nn.MultiheadAttention(target_dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(target_dim)
        self.norm2 = nn.LayerNorm(target_dim)
        
    def forward(self, E_V: torch.Tensor, E_A: torch.Tensor, E_S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
            E_V: 访问模式特征 [batch_size, llm_dim]
            E_A: 地址特征 [batch_size, llm_dim]
            E_S: 周边环境特征 [batch_size, llm_dim]
        输出:
            E_A_V: 地址-访问模式对齐特征
            E_A_S: 地址-周边环境对齐特征
        """
        # 线性映射到目标维度
        E_V_mapped = self.W_V(E_V).unsqueeze(1)  # [batch_size, 1, target_dim]
        E_A_mapped = self.W_A(E_A).unsqueeze(1)  # [batch_size, 1, target_dim]
        E_S_mapped = self.W_S(E_S).unsqueeze(1)  # [batch_size, 1, target_dim]
        
        # 地址-访问模式对齐
        E_A_V, _ = self.attention_VA(E_A_mapped, E_V_mapped, E_V_mapped)
        E_A_V = self.norm1(E_A_V + E_A_mapped)
        
        # 地址-周边环境对齐
        E_A_S, _ = self.attention_SA(E_A_mapped, E_S_mapped, E_S_mapped)
        E_A_S = self.norm2(E_A_S + E_A_mapped)
        
        return E_A_V.squeeze(1), E_A_S.squeeze(1)

class SemanticFeatureFusion(nn.Module):
    """语义特征融合模块"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 注意力权重计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, E_A_V: torch.Tensor, E_A_S: torch.Tensor) -> torch.Tensor:
        """
        输入:
            E_A_V: 地址-访问模式对齐特征 [batch_size, feature_dim]
            E_A_S: 地址-周边环境对齐特征 [batch_size, feature_dim]
        输出:
            E_LLM: 融合后的语义向量 [batch_size, feature_dim]
        """
        # 计算注意力权重
        concat_features = torch.cat([E_A_V, E_A_S], dim=-1)  # [batch_size, feature_dim * 2]
        attention_weights = self.attention_net(concat_features)  # [batch_size, 2]
        
        # 加权融合
        w_A_V = attention_weights[:, 0:1]  # [batch_size, 1]
        w_A_S = attention_weights[:, 1:2]  # [batch_size, 1]
        
        E_LLM = w_A_V * E_A_V + w_A_S * E_A_S
        
        return E_LLM

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""
    
    def __init__(self, d_model: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # 多层Transformer结构
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mqa': MultiQueryAttention(d_model, num_heads),
                'paf': ParallelAttentionFeedforward(d_model),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
    def forward(self, E_LLM: torch.Tensor, E_POI: torch.Tensor) -> torch.Tensor:
        """
        输入:
            E_LLM: LLM语义向量 [batch_size, d_model]
            E_POI: 原始POI表示向量 [batch_size, d_model]
        输出:
            E_FUSE: 最终融合表示 [batch_size, d_model]
        """
        X = E_POI.unsqueeze(1)  # [batch_size, 1, d_model]
        E_LLM_expanded = E_LLM.unsqueeze(1)  # [batch_size, 1, d_model]
        
        for layer in self.layers:
            # 多查询注意力
            attn_out = layer['mqa'](X, E_LLM_expanded, E_LLM_expanded)
            X = layer['norm1'](X + attn_out)
            
            # 并行注意力前馈网络
            ff_out = layer['paf'](X)
            X = layer['norm2'](X + ff_out)
        
        return X.squeeze(1)  # [batch_size, d_model]

class LLMTextEnhancer:
    """
    LLM文本增强器 - 双线路并行架构
    
    线路1: 原始POI文本 → MiniLM编码 → 原始嵌入特征
    线路2: POI信息 → 大模型生成三种文本 → 特征对齐 → 语义融合 → 交叉注意力融合
    """
    
    def __init__(self, 
                 llm_model_path: Optional[str] = None,
                 text_encoder_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 use_llm: bool = True,
                 device: str = "cuda",
                 target_dim: int = 384,
                 model_type: str = "huggingface",
                 ollama_base_url: str = "http://localhost:11434"):
        """
        初始化LLM文本增强器
        
        Args:
            llm_model_path: 大模型路径或OLLAMA模型名称
            text_encoder_name: 文本编码器名称
            use_llm: 是否使用大模型
            device: 设备
            target_dim: 目标特征维度
            model_type: 模型类型 ("huggingface" 或 "ollama")
            ollama_base_url: OLLAMA服务的基础URL
        """
        self.device = device
        self.use_llm = use_llm and llm_model_path is not None
        self.target_dim = target_dim
        self.model_type = model_type
        self.ollama_base_url = ollama_base_url
        self.model_name = llm_model_path  # 保存模型名称用于缓存
        self.llm_model_path = llm_model_path  # 保存模型路径用于判断模型类型
        
        # 初始化文本编码器（线路1）
        self._init_text_encoder(text_encoder_name)
        
        # 初始化大模型（线路2）
        if self.use_llm:
            if model_type == "ollama":
                self._init_ollama_model(llm_model_path)
            else:
                self._init_llm_model(llm_model_path)
        else:
            print("未指定LLM模型路径或use_llm为False，将使用规则增强")
            self.llm_model = None
            self.llm_tokenizer = None
        
        # 初始化神经网络模块
        if self.use_llm:
            self._init_neural_modules()
    
    def _init_text_encoder(self, text_encoder_name: str):
        """初始化文本编码器"""
        try:
            # 尝试加载本地模型
            # 获取当前项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            local_model_path = os.path.join(project_root, "models", text_encoder_name)
            
            if os.path.exists(local_model_path):
                print(f"加载本地miniLM模型: {local_model_path}")
                self.text_encoder = SentenceTransformer(local_model_path, device=self.device)
                print("本地miniLM模型加载成功")
            else:
                print(f"本地模型不存在，尝试在线加载: {text_encoder_name}")
                self.text_encoder = SentenceTransformer(text_encoder_name, device=self.device)
                print("在线miniLM模型加载成功")
        except Exception as e:
            print(f"文本编码器加载失败: {e}")
            raise
    
    def _init_ollama_model(self, model_name: str):
        """初始化OLLAMA模型"""
        try:
            print(f"初始化OLLAMA模型: {model_name}")
            
            # 检查OLLAMA服务是否可用
            try:
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    raise Exception(f"OLLAMA服务不可用，状态码: {response.status_code}")
                
                # 检查模型是否存在
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if model_name not in model_names:
                    print(f"模型 {model_name} 不存在，可用模型: {model_names}")
                    print(f"尝试拉取模型 {model_name}...")
                    
                    # 尝试拉取模型
                    pull_response = requests.post(
                        f"{self.ollama_base_url}/api/pull",
                        json={"name": model_name},
                        timeout=300
                    )
                    
                    if pull_response.status_code != 200:
                        raise Exception(f"拉取模型失败: {pull_response.text}")
                    
                    print(f"模型 {model_name} 拉取成功")
                
                self.llm_model = model_name  # 对于OLLAMA，我们只需要存储模型名称
                self.llm_tokenizer = None  # OLLAMA不需要tokenizer
                
                # 测试嵌入功能并获取实际维度
                embed_response = requests.post(
                    f"{self.ollama_base_url}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": "测试文本"
                    },
                    timeout=30
                )
                
                if embed_response.status_code == 200:
                    result = embed_response.json()
                    embeddings = result.get('embedding', [])
                    if embeddings:
                        self.ollama_embedding_dim = len(embeddings)
                        print(f"OLLAMA模型嵌入维度: {self.ollama_embedding_dim}")
                    else:
                        self.ollama_embedding_dim = 768  # 默认维度
                        print("无法获取OLLAMA嵌入维度，使用默认值768")
                else:
                    self.ollama_embedding_dim = 768  # 默认维度
                    print("嵌入测试失败，使用默认维度768")
                
                print(f"OLLAMA模型 {model_name} 初始化成功")
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"无法连接到OLLAMA服务 ({self.ollama_base_url}): {e}")
                
        except Exception as e:
            print(f"OLLAMA模型初始化失败: {e}")
            raise
    
    def _init_llm_model(self, llm_model_path: str):
        """初始化HuggingFace大模型"""
        try:
            print(f"加载大模型: {llm_model_path}")
            
            # 检查模型文件是否存在
            if not os.path.exists(llm_model_path):
                raise FileNotFoundError(f"模型路径不存在: {llm_model_path}")
            
            # 检查是否有PyTorch模型文件（只检查权重文件，不包括config.json）
            pytorch_files = ['pytorch_model.bin', 'model.safetensors']
            has_pytorch_model = any(os.path.exists(os.path.join(llm_model_path, f)) for f in pytorch_files)
            
            # 检查是否有TensorFlow模型文件
            tf_files = ['tf_model.h5']
            has_tf_model = any(os.path.exists(os.path.join(llm_model_path, f)) for f in tf_files)
            
            if has_pytorch_model:
                print("使用本地PyTorch模型")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
                
                # 检查是否为Qwen3模型，使用特殊的加载参数
                if "Qwen3" in llm_model_path or "qwen3" in llm_model_path.lower():
                    print("检测到Qwen3模型，使用优化的加载参数...")
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        torch_dtype=torch.float16,  # 使用半精度以节省显存
                        device_map="auto",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                else:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        torch_dtype=torch.float16,  # 使用半精度以节省显存
                        low_cpu_mem_usage=True,
                        device_map="auto"
                    )
            elif has_tf_model:
                print("使用本地TensorFlow模型，转换为PyTorch")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
                # 从TensorFlow模型加载并转换为PyTorch
                try:
                    # 检查是否为Qwen3模型
                    if "Qwen3" in llm_model_path or "qwen3" in llm_model_path.lower():
                        print("检测到Qwen3模型，使用优化的加载参数...")
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            llm_model_path,
                            from_tf=True,
                            torch_dtype=torch.float16,  # 使用半精度以节省显存
                            device_map="auto",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                    else:
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            llm_model_path,
                            from_tf=True,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                        )
                except Exception as tf_error:
                    print(f"TensorFlow转换失败: {tf_error}")
                    print("尝试使用在线模型作为备选...")
                    # 尝试从HuggingFace Hub加载相同的模型
                    model_name = os.path.basename(llm_model_path)
                    if model_name == "DialoGPT-small":
                        online_model_name = "microsoft/DialoGPT-small"
                        self.llm_model = AutoModelForCausalLM.from_pretrained(
                            online_model_name,
                            torch_dtype=torch.float32,
                            low_cpu_mem_usage=True
                        )
                        print(f"成功加载在线模型: {online_model_name}")
                    else:
                        raise tf_error
            else:
                print("未找到模型权重文件，尝试在线加载")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
                
                # 检查是否为Qwen3模型
                if "Qwen3" in llm_model_path or "qwen3" in llm_model_path.lower():
                    print("检测到Qwen3模型，使用优化的加载参数...")
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                else:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        llm_model_path,
                        torch_dtype=torch.float32
                    )
            
            # 设置pad_token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # 移动模型到指定设备（Qwen3模型跳过，因为使用了device_map="auto"）
            if self.device != "cpu" and not ("Qwen3" in llm_model_path or "qwen3" in llm_model_path.lower()):
                try:
                    self.llm_model = self.llm_model.to(self.device)
                except NotImplementedError as e:
                    if "meta tensor" in str(e):
                        # 处理meta tensor问题，使用to_empty()方法
                        print("检测到meta tensor问题，使用to_empty()方法...")
                        self.llm_model = self.llm_model.to_empty(device=self.device)
                    else:
                        raise e
            elif "Qwen3" in llm_model_path or "qwen3" in llm_model_path.lower():
                print("Qwen3模型使用device_map='auto'，跳过手动设备移动")
            
            self.llm_model.eval()
            print(f"大模型加载成功，设备: {self.device}")
            
        except Exception as e:
            print(f"大模型加载失败: {e}")
            raise

    def _init_neural_modules(self):
        """初始化神经网络模块"""
        # 根据模型类型确定LLM隐藏维度
        if self.model_type == "ollama":
            llm_hidden_dim = getattr(self, 'ollama_embedding_dim', 768)
        else:
            # 根据模型类型确定隐藏维度
            if "Qwen3" in str(self.llm_model_path) or "qwen3" in str(self.llm_model_path).lower():
                # Qwen3-0.6B的隐藏状态维度为1024
                llm_hidden_dim = 1024
            else:
                # DialoGPT-small的隐藏状态维度为768
                llm_hidden_dim = 768
        
        print(f"LLM隐藏维度: {llm_hidden_dim}")
        
        # 初始化各个模块
        self.dual_alignment = DualFeatureAlignment(
            llm_dim=llm_hidden_dim,
            target_dim=self.target_dim
        ).to(self.device)
        
        self.semantic_fusion = SemanticFeatureFusion(
            feature_dim=self.target_dim
        ).to(self.device)
        
        self.cross_attention = CrossAttentionFusion(
            d_model=self.target_dim,
            num_heads=8,
            num_layers=2
        ).to(self.device)
    
    def _generate_prompts(self, poi_info: Dict[str, str]) -> Dict[str, str]:
        """
        生成三个维度的提示词
        
        Args:
            poi_info: POI信息字典，包含name、address、category
            
        Returns:
            Dict[str, str]: 包含三个维度提示词的字典
        """
        name = poi_info.get('name', '')
        address = poi_info.get('address', '')
        category = poi_info.get('category', '')
        
        # 名称维度提示（Venue）- 2-3句话版
        prompt_V = f"""名称：{name}
请基于这个名称特征，描述其品牌特色、服务类型。用1-2句话说明。"""

        # 地址维度提示（Address）- 2-3句话版
        prompt_A = f"""地址：{address}
请基于这个地址信息，描述其地理位置、邮政编码和区域特点。用1-2句话说明。"""

        # 类别维度提示（Category）- 2-3句话版
        prompt_S = f"""类别：{category}
请基于这个类别特征，描述其行业属性、服务功能和目标客群。用1-2句话说明。"""
        
        return {
            'V': prompt_V,
            'A': prompt_A,
            'S': prompt_S
        }
    
    def _generate_with_llm(self, prompt: str) -> str:
        """使用大模型生成文本"""
        if not self.use_llm or self.llm_model is None:
            return ""
        
        try:
            if self.model_type == "ollama":
                return self._generate_with_ollama(prompt)
            else:
                return self._generate_with_huggingface(prompt)
        except Exception as e:
            print(f"LLM生成失败: {e}")
            return ""
    
    def _batch_generate_with_llm(self, prompts: List[str], batch_size: int = 256) -> List[str]:
        """批量根据模型类型选择生成方法"""
        if self.model_type == "ollama":
            # Ollama暂时不支持批处理，回退到单个生成
            return [self._generate_with_ollama(prompt) for prompt in prompts]
        elif "qwen" in str(self.llm_model_path).lower():
            return self._batch_generate_with_qwen3(prompts, batch_size)
        else:
            # 使用真正的并行HuggingFace批处理
            return self._batch_generate_with_huggingface(prompts, batch_size)
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """使用OLLAMA生成文本"""
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 100,
                        "stop": ["\n\n"]
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                return generated_text
            else:
                print(f"OLLAMA生成失败，状态码: {response.status_code}")
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"OLLAMA请求失败: {e}")
            return ""
    
    def _generate_with_huggingface(self, prompt: str) -> str:
        """使用HuggingFace模型生成文本"""
        try:
            # 检查是否为Qwen3模型
            if "Qwen3" in str(self.llm_model_path) or "qwen3" in str(self.llm_model_path).lower():
                return self._generate_with_qwen3(prompt)
            else:
                return self._generate_with_standard_model(prompt)
        except Exception as e:
            print(f"HuggingFace生成失败: {e}")
            return ""
    
    def _generate_with_qwen3(self, prompt: str) -> str:
        """使用Qwen3模型生成文本（直接输出模式）"""
        try:
            # 准备消息格式
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 应用chat template（关闭thinking模式）
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # 关闭thinking模式
            )
            
            # 编码输入
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
            
            # 生成文本
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=60,  # 减少到60，适应固定格式的简洁输出
                    do_sample=True,
                    temperature=0.2,  # 进一步降低temperature，确保严格遵循词汇约束
                    top_p=0.5,  # 降低top_p，减少词汇变化
                    top_k=20,  # 进一步限制候选词数量
                    repetition_penalty=1.0,  # 取消重复惩罚，因为格式固定
                    length_penalty=0.8,  # 鼓励简洁输出
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,  # 确保模型能正确结束生成
                    early_stopping=True  # 启用早停，让模型在合适的地方停止生成
                )
            
            # 提取新生成的token并解码
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            generated_text = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Qwen3生成失败: {e}")
            return ""

    def _batch_generate_with_qwen3(self, prompts: List[str], batch_size: int = 256) -> List[str]:
        """批量使用Qwen3模型生成文本"""
        results = []
        
        # 添加进度条
        with tqdm(total=len(prompts), desc="LLM文本生成", unit="prompt", leave=False) as pbar:
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                try:
                    # 准备批量消息格式
                    batch_texts = []
                    for prompt in batch_prompts:
                        messages = [{"role": "user", "content": prompt}]
                        text = self.llm_tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False
                        )
                        batch_texts.append(text)
                    
                    # 批量编码输入
                    model_inputs = self.llm_tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=512
                    ).to(self.llm_model.device)
                    
                    # 批量生成文本（使用混合精度）
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            generated_ids = self.llm_model.generate(
                                **model_inputs,
                                max_new_tokens=60,
                                do_sample=True,
                                temperature=0.2,
                                top_p=0.5,
                                top_k=20,
                                repetition_penalty=1.0,
                                length_penalty=0.8,
                                pad_token_id=self.llm_tokenizer.eos_token_id,
                                eos_token_id=self.llm_tokenizer.eos_token_id,
                                early_stopping=True
                            )
                    
                    # 批量解码结果
                    batch_results = []
                    for j, generated_id in enumerate(generated_ids):
                        input_length = len(model_inputs.input_ids[j])
                        output_ids = generated_id[input_length:].tolist()
                        generated_text = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                        batch_results.append(generated_text)
                    
                    results.extend(batch_results)
                    
                except Exception as e:
                    print(f"批量生成失败: {e}")
                    # 回退到单个生成
                    for prompt in batch_prompts:
                        try:
                            result = self._generate_with_qwen3(prompt)
                            results.append(result)
                        except:
                            results.append("")
                    
                    # 更新进度条
                    pbar.update(len(batch_prompts))
        
        return results
    
    def _batch_generate_with_huggingface(self, prompts: List[str], batch_size: int = 256) -> List[str]:
        """
        真正的并行HuggingFace批处理生成
        
        Args:
            prompts: 提示词列表
            batch_size: 批处理大小
            
        Returns:
            生成的文本列表
        """
        results = []
        
        try:
            # 添加进度条
            with tqdm(total=len(prompts), desc="LLM文本生成", unit="prompt", leave=False) as pbar:
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]
                    
                    try:
                        # 批量编码
                        inputs = self.llm_tokenizer(
                            batch_prompts,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(self.device)
                        
                        # 批量生成（使用混合精度）
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                                outputs = self.llm_model.generate(
                                    **inputs,
                                    max_new_tokens=256,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                    pad_token_id=self.llm_tokenizer.eos_token_id,
                                    num_return_sequences=1
                                )
                        
                        # 批量解码
                        batch_results = []
                        for j, output in enumerate(outputs):
                            # 移除输入部分，只保留生成的文本
                            input_length = inputs['input_ids'][j].shape[0]
                            generated_tokens = output[input_length:]
                            generated_text = self.llm_tokenizer.decode(
                                generated_tokens, 
                                skip_special_tokens=True
                            ).strip()
                            batch_results.append(generated_text)
                        
                        results.extend(batch_results)
                        
                    except Exception as e:
                        print(f"批处理生成失败: {e}")
                        # 回退到单个生成
                        for prompt in batch_prompts:
                            try:
                                result = self._generate_with_standard_model(prompt)
                                results.append(result)
                            except Exception as single_e:
                                print(f"单个生成也失败: {single_e}")
                                results.append("")
                    
                    # 更新进度条
                    pbar.update(len(batch_prompts))
        
        except Exception as e:
            print(f"批处理初始化失败: {e}")
            # 完全回退到单个生成
            results = [self._generate_with_standard_model(prompt) for prompt in prompts]
        
        return results

    def _generate_with_standard_model(self, prompt: str) -> str:
        """使用标准HuggingFace模型生成文本"""
        inputs = self.llm_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs,
                max_length=inputs.shape[1] + 60,  # 限制生成长度为50-100字左右
                do_sample=True,
                temperature=0.6,  # 降低temperature提高生成速度
                top_p=0.9,  # 添加top_p采样提高质量
                pad_token_id=self.llm_tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始prompt，只保留生成的部分
        generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def _extract_llm_features(self, text: str) -> torch.Tensor:
        """从LLM中提取特征向量"""
        if not self.use_llm or self.llm_model is None:
            # 返回随机向量作为占位符
            fallback_dim = getattr(self, 'ollama_embedding_dim', 768) if self.model_type == "ollama" else 768
            return torch.randn(fallback_dim, device=self.device)
        
        try:
            if self.model_type == "ollama":
                return self._extract_ollama_features(text)
            else:
                return self._extract_huggingface_features(text)
        except Exception as e:
            print(f"LLM特征提取失败: {e}")
            # 返回随机向量作为回退
            fallback_dim = getattr(self, 'ollama_embedding_dim', 768) if self.model_type == "ollama" else 768
            return torch.randn(fallback_dim, device=self.device)
    
    def _extract_ollama_features(self, text: str) -> torch.Tensor:
        """从OLLAMA模型中提取特征向量"""
        try:
            # 对于OLLAMA模型，我们使用embeddings API
            response = requests.post(
                f"{self.ollama_base_url}/api/embeddings",
                json={
                    "model": self.llm_model,
                    "prompt": text
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = result.get('embedding', [])
                if embeddings:
                    # 转换为torch tensor
                    features = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
                    return features
                else:
                    print("OLLAMA返回空的嵌入向量")
                    fallback_dim = getattr(self, 'ollama_embedding_dim', 768)
                    return torch.randn(fallback_dim, device=self.device)
            else:
                print(f"OLLAMA嵌入提取失败，状态码: {response.status_code}")
                fallback_dim = getattr(self, 'ollama_embedding_dim', 768)
                return torch.randn(fallback_dim, device=self.device)
                
        except requests.exceptions.RequestException as e:
            print(f"OLLAMA嵌入请求失败: {e}")
            fallback_dim = getattr(self, 'ollama_embedding_dim', 768)
            return torch.randn(fallback_dim, device=self.device)
    
    def _extract_huggingface_features(self, text: str) -> torch.Tensor:
        """从HuggingFace模型中提取特征向量"""
        try:
            # 检查是否为Qwen3模型
            if "Qwen3" in str(self.llm_model_path) or "qwen3" in str(self.llm_model_path).lower():
                return self._extract_qwen3_features(text)
            else:
                return self._extract_standard_features(text)
        except Exception as e:
            print(f"特征提取失败: {e}")
            # 返回随机向量作为回退，根据模型类型确定维度
            fallback_dim = 1024 if ("Qwen3" in str(self.llm_model_path) or "qwen3" in str(self.llm_model_path).lower()) else 768
            return torch.randn(fallback_dim, device=self.device)
    
    def _batch_extract_llm_features(self, texts: List[str]) -> List[torch.Tensor]:
        """批量提取LLM特征"""
        try:
            if self.model_type == "ollama":
                # OLLAMA模型需要单独处理
                return [self._extract_llm_features(text) for text in texts]
            elif "qwen" in str(self.llm_model_path).lower():
                # Qwen模型批量特征提取
                return self._batch_extract_qwen3_features(texts)
            else:
                # 其他HuggingFace模型批量特征提取
                return self._batch_extract_huggingface_features(texts)
            
        except Exception as e:
            print(f"批量特征提取失败: {e}")
            # 回退到单个提取
            return [self._extract_llm_features(text) for text in texts]

    def _batch_extract_qwen3_features(self, texts: List[str]) -> List[torch.Tensor]:
        """批量提取Qwen3特征"""
        try:
            features_list = []
            batch_size = 128  # Qwen3批处理大小，为RTX 5090优化
            
            # 添加进度条
            with tqdm(total=len(texts), desc="特征提取", unit="text", leave=False) as pbar:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    
                    # 批量编码
                    inputs = self.llm_tokenizer(
                        batch_texts, 
                        return_tensors='pt', 
                        max_length=512, 
                        truncation=True, 
                        padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.llm_model(**inputs, output_hidden_states=True)
                        # 提取最后一层隐藏状态的平均值
                        last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
                        # 对序列长度维度求平均
                        batch_features = last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
                        
                        # 分离每个样本的特征并立即移到CPU
                        for j in range(batch_features.size(0)):
                            features = batch_features[j].clone().detach().float()
                            features_list.append(features)
                        
                        # 清理GPU内存
                        del outputs, last_hidden_state, batch_features
                        del inputs
                        
                        # 更新进度条
                        pbar.update(len(batch_texts))
                    
                    # 每个批次后清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return features_list
            
        except Exception as e:
            print(f"Qwen3批量特征提取失败: {e}")
            # 回退到单个提取
            return [self._extract_qwen3_features(text) for text in texts]

    def _batch_extract_huggingface_features(self, texts: List[str]) -> List[torch.Tensor]:
        """批量提取HuggingFace特征"""
        try:
            features_list = []
            batch_size = 128  # 为RTX 5090优化
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 批量编码
                inputs = self.llm_tokenizer(
                    batch_texts, 
                    return_tensors='pt', 
                    max_length=512, 
                    truncation=True, 
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.llm_model(**inputs, output_hidden_states=True)
                    # 提取最后一层隐藏状态的平均值
                    last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
                    # 对序列长度维度求平均
                    batch_features = last_hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
                    
                    # 分离每个样本的特征并立即移到CPU
                    for j in range(batch_features.size(0)):
                        features = batch_features[j].clone().detach().float()
                        features_list.append(features)
                    
                    # 清理GPU内存
                    del outputs, last_hidden_state, batch_features
                    del inputs
                    
                    # 每个批次后清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return features_list
            
        except Exception as e:
            print(f"HuggingFace批量特征提取失败: {e}")
            # 回退到单个提取
            return [self._extract_huggingface_features(text) for text in texts]
    
    def _extract_qwen3_features(self, text: str) -> torch.Tensor:
        """从Qwen3模型中提取特征向量"""
        try:
            # 对于Qwen3，我们使用简单的编码方式获取特征
            inputs = self.llm_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model(inputs, output_hidden_states=True)
                # 提取最后一层隐藏状态的平均值
                last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
                features = last_hidden_state.mean(dim=1).squeeze(0)  # [hidden_dim]
                # 克隆tensor以避免推理模式下的反向传播问题，并确保为float32类型
                features = features.clone().detach().float()
            
            return features
            
        except Exception as e:
            print(f"Qwen3特征提取失败: {e}")
            # 返回随机向量作为回退，使用Qwen3的隐藏维度
            return torch.randn(1024, device=self.device)
    
    def _extract_standard_features(self, text: str) -> torch.Tensor:
        """从标准HuggingFace模型中提取特征向量"""
        inputs = self.llm_tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model(inputs, output_hidden_states=True)
            # 提取最后一层隐藏状态的平均值
            last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            features = last_hidden_state.mean(dim=1).squeeze(0)  # [hidden_dim]
            # 克隆tensor以避免推理模式下的反向传播问题，并确保为float32类型
            features = features.clone().detach().float()
        
        return features

    def batch_enhance_poi_features(self, poi_infos: List[Dict[str, str]], batch_size: int = 256) -> List[np.ndarray]:
        """
        批量增强POI特征
        
        Args:
            poi_infos: POI信息字典列表
            batch_size: 批处理大小
            
        Returns:
            增强后的特征向量列表
        """
        if not self.use_llm:
            # 如果不使用LLM，批量处理原始特征
            original_texts = [f"{poi['name']} | {poi['category']} | {poi['address']}" for poi in poi_infos]
            embeddings = self.text_encoder.encode(
                original_texts, 
                convert_to_tensor=True, 
                device=self.device,
                batch_size=batch_size
            )
            return [emb.cpu().numpy() for emb in embeddings]
        
        enhanced_features = []
        
        # 计算总批次数
        total_batches = (len(poi_infos) + batch_size - 1) // batch_size
        
        # 分批处理，添加进度条
        with tqdm(total=len(poi_infos), desc="生成LLM增强特征", unit="POI") as pbar:
            for i in range(0, len(poi_infos), batch_size):
                batch_poi_infos = poi_infos[i:i + batch_size]
                
                try:
                    # 1. 批量生成原始文本特征
                    original_texts = [f"{poi['name']} | {poi['category']} | {poi['address']}" for poi in batch_poi_infos]
                    E_POI_batch = self.text_encoder.encode(
                        original_texts, 
                        convert_to_tensor=True, 
                        device=self.device,
                        batch_size=len(batch_poi_infos)
                    )
                    
                    # 2. 批量生成所有prompts
                    all_prompts_V = []
                    all_prompts_A = []
                    all_prompts_S = []
                    
                    for poi_info in batch_poi_infos:
                        prompts = self._generate_prompts(poi_info)
                        all_prompts_V.append(prompts['V'])
                        all_prompts_A.append(prompts['A'])
                        all_prompts_S.append(prompts['S'])
                    
                    # 3. 批量LLM生成
                    llm_batch_size = min(batch_size // 4, len(batch_poi_infos))  # 使用更大的LLM批处理大小
                    texts_V = self._batch_generate_with_llm(all_prompts_V, llm_batch_size)
                    texts_A = self._batch_generate_with_llm(all_prompts_A, llm_batch_size)
                    texts_S = self._batch_generate_with_llm(all_prompts_S, llm_batch_size)
                    
                    # 4. 批量特征提取
                    E_V_batch = self._batch_extract_llm_features(texts_V)
                    E_A_batch = self._batch_extract_llm_features(texts_A)
                    E_S_batch = self._batch_extract_llm_features(texts_S)
                    
                    # 5. 逐个处理神经网络部分（因为需要单独的前向传播）
                    for j in range(len(batch_poi_infos)):
                        try:
                            E_POI = E_POI_batch[j].clone().detach().float()
                            E_V = E_V_batch[j].clone().detach().requires_grad_(True).float()
                            E_A = E_A_batch[j].clone().detach().requires_grad_(True).float()
                            E_S = E_S_batch[j].clone().detach().requires_grad_(True).float()
                            
                            # 双特征对齐
                            E_A_V, E_A_S = self.dual_alignment(
                                E_V.unsqueeze(0), E_A.unsqueeze(0), E_S.unsqueeze(0)
                            )
                            
                            # 语义特征融合
                            E_LLM = self.semantic_fusion(E_A_V, E_A_S)
                            
                            # 交叉注意力融合
                            E_FUSE = self.cross_attention(E_LLM, E_POI.unsqueeze(0))
                            
                            # 立即移到CPU并清理GPU内存
                            enhanced_features.append(E_FUSE.squeeze(0).detach().cpu().numpy())
                            
                            # 清理中间变量的GPU内存
                            del E_POI, E_V, E_A, E_S, E_A_V, E_A_S, E_LLM, E_FUSE
                            
                        except Exception as e:
                            print(f"处理第{i+j}个POI的神经网络部分失败: {e}")
                            enhanced_features.append(E_POI_batch[j].cpu().numpy())
                    
                    # 清理批处理的GPU内存
                    del E_POI_batch, E_V_batch, E_A_batch, E_S_batch
                    
                    # 强制清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"批量处理失败，回退到单个处理: {e}")
                    # 回退到单个处理
                    for poi_info in batch_poi_infos:
                        try:
                            feature = self.enhance_single_poi(poi_info)
                            enhanced_features.append(feature)
                        except:
                            # 最后的回退
                            original_text = f"{poi_info['name']} | {poi_info['category']} | {poi_info['address']}"
                            fallback_feature = self.text_encoder.encode([original_text], convert_to_tensor=True, device=self.device)[0]
                            enhanced_features.append(fallback_feature.cpu().numpy())
                    
                # 更新进度条
                pbar.update(len(batch_poi_infos))
        
        return enhanced_features

    def enhance_single_poi(self, poi_info: Dict[str, str]) -> np.ndarray:
        """
        增强单个POI的特征
        
        Args:
            poi_info: POI信息字典
            
        Returns:
            增强后的特征向量
        """
        # 线路1: 原始文本编码
        original_text = f"{poi_info['name']} | {poi_info['category']} | {poi_info['address']}"
        E_POI = self.text_encoder.encode([original_text], convert_to_tensor=True, device=self.device)[0]
        # 克隆tensor以避免推理模式下的问题
        E_POI = E_POI.clone().detach()
        
        if not self.use_llm:
            # 如果不使用LLM，直接返回原始特征
            return E_POI.cpu().numpy()
        
        # 线路2: LLM增强路径
        try:
            # 1. 生成三种提示
            prompts = self._generate_prompts(poi_info)
            
            # 2. 使用LLM生成文本并提取特征
            text_V = self._generate_with_llm(prompts['V'])
            text_A = self._generate_with_llm(prompts['A'])
            text_S = self._generate_with_llm(prompts['S'])
            
            E_V = self._extract_llm_features(text_V)
            E_A = self._extract_llm_features(text_A)
            E_S = self._extract_llm_features(text_S)
            
            # 确保所有tensor都是可训练的
            E_V = E_V.clone().detach().requires_grad_(True)
            E_A = E_A.clone().detach().requires_grad_(True)
            E_S = E_S.clone().detach().requires_grad_(True)
            
            # 3. 双特征对齐
            # 确保数据类型一致性（转换为float32）
            E_V = E_V.float()
            E_A = E_A.float()
            E_S = E_S.float()
            
            E_A_V, E_A_S = self.dual_alignment(
                E_V.unsqueeze(0), E_A.unsqueeze(0), E_S.unsqueeze(0)
            )
            
            # 4. 语义特征融合
            E_LLM = self.semantic_fusion(E_A_V, E_A_S)
            
            # 5. 交叉注意力融合
            # 确保POI特征也是float32类型
            E_POI = E_POI.float()
            E_FUSE = self.cross_attention(E_LLM, E_POI.unsqueeze(0))
            
            return E_FUSE.squeeze(0).detach().cpu().numpy()
            
        except Exception as e:
            print(f"LLM增强失败，使用原始特征: {e}")
            return E_POI.cpu().numpy()
    
    def generate_enhanced_text(self, poi_data: dict, dimension: str = 'V') -> str:
        """
        生成增强文本
        
        Args:
            poi_data: POI数据字典，包含name, category, address
            dimension: 生成维度 ('V': 名称, 'A': 地址, 'S': 类别)
            
        Returns:
            str: 生成的增强文本
        """
        try:
            # 生成简单描述
            prompts = self._generate_prompts(poi_data)
            prompt = prompts.get(dimension, "")
            
            if self.use_llm:
                if self.model_type == "huggingface":
                    return self._generate_with_standard_model(prompt)
                elif self.model_type == "qwen":
                    return self._generate_with_qwen3(prompt)
                else:
                    return self._generate_with_llm(prompt)
            else:
                # 使用基础模板
                return self._generate_basic_template(poi_data, dimension)
                
        except Exception as e:
            print(f"生成增强文本失败: {e}")
            return ""
    


    def batch_generate_enhanced_features(self, poi_texts: List[str], batch_size: int = 512) -> List[np.ndarray]:
        """
        批量生成增强特征
        
        Args:
            poi_texts: POI文本列表，格式为 "name | category | address"
            batch_size: 批处理大小
            
        Returns:
            增强后的特征向量列表
        """
        # 解析所有POI文本
        poi_infos = []
        for poi_text in poi_texts:
            parts = poi_text.split(' | ')
            if len(parts) >= 3:
                poi_info = {
                    'name': parts[0].strip(),
                    'category': parts[1].strip(),
                    'address': parts[2].strip()
                }
            else:
                # 处理格式不正确的情况
                poi_info = {
                    'name': poi_text.strip(),
                    'category': '未知',
                    'address': '未知'
                }
            poi_infos.append(poi_info)
        
        # 使用批量增强方法
        enhanced_features = self.batch_enhance_poi_features(poi_infos, batch_size)
        
        print(f"批量处理完成，共处理 {len(enhanced_features)} 个POI")
        return enhanced_features