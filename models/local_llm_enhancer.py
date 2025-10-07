"""
本地LLM增强器
作者: Dionysus
"""

import numpy as np


# 本地LLM增强器
class LocalLLMEnhancer:
    """本地LLM增强器，使用规则和简单模型"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 预定义的增强模板
        self.enhancement_templates = {
            'name': [
                "{name}({category})",
                "{category}类{name}",
                "{name}·{category}"
            ],
            'address': [
                "{name}位于{address}",
                "{address}的{name}",
                "{name}@{address}"
            ],
            'category': [
                "{category}类型的{name}",
                "{name}-{category}服务",
                "{category}:{name}"
            ]
        }
    
    def generate_enhanced_text(self, prompt, poi_text):
        """生成增强文本"""
        # 解析POI文本
        parts = poi_text.split(" | ")
        if len(parts) >= 3:
            name, category, address = parts[0], parts[1], parts[2]
        else:
            name, category, address = poi_text, "未知", "未知"
        
        # 根据提示词选择模板
        if "名称" in prompt or "name" in prompt.lower():
            template_type = 'name'
        elif "地址" in prompt or "address" in prompt.lower():
            template_type = 'address'
        elif "类别" in prompt or "category" in prompt.lower():
            template_type = 'category'
        else:
            template_type = 'name'  # 默认
        
        # 选择模板并生成
        templates = self.enhancement_templates[template_type]
        template = templates[hash(poi_text) % len(templates)]  # 确定性选择
        
        try:
            enhanced = template.format(name=name, category=category, address=address)
        except:
            enhanced = f"{name} - {category} - {address}"
        
        return enhanced
    
    def batch_generate_enhanced_features(self, poi_texts, prompts):
        """批量生成增强特征"""
        enhanced_features = []
        
        for poi_text in poi_texts:
            # 简单的特征：使用文本长度和字符统计
            feature = np.array([
                len(poi_text),  # 文本长度
                poi_text.count('|'),  # 分隔符数量
                len(poi_text.split()),  # 词数
                hash(poi_text) % 1000 / 1000.0,  # 哈希特征
            ] * 96)  # 重复到384维
            
            enhanced_features.append(feature[:384])  # 确保维度一致
        
        return enhanced_features
