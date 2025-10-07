#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别定义匹配脚本

作者: Dionysus
功能: 为CSV文件根据category列匹配category_definitions.xlsx中的定义，新增def列
"""

import pandas as pd
import os
import sys
from pathlib import Path


def load_category_definitions(excel_path):
    """
    加载类别定义文件
    
    参数:
        excel_path: Excel文件路径
    
    返回:
        dict: 类别到定义的映射字典
    """
    try:
        df = pd.read_excel(excel_path)
        print(f"成功加载类别定义文件: {excel_path}")
        print(f"包含 {len(df)} 个类别定义")
        
        # 创建类别到定义的映射字典
        category_def_map = dict(zip(df['category'], df['definition']))
        
        return category_def_map
    
    except Exception as e:
        print(f"加载类别定义文件失败: {e}")
        return {}


def add_definitions_to_csv(csv_path, category_def_map, output_path=None):
    """
    为CSV文件添加定义列
    
    参数:
        csv_path: CSV文件路径
        category_def_map: 类别到定义的映射字典
        output_path: 输出文件路径，如果为None则覆盖原文件
    
    返回:
        bool: 是否成功处理
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        print(f"\n处理文件: {csv_path}")
        print(f"原始数据形状: {df.shape}")
        
        # 检查是否有category列
        if 'category' not in df.columns:
            print(f"错误: 文件 {csv_path} 中没有找到 'category' 列")
            return False
        
        # 添加def列
        df['def'] = df['category'].map(category_def_map)
        
        # 统计匹配情况
        matched_count = df['def'].notna().sum()
        total_count = len(df)
        unmatched_count = total_count - matched_count
        
        print(f"匹配统计:")
        print(f"  总记录数: {total_count}")
        print(f"  成功匹配: {matched_count}")
        print(f"  未匹配: {unmatched_count}")
        print(f"  匹配率: {matched_count/total_count*100:.2f}%")
        
        # 显示未匹配的类别
        if unmatched_count > 0:
            unmatched_categories = df[df['def'].isna()]['category'].unique()
            print(f"未匹配的类别 ({len(unmatched_categories)} 个):")
            for cat in unmatched_categories[:10]:  # 只显示前10个
                print(f"  - {cat}")
            if len(unmatched_categories) > 10:
                print(f"  ... 还有 {len(unmatched_categories) - 10} 个")
        
        # 保存文件
        if output_path is None:
            output_path = csv_path
        
        df.to_csv(output_path, index=False)
        print(f"已保存到: {output_path}")
        print(f"新数据形状: {df.shape}")
        
        return True
    
    except Exception as e:
        print(f"处理文件 {csv_path} 时出错: {e}")
        return False


def main():
    """
    主函数
    """
    # 设置文件路径
    current_dir = Path(__file__).parent
    excel_path = current_dir / 'category_definitions.xlsx'
    
    csv_files = [
        current_dir / 'bd_english_data.csv',
        current_dir / 'gd_chinese_data.csv'
    ]
    
    print("=" * 60)
    print("类别定义匹配脚本")
    print("=" * 60)
    
    # 检查文件是否存在
    if not excel_path.exists():
        print(f"错误: 类别定义文件不存在: {excel_path}")
        sys.exit(1)
    
    # 加载类别定义
    category_def_map = load_category_definitions(excel_path)
    if not category_def_map:
        print("错误: 无法加载类别定义")
        sys.exit(1)
    
    print(f"可用的类别定义:")
    for i, (cat, definition) in enumerate(list(category_def_map.items())[:5]):
        print(f"  {i+1}. {cat}: {definition[:100]}...")
    if len(category_def_map) > 5:
        print(f"  ... 还有 {len(category_def_map) - 5} 个定义")
    
    # 处理每个CSV文件
    success_count = 0
    for csv_path in csv_files:
        if csv_path.exists():
            if add_definitions_to_csv(csv_path, category_def_map):
                success_count += 1
        else:
            print(f"警告: CSV文件不存在: {csv_path}")
    
    print("\n" + "=" * 60)
    print(f"处理完成! 成功处理 {success_count}/{len(csv_files)} 个文件")
    print("=" * 60)


if __name__ == '__main__':
    main()