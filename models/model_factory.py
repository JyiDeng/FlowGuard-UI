# -*- coding: utf-8 -*-
"""
模型工厂
负责创建和管理模型实例，支持快速切换
"""
from typing import Optional
from .base_model import BaseModel
from .dummy_model import DummyModel
from .real_model import RealModel
import config

# 全局模型实例
_model_instance: Optional[BaseModel] = None

def get_model(force_reload: bool = False) -> BaseModel:
    """
    获取模型实例（单例模式）
    
    参数:
        force_reload: 是否强制重新加载模型
    
    返回:
        BaseModel: 模型实例
    """
    global _model_instance
    
    # 如果已有实例且不强制重载，直接返回
    if _model_instance is not None and not force_reload:
        return _model_instance
    
    # 如果强制重载，先卸载旧模型
    if force_reload and _model_instance is not None:
        print(f"ModelFactory: 卸载旧模型 {_model_instance.__class__.__name__}", flush=True)
        _model_instance.unload()
        _model_instance = None
    
    # 根据配置创建新模型
    if config.is_dummy_model():
        print("ModelFactory: 创建 DummyModel", flush=True)
        _model_instance = DummyModel()
    else:
        print("ModelFactory: 创建 RealModel", flush=True)
        _model_instance = RealModel()
    
    # 加载模型
    _model_instance.load()
    
    return _model_instance

def switch_model(model_type: str) -> BaseModel:
    """
    切换模型类型
    
    参数:
        model_type: 模型类型，"dummy" 或 "real"
    
    返回:
        BaseModel: 新的模型实例
    """
    global _model_instance
    
    # 更新配置
    config.MODEL_TYPE = model_type
    
    print(f"ModelFactory: 切换到 {model_type} 模型", flush=True)
    
    # 卸载旧模型
    if _model_instance is not None:
        _model_instance.unload()
        _model_instance = None
    
    # 创建新模型
    return get_model(force_reload=True)

def get_current_model_info() -> dict:
    """
    获取当前模型信息
    
    返回:
        dict: 模型信息
    """
    if _model_instance is None:
        return {
            "status": "not_loaded",
            "type": config.MODEL_TYPE
        }
    
    info = _model_instance.get_model_info()
    info["config_type"] = config.MODEL_TYPE
    return info

def ensure_model_loaded() -> bool:
    """
    确保模型已加载
    
    返回:
        bool: 模型是否已成功加载
    """
    try:
        model = get_model()
        return model.is_model_loaded()
    except Exception as e:
        print(f"ModelFactory: 确保模型加载失败: {e}", flush=True)
        return False
