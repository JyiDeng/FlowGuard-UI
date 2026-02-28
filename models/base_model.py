# -*- coding: utf-8 -*-
"""
模型基类
定义所有模型必须实现的接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseModel(ABC):
    """
    模型基类
    所有模型（包括dummy和real）都应该继承此类并实现generate方法
    """
    
    def __init__(self):
        """初始化模型"""
        self.model_name = "BaseModel"
        self.is_loaded = False
    
    @abstractmethod
    def generate(self, conversation: List[Dict[str, str]], plantuml: str = None) -> Tuple[str, str]:
        """
        生成回复
        
        参数:
            conversation: 对话历史，格式为 [{"role": "system/user/assistant", "content": "..."}]
            plantuml: 当前使用的PlantUML流程图代码
        
        返回:
            Tuple[str, str]: (下一状态, 回复内容)
        """
        pass
    
    @abstractmethod
    def load(self):
        """加载模型"""
        pass
    
    def unload(self):
        """卸载模型（可选实现）"""
        self.is_loaded = False
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_loaded
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        return {
            "name": self.model_name,
            "is_loaded": self.is_loaded,
            "type": self.__class__.__name__
        }
