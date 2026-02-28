# -*- coding: utf-8 -*-
"""
Dummy模型实现
用于测试和开发，自动进行状态转移
"""
import re
import random
from typing import List, Dict, Tuple
from .base_model import BaseModel

class DummyModel(BaseModel):
    """
    模拟模型
    根据PlantUML自动进行状态转移，不使用真实的AI模型
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "DummyModel"
        self.is_loaded = True  # Dummy模型总是已加载
        self.current_plantuml = None
        self.state_graph = None
    
    def load(self):
        """Dummy模型不需要加载"""
        self.is_loaded = True
        print("DummyModel: 已就绪（无需加载）", flush=True)
    
    def _parse_plantuml_states(self, plantuml: str) -> List[str]:
        """
        从PlantUML中解析所有状态节点
        
        参数:
            plantuml: PlantUML代码
        
        返回:
            List[str]: 状态列表
        """
        states = ["<start>"]  # 开始状态
        
        lines = plantuml.split('\n')
        for line in lines:
            line = line.strip()
            
            # 匹配活动节点: :活动名称;
            if line.startswith(':') and line.endswith(';'):
                state = line[1:-1].strip()
                if state:
                    states.append(state)
            
            # 匹配决策节点: if (条件?) then (分支)
            elif 'if (' in line and ') then (' in line:
                match = re.search(r'if\s*\((.*?)\)', line)
                if match:
                    condition = match.group(1).strip()
                    if condition:
                        states.append(condition)
            
            # 匹配 repeat while 决策节点
            elif line.startswith('repeat while'):
                match = re.search(r'repeat\s+while\s*\((.*?)\)', line)
                if match:
                    condition = match.group(1).strip()
                    if condition:
                        states.append(condition)
        
        states.append("<end>")  # 结束状态
        return states
    
    def _build_state_graph(self, plantuml: str) -> Dict[str, str]:
        """
        构建状态转移图
        简单的线性转移，从一个状态到下一个状态
        
        参数:
            plantuml: PlantUML代码
        
        返回:
            Dict[str, str]: 状态转移映射 {当前状态: 下一状态}
        """
        states = self._parse_plantuml_states(plantuml)
        state_graph = {}
        
        # 构建简单的线性转移
        for i in range(len(states) - 1):
            state_graph[states[i]] = states[i + 1]
        
        # 最后一个状态指向end
        if states:
            state_graph[states[-1]] = "<end>"
        
        return state_graph
    
    def _get_next_state(self, current_state: str, plantuml: str) -> str:
        """
        获取下一个状态
        
        参数:
            current_state: 当前状态
            plantuml: PlantUML代码
        
        返回:
            str: 下一个状态
        """
        # 如果PlantUML变了，重新构建状态图
        if plantuml != self.current_plantuml:
            self.current_plantuml = plantuml
            self.state_graph = self._build_state_graph(plantuml)
        
        # 获取下一状态
        next_state = self.state_graph.get(current_state, "<end>")
        return next_state
    
    def _generate_response(self, next_state: str) -> str:
        """
        生成回复内容
        
        参数:
            next_state: 下一个状态
        
        返回:
            str: 回复内容
        """
        use_zh = bool(
            re.search(r"[\u4e00-\u9fff]", next_state or "")
            or re.search(r"[\u4e00-\u9fff]", self.current_plantuml or "")
        )
        if next_state == "<end>":
            return random.choice([
                "流程已完成，感谢你的使用。",
                "任务结束，感谢配合。",
                "流程完成，祝你今天顺利。",
                "已经全部完成，谢谢！"
            ]) if use_zh else random.choice([
                "All set. Thanks for using the service.",
                "We're done here. Appreciate your time.",
                "Process complete. Have a great day.",
                "That's everything. Thanks!"
            ])
        elif next_state == "<start>":
            return random.choice([
                "我们开始吧。",
                "好的，开始。",
                "准备好了，开始。",
                "现在开始处理。"
            ]) if use_zh else random.choice([
                "Let's get started.",
                "Okay, let's begin.",
                "Great, starting now.",
                "Ready to kick this off."
            ])
        else:
            # 根据状态生成回复
            if "?" in next_state:
                # 决策节点
                return random.choice([
                    f"请确认：{next_state}",
                    f"请确认这个条件：{next_state}",
                    f"需要你确认一下：{next_state}",
                    f"确认后我们继续：{next_state}"
                ]) if use_zh else random.choice([
                    f"Please confirm: {next_state}",
                    f"Quick check for you: {next_state}",
                    f"Can you confirm this? {next_state}",
                    f"Just need your confirmation: {next_state}"
                ])
            else:
                # 活动节点
                return random.choice([
                    f"正在处理：{next_state}",
                    f"当前步骤：{next_state}",
                    f"下一步：{next_state}",
                    f"好的，继续执行：{next_state}"
                ]) if use_zh else random.choice([
                    f"Proceeding with: {next_state}",
                    f"Working on: {next_state}",
                    f"Next step: {next_state}",
                    f"Got it. Moving ahead with: {next_state}"
                ])
    
    def generate(self, conversation: List[Dict[str, str]], plantuml: str = None) -> Tuple[str, str]:
        """
        生成回复（Dummy模式：自动状态转移）
        
        参数:
            conversation: 对话历史
            plantuml: PlantUML流程图代码
        
        返回:
            Tuple[str, str]: (下一状态, 回复内容)
        """
        if not plantuml:
            return "<end>", "错误：未提供PlantUML流程图"
        
        # 从对话历史中提取当前状态
        current_state = "<start>"
        user_content = ""
        
        for msg in conversation:
            if msg["role"] == "user":
                content = msg["content"]
                user_content = content
                # 尝试从用户消息中提取当前状态
                if "当前状态：" in content:
                    try:
                        current_state = content.split("当前状态：")[1].split("\n")[0].strip()
                    except:
                        pass
                elif "Current state:" in content:
                    try:
                        current_state = content.split("Current state:")[1].split("\n")[0].strip()
                    except:
                        pass
        
        # 获取下一状态
        next_state = self._get_next_state(current_state, plantuml)
        
        # 生成回复
        response = self._generate_response(next_state)
        
        print(f"DummyModel: {current_state} -> {next_state}", flush=True)
        print(f"DummyModel: 回复 = {response}", flush=True)
        
        return next_state, response
