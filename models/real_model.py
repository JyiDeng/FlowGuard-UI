# -*- coding: utf-8 -*-
"""
真实模型实现
使用实际的语言模型进行状态转移和对话生成
"""
import re
import traceback
from typing import List, Dict, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from threading import Thread
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: torch未安装，真实模型不可用", flush=True)

from .base_model import BaseModel
import config

class RealModel(BaseModel):
    """
    真实模型
    使用Transformer模型进行推理
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "RealModel"
        self.tokenizer = None
        self.model = None
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch未安装，无法使用真实模型。请安装: pip install torch transformers")
        
        self.device = self._resolve_device()
    
    def _resolve_device(self):
        """解析设备配置"""
        if config.DEVICE:
            return config.DEVICE
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"
    
    def load(self):
        """加载真实模型"""
        if self.is_loaded:
            print("RealModel: 模型已加载，跳过", flush=True)
            return
        
        print(f"RealModel: 正在从 {config.MODEL_PATH} 加载模型到 {self.device}...", flush=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_PATH, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_PATH,
                trust_remote_code=True
            ).to(self.device)
            
            self.is_loaded = True
            print("RealModel: 模型加载成功", flush=True)
            
        except Exception as e:
            print(f"RealModel: 模型加载失败: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            self.is_loaded = False
            raise RuntimeError(f"模型加载失败: {e}")
    
    def unload(self):
        """卸载模型，释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        print("RealModel: 模型已卸载", flush=True)
    
    def _parse_response(self, response: str, use_english: bool = False) -> Tuple[str, str]:
        """
        解析模型响应，提取下一状态和回复内容
        
        参数:
            response: 模型生成的原始响应
            use_english: 是否使用英文格式
        
返回:
            Tuple[str, str]: (下一状态, 回复内容)
        """
        try:
            response = (response or "").strip()

            next_state = ""
            prompt = ""

            def _clean_state_text(value: str) -> str:
                value = (value or "").strip()
                if not value:
                    return ""
                # 状态名必须是单个节点标签；模型若附带解释，只取首个非空行。
                for line in value.splitlines():
                    cleaned = line.strip().strip("`").strip("\"'").strip()
                    cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", cleaned).strip()
                    cleaned = re.sub(r"\s*;\s*$", "", cleaned).strip()
                    if cleaned:
                        return cleaned
                return ""

            def _extract_field(text: str, labels: List[str], stop_labels: List[str]) -> str:
                # 兼容中英文冒号、空格，以及字段值换行的情况
                label_pattern = "|".join(re.escape(label) for label in labels)
                stop_pattern = "|".join(re.escape(label) for label in stop_labels)
                pattern = rf"(?:{label_pattern})\s*[:：]\s*(.*?)(?=\n\s*(?:{stop_pattern})\s*[:：]|\Z)"
                match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
                if not match:
                    return ""
                return match.group(1).strip()

            def _extract_by_mode(prefer_english: bool) -> Tuple[str, str]:
                if prefer_english:
                    ns = _extract_field(
                        response,
                        labels=["Next state", "Next State"],
                        stop_labels=["Prompt", "User input", "User message", "Reply", "Response"]
                    )
                    pmpt = _extract_field(
                        response,
                        labels=["Prompt", "User input", "User message", "Reply", "Response"],
                        stop_labels=["Next state", "Next State"]
                    )
                else:
                    ns = _extract_field(
                        response,
                        labels=["下一状态", "下一步状态"],
                        stop_labels=["提示", "用户输入", "回复", "输出"]
                    )
                    pmpt = _extract_field(
                        response,
                        labels=["提示", "用户输入", "回复", "输出"],
                        stop_labels=["下一状态", "下一步状态"]
                    )
                return ns, pmpt

            # 不完全信任 config.USE_ENGLISH：模型可能返回另一种语言格式
            next_state, prompt = _extract_by_mode(use_english)
            if not next_state or not prompt:
                alt_next_state, alt_prompt = _extract_by_mode(not use_english)
                next_state = next_state or alt_next_state
                prompt = prompt or alt_prompt

            # 兜底：字段值有时会被模型拆到下一行，或使用非常规标签格式
            if not next_state:
                for line in response.splitlines():
                    if re.search(r"^\s*(下一状态|下一步状态|Next state|Next State)\s*[:：]\s*", line, re.IGNORECASE):
                        next_state = re.sub(
                            r"^\s*(下一状态|下一步状态|Next state|Next State)\s*[:：]\s*",
                            "",
                            line,
                            flags=re.IGNORECASE,
                        ).strip()
                        break

            if not prompt:
                for idx, line in enumerate(response.splitlines()):
                    if re.search(r"^\s*(提示|用户输入|回复|输出|Prompt|User input|User message|Reply|Response)\s*[:：]\s*", line, re.IGNORECASE):
                        prompt_line = re.sub(
                            r"^\s*(提示|用户输入|回复|输出|Prompt|User input|User message|Reply|Response)\s*[:：]\s*",
                            "",
                            line,
                            flags=re.IGNORECASE,
                        ).strip()
                        if prompt_line:
                            prompt = prompt_line
                        else:
                            remaining = "\n".join(response.splitlines()[idx + 1:]).strip()
                            prompt = remaining
                        break

            if not next_state or not prompt:
                raise ValueError("missing next_state or prompt")

            next_state = _clean_state_text(next_state)
            if not next_state:
                raise ValueError("missing normalized next_state")

            return next_state, prompt
        except (IndexError, AttributeError, ValueError) as e:
            print(f"RealModel: 响应解析失败: {e}", flush=True)
            print(f"RealModel: 原始响应: {response}", flush=True)
            return "<end>", "抱歉，我无法理解当前的状态转移。"
    
    def generate(self, conversation: List[Dict[str, str]], plantuml: str = None) -> Tuple[str, str]:
        """
        使用真实模型生成回复
        
        参数:
            conversation: 对话历史
            plantuml: PlantUML流程图代码（如果需要）
        
        返回:
            Tuple[str, str]: (下一状态, 回复内容)
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用load()方法")
        
        print(f"RealModel: 生成中...", flush=True)
        print(f"RealModel: 对话历史长度 = {len(conversation)}", flush=True)
        
        # 使用tokenizer应用对话模板
        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=self.device)
            inputs["attention_mask"] = attention_mask
        
        # 使用流式生成
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer, 
            skip_prompt=True, 
            timeout=60.0, 
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            **config.GENERATION_CONFIG
        }
        
        # 在后台线程中生成
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 收集生成的文本
        response = ''.join([new_text for new_text in streamer])
        
        print(f"RealModel: 原始响应 = {response}", flush=True)
        
        # 解析响应
        next_state, prompt = self._parse_response(response, use_english=config.USE_ENGLISH)
        
        print(f"RealModel: 下一状态 = {next_state}", flush=True)
        print(f"RealModel: 回复 = {prompt}", flush=True)
        
        return next_state, prompt
    
    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        info = super().get_model_info()
        info.update({
            "model_path": config.MODEL_PATH,
            "device": self.device,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_loaded": self.model is not None
        })
        return info
