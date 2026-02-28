# -*- coding: utf-8 -*-
import os

# ==================== Model Config ====================

# Model: "dummy" or "real"
MODEL_TYPE = os.environ.get("FLOWGUARD_MODEL_TYPE", "dummy")

MODEL_PATH = os.environ.get(
    "FLOWGUARD_MODEL_PATH", 
)

GENERATION_CONFIG = {
    "pad_token_id": 151643,
    "eos_token_id": [151643, 151645],
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True
}

DEVICE = os.environ.get("FLOWGUARD_DEVICE", "") 

# ==================== Flask Config ====================

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5002
FLASK_DEBUG = os.environ.get("FLOWGUARD_DEBUG", "true").lower() == "true"

# ==================== Security Config ====================

ENVIRONMENT = os.environ.get("FLOWGUARD_ENV", "development")
SECRET_KEY = os.environ.get("FLOWGUARD_SECRET_KEY", "dev-insecure-secret")
REQUIRE_HTTPS = os.environ.get("FLOWGUARD_REQUIRE_HTTPS", "false").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.environ.get("FLOWGUARD_RATE_LIMIT", "60"))
TOKEN_TTL_SECONDS = int(os.environ.get("FLOWGUARD_TOKEN_TTL", "28800"))
NONCE_TTL_SECONDS = int(os.environ.get("FLOWGUARD_NONCE_TTL", "300"))

# ==================== Prompts ====================

SYSTEM_PROMPT_ZH = """你是一个根据当前状态预测下一状态的机器人。这是一段使用plantuml描述的流程图：

{plantuml}

你需要根据plantuml的内容、当前状态以及用户的回复，进行状态切换，预测下一状态，并输出提示。开始和结束状态分别是<start>和<end>，其他状态是plantuml中的节点名称。请根据信息输出下一状态和回复。你的回复应该严格遵循下面的格式：
下一状态：
状态名称

提示：
提示内容。

如果给出了“合法下一状态”列表，你必须只从这个列表里选择一个完全一致的节点名称作为“下一状态”，不要输出列表之外的节点，也不要添加分号、解释或额外前缀。

其中，提示内容可以不用特别刻板，你可以首先感谢、回应或者简要重复用户的上一轮回答，使得用户和你的交互更愉悦。然后，你需要根据用户的选项，表达这一轮需要用户做的事，可以加入更加亲和、拟人化的表达。最后，你可以根据情况提供一些额外的提示或者建议，并且表达你会倾心服务用户。譬如在生成方案、按偏好完成用户的需求的阶段、生成了用户结果的时候，可以简要用十五个字左右概括一下可能的方案。"""

USER_PROMPT_ZH = """当前状态：{current_state}
用户输入：{user_input}
合法下一状态：{allowed_next_states}"""

SYSTEM_PROMPT_EN = """You are an assistant that predicts the next state based on the current state. Here is a flowchart written in PlantUML:

{plantuml}

Using this PlantUML, the current state, and the user's message, switch states, predict the next state, and provide a prompt. The start and end states are <start> and <end>; other states are node names in the PlantUML. Please output the next state and a prompt in the following format:
Next state:
<state name>

Prompt:
<prompt content>.

If an \"Allowed next states\" list is provided, you must choose exactly one state name from that list and copy it exactly. Do not output a state outside the list, and do not add semicolons or explanations to the state name.
The prompt content doesn't need to be overly rigid. You can first express gratitude, respond, or briefly repeat the user's previous response to make the interaction between the user and you more pleasant. Then, based on the user's options, you need to express what the user needs to do in this round. You can incorporate more friendly and personified expressions. Finally, you can provide some additional prompts or suggestions based on the situation, and express your commitment to serving the user. For example, during the stage of generating a solution, fulfilling the user's requests according to preferences, or when generating the user's results, you can briefly summarize the possible solutions in about fifteen words.
"""

USER_PROMPT_EN = """Current state: {current_state}
User message: {user_input}
Allowed next states: {allowed_next_states}"""

USE_ENGLISH = True

# ==================== UI ====================

# default lang: True=English, False=Chinese
UI_USE_ENGLISH = os.environ.get("FLOWGUARD_UI_USE_ENGLISH", "true").lower() == "true"

USER_STUDY_TASK_KEYS = [
    key.strip() for key in os.environ.get("FLOWGUARD_USER_STUDY_TASK_KEYS", "").split(",") if key.strip()
]

USER_STUDY_TASK_LIMIT = int(os.environ.get("FLOWGUARD_USER_STUDY_TASK_LIMIT", "0"))

# ==================== funcs ====================

def get_system_prompt(plantuml, use_english=None):
    if use_english is None:
        use_english = USE_ENGLISH
    
    template = SYSTEM_PROMPT_EN if use_english else SYSTEM_PROMPT_ZH
    return template.format(plantuml=plantuml)

def get_user_prompt(current_state, user_input, use_english=None, allowed_next_states=None):
    if use_english is None:
        use_english = USE_ENGLISH
    
    template = USER_PROMPT_EN if use_english else USER_PROMPT_ZH
    allowed = allowed_next_states or "(none)"
    return template.format(current_state=current_state, user_input=user_input, allowed_next_states=allowed)

def is_dummy_model():
    return MODEL_TYPE.lower() == "dummy"

def is_real_model():
    return MODEL_TYPE.lower() == "real"

def validate_config():
    if ENVIRONMENT.lower() == "production":
        if SECRET_KEY == "dev-insecure-secret":
            raise RuntimeError("生产环境必须设置 FLOWGUARD_SECRET_KEY")
        if FLASK_DEBUG:
            raise RuntimeError("生产环境必须关闭调试模式 (FLOWGUARD_DEBUG=false)")
        if REQUIRE_HTTPS and os.environ.get("FLOWGUARD_BEHIND_TLS", "false").lower() != "true":
            raise RuntimeError("要求 HTTPS 时需设置 FLOWGUARD_BEHIND_TLS=true 或终止代理层提供 TLS")
