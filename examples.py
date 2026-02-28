# -*- coding: utf-8 -*-
"""
PlantUML Examples
"""
import config


# User test task 1: Travel Itinerary Planning Service (English)
EXAMPLE_USER_TEST_TASK_1 = """
@startuml
start

:Hello, this is the itinerary planning service;
:Please choose a destination (city/country/etc.);
:Please choose trip duration (e.g., 3/5/7/etc. days);
:Please choose a budget range (budget/comfort/premium/etc.);

repeat
  if (Prefer a lively area around the accommodation?) then (Yes)
    :Suggest staying in the city center or scenic area;
  else (No)
    :Recommend a quiet place to stay;
  endif

  if (Car rental selected?) then (Yes)
    :Remind user to prepare license/credit card and confirm child seat needs;
  else (No)
    :Generate a public transit or chartered-car plan;
  endif

  repeat while(Generate itinerary draft and send to the user, does the user want changes?) is(Yes) not (No)

  if (Confirm whether to pay a deposit to lock resources?) then (Yes)
    :Lock hotel/car/tickets and send confirmation;
    stop
  else (No)
    :Provide other payment methods or defer payment;
  endif

stop
@enduml
"""


# User test task 2: Fitness Plan Coaching Service (English)
EXAMPLE_USER_TEST_TASK_2 = """
@startuml
start

:Hello, this is the fitness plan coaching service;
:Please choose a goal (fat loss/muscle gain/posture/endurance/etc.);
:Please choose weekly frequency (2/3/4/5/etc. days);
:Please choose training location (home/gym/etc.);

if (Beginner user?) then (Yes)
  :Recommend exercises: squat/knee push-up/plank;
else (No)
  :Recommend exercises: squat/bench press/deadlift/row;
endif

repeat
  :Ask user to finish warm-up (5 min) and start training;
  
  if (Discomfort or pain during training?) then (Yes)
    :Advise stopping that exercise and switching to a lower-intensity alternative;
  else (No)
    :Continue and complete the sets;
  endif


  if (Record today's completion (sets/weight/duration). Plan completed?) then (Yes)
    :Today's check-in completed;
  else (No)
    :Suggest adjustments: reduce sets or switch to easy intensity;
  endif

  if (User wants to modify the plan?) then (Yes)
    :Modify frequency/intensity/exercise library and save;
  else (No)
    :Keep the current plan;
  endif

repeat while(Weekly check-ins completed?) is(No) not (Yes)

:Congratulations on completing this week's plan; generate next-week recommendations;
stop
@enduml
"""

# User test task 1: Travel Itinerary Planning Service (Chinese)
EXAMPLE_USER_TEST_TASK_1_ZH = """@startuml
start
:您好，这里是行程定制客服;
:请选择目的地(城市/国家);
:请选择出行天数(如: 3/5/7天);
:请选择预算区间(经济/舒适/高端);

repeat
  if (喜欢住宿周围较为热闹?) then (是)
    :建议选择市中心或景区内住宿;
  else (否)
    :推荐安静的住宿地点;
  endif

  if (选择租车?) then (是)
    :提醒准备驾照/信用卡，并确认是否需要儿童座椅;
  else (否)
    :生成公共交通或包车方案;
  endif

  repeat while(生成行程草案并发送给用户，用户要改动?) is(否) not (是)

  if (确认是否支付定金锁定资源，支付成功?) then (是)
    :锁定酒店/车/门票并发送确认单;
    stop
  else (否)
    :提供其他支付方式或延后支付;
  endif


stop
@enduml"""

# User test task 2: Fitness Plan Coaching Service (Chinese)
EXAMPLE_USER_TEST_TASK_2_ZH = """@startuml
start
:您好，这里是健身计划制定客服;
:请选择目标(减脂/增肌/体态/提升耐力);
:请选择每周频次(2/3/4/5天);
:请选择训练地点(家/健身房);

if (用户新手?) then (是)
  :推荐动作: 深蹲/俯卧撑(跪姿)/平板支撑;
else (否)
  :推荐动作: 硬拉/划船(按目标调整);
endif
repeat

  :请用户完成热身(5分钟)，此后将开始训练;

  if (训练中不适/疼痛?) then (是)
    :建议停止该动作并改为低强度替代动作;
  else (否)
    :继续完成组数;
  endif

  if (请记录今日完成情况(组数/重量/时长)。完成计划?) then (是)
    :今日打卡成功;
  else (否)
    :建议调整: 降低组数或改为轻松强度;
  endif

  if (用户要修改计划?) then (是)
    :修改频次/强度/动作库并保存;
  else (否)
    :保持原计划;
  endif

repeat while(本周打卡完成?) is(否) not (是)

:恭喜完成本周计划，生成下周建议;
stop
@enduml"""



DEFAULT_EXAMPLE_KEY = "user_test_task_1"
DEFAULT_EXAMPLE = EXAMPLE_USER_TEST_TASK_1

AVAILABLE_EXAMPLES = {
    
    "user_test_task_1": {
        "name": "Travel Itinerary Planning Service",
        "plantuml": EXAMPLE_USER_TEST_TASK_1,
        "description": "User testing task: complete a travel itinerary planning workflow.",
        "policy": {"roles": ["guest", "admin"]},
    },
    "user_test_task_2": {
        "name": "Fitness Plan Coaching Service",
        "plantuml": EXAMPLE_USER_TEST_TASK_2,
        "description": "User testing task: complete a fitness plan coaching workflow.",
        "policy": {"roles": ["guest", "admin"]},
    },
}

# route /user-test
USER_TEST_TASK_KEYS = [     
    "user_test_task_1",
    "user_test_task_2",
]

# route /
VISIBLE_UML_EXAMPLE_KEYS = [
    "user_test_task_1",
    "user_test_task_2",
]


_EXAMPLE_I18N = {
    "user_test_task_1": {
        "en": {"name": "Travel Itinerary Planning Workflow", "description": "User testing task: complete a travel itinerary planning workflow.", "plantuml": EXAMPLE_USER_TEST_TASK_1},
        "zh": {"name": "行程定制客服流程", "description": "用户测试任务：完成行程定制客服流程。", "plantuml": EXAMPLE_USER_TEST_TASK_1_ZH},
    },
    "user_test_task_2": {
        "en": {"name": "Fitness Plan Coaching Workflow", "description": "User testing task: complete a fitness plan coaching workflow.", "plantuml": EXAMPLE_USER_TEST_TASK_2},
        "zh": {"name": "健身计划制定客服流程", "description": "用户测试任务：完成健身计划制定客服流程。", "plantuml": EXAMPLE_USER_TEST_TASK_2_ZH},
    },
}


def normalize_lang(lang=None):
    if lang in ("zh", "cn", "zh-CN", "zh_CN"):
        return "zh"
    if lang in ("en", "en-US", "en_US"):
        return "en"
    return "en" if config.UI_USE_ENGLISH else "zh"


def _localized_info(example_key, lang=None):
    base = AVAILABLE_EXAMPLES.get(example_key)
    if not base:
        return None
    resolved = normalize_lang(lang)
    localized = _EXAMPLE_I18N.get(example_key, {}).get(resolved, {})
    merged = dict(base)
    if "name" in localized:
        merged["name"] = localized["name"]
    if "description" in localized:
        merged["description"] = localized["description"]
    if "plantuml" in localized:
        merged["plantuml"] = localized["plantuml"]
    return merged


def get_example(example_key=None, lang=None):
    """
    获取示例流程图
    :param example_key: 示例的key，如果为None则返回默认示例
    :return: PlantUML代码字符串
    """
    if example_key is None:
        return _localized_info(DEFAULT_EXAMPLE_KEY, lang)["plantuml"]

    example = _localized_info(example_key, lang)
    if example:
        return example["plantuml"]
    return _localized_info(DEFAULT_EXAMPLE_KEY, lang)["plantuml"]


def get_all_examples(lang=None):
    """
    获取所有可用的示例
    :return: 示例字典
    """
    return {key: _localized_info(key, lang) for key in AVAILABLE_EXAMPLES.keys()}


def get_user_test_tasks(task_keys=None, limit=None, lang=None):
    """
    获取用户测试任务列表（按顺序）
    :return: List[Tuple[key, info]]
    """
    keys = list(task_keys or config.USER_STUDY_TASK_KEYS or USER_TEST_TASK_KEYS)
    effective_limit = config.USER_STUDY_TASK_LIMIT if limit is None else limit
    tasks = []
    for key in keys:
        info = _localized_info(key, lang)
        if info:
            meta = USER_TEST_TASK_KEYS.get(key, {})
            merged = dict(info)
            if meta.get("name"):
                if normalize_lang(lang) == "en":
                    merged["name"] = meta["name"]
            if meta.get("description"):
                if normalize_lang(lang) == "en":
                    merged["description"] = meta["description"]
            tasks.append((key, merged))
            if effective_limit and len(tasks) >= effective_limit:
                break
    return tasks


def list_example_names():
    """
    列出所有示例名称
    :return: 示例名称列表
    """
    return [(key, info["name"]) for key, info in AVAILABLE_EXAMPLES.items()]
