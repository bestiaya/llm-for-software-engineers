"""天气应用的两版约定：读模型返回的文本，决定应用下一步做什么。

第 01 章的约定只有城市和天数两个字段，这里叫 v1。
第 02 章把约定扩成查询、请求澄清、不支持三种结果，这里叫 v2。

两版都只做判定，不联网。需要查天气时，只组装将要发出的那条查询，不发送。
第 03 章用判定的结果核对应用的行为。
"""

from __future__ import annotations

import json
from urllib.parse import urlencode


FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Asia/Shanghai"

# 应用自己配置的查询点：城市名 -> (纬度, 经度)。
LOCATIONS = {
    "成都": (30.67, 104.07),
    "上海": (31.23, 121.47),
    "北京": (39.90, 116.40),
    "杭州": (30.27, 120.15),
    "重庆": (29.56, 106.55),
}

# 下面三个数由应用决定，不由模型决定。
DEFAULT_DAYS = 1          # 用户没说天数时查几天
MIN_DAYS = 1              # 应用自己定的下限
MAX_DAYS = 16             # 上限来自天气服务：Open-Meteo 的 /v1/forecast 最多给 16 天

ASK_CITY = "请问要查哪个城市？"
DEFAULT_NOTICE = f"你没有说天数，按 {DEFAULT_DAYS} 天查询。"

# 应用没有去查询时的原因。期望行为可以写明要哪一种。
REASONS = {
    "days_out_of_range": "天数超出应用能查的范围",
    "city_not_configured": "应用没有配置这个城市的查询点",
    "model_unsupported": "模型判断这个版本做不了",
    "invalid_output": "模型交回的文本不符合约定",
}


class ContractError(Exception):
    """模型返回的文本不符合约定。"""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 对象重复出现字段 {key}")
        result[key] = value
    return result


def _parse_object(text):
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except ValueError as error:
        raise ContractError(f"无法解析模型返回的 JSON：{error}") from error
    if not isinstance(value, dict):
        raise ContractError("模型返回的不是一个 JSON 对象。")
    return value


def weather_request(city, days):
    """应用将要发给天气服务的查询。这里只组装，不发送。"""
    latitude, longitude = LOCATIONS[city]
    query = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": TIMEZONE,
        "forecast_days": str(days),
    }
    return {"city": city, "days": days, "url": FORECAST_ENDPOINT + "?" + urlencode(query)}


def _decision(version, outcome, decided_by, message, *, reason=None, extracted=(None, None), request=None):
    return {
        "version": version,
        "outcome": outcome,                 # query / clarify / unsupported / error
        "decided_by": decided_by,           # model：照模型的结果做；program：程序自己的规则
        "reason": reason,                   # 没有去查询时的原因，见 REASONS
        "extracted_city": extracted[0],     # 模型从原话里提取的城市，没有提取时为 None
        "extracted_days": extracted[1],     # 模型从原话里提取的天数，没有提取时为 None
        "weather_request": request,         # 将要发出的查询；不查询时为 None
        "message": message,                 # 应用对用户说的话
    }


def decide_v1(text):
    """两字段约定的对照实现：拿到 city 和 days 就准备去查。"""
    try:
        value = _parse_object(text)
        if value.keys() != {"city", "days"}:
            raise ContractError("查询参数需要 city、days 两个字段。")
        if not isinstance(value["city"], str) or type(value["days"]) is not int:
            raise ContractError("city 须为城市名，days 须为整数。")
    except ContractError as error:
        return _decision("v1", "error", "program", f"应用停止：{error}", reason="invalid_output")
    city, days = value["city"], value["days"]
    if city not in LOCATIONS:
        return _decision(
            "v1", "error", "program", f"应用停止：应用没有配置“{city}”的查询点。",
            reason="city_not_configured", extracted=(city, days),
        )
    return _decision(
        "v1", "query", "model", f"查询{city} {days} 天的预报。",
        extracted=(city, days), request=weather_request(city, days),
    )


def decide_v2(text):
    """三种结果的约定，加上应用自己的查询点、默认天数和天数范围。"""
    try:
        value = _parse_object(text)
        result = value.get("result")
        if result == "query":
            if value.keys() != {"result", "city", "days"}:
                raise ContractError("查询结果需要 result、city、days 三个字段。")
            city, days = value["city"], value["days"]
            if not isinstance(city, str) or not city:
                raise ContractError("city 须为城市名。")
            if days is not None and type(days) is not int:
                raise ContractError("days 须为整数，用户没说时为 null。")
        elif result == "clarify":
            if value.keys() != {"result", "missing"} or value["missing"] != "city":
                raise ContractError("请求澄清时需要 result 和 missing，missing 只能是 city。")
        elif result == "unsupported":
            if value.keys() != {"result", "reason"} or not isinstance(value["reason"], str) or not value["reason"]:
                raise ContractError("不支持时需要 result 和一句 reason。")
        else:
            raise ContractError("result 只能是 query、clarify 或 unsupported。")
    except ContractError as error:
        return _decision("v2", "error", "program", f"应用停止：{error}", reason="invalid_output")

    if result == "clarify":
        return _decision("v2", "clarify", "model", ASK_CITY)
    if result == "unsupported":
        return _decision(
            "v2", "unsupported", "model", f"这个版本还做不了：{value['reason']}", reason="model_unsupported",
        )

    # 以下是程序的规则：查询点、默认天数、天数范围。
    extracted = (city, days)
    if city not in LOCATIONS:
        return _decision(
            "v2", "unsupported", "program", f"这个版本还做不了：没有配置“{city}”的查询点。",
            reason="city_not_configured", extracted=extracted,
        )
    used_default = days is None
    if used_default:
        days = DEFAULT_DAYS
    if not MIN_DAYS <= days <= MAX_DAYS:
        return _decision(
            "v2", "unsupported", "program",
            f"这个版本还做不了：只能查 {MIN_DAYS} 到 {MAX_DAYS} 天，收到的是 {days} 天。",
            reason="days_out_of_range", extracted=extracted,
        )
    message = f"查询{city} {days} 天的预报。"
    if used_default:
        message = DEFAULT_NOTICE + message
    return _decision("v2", "query", "model", message, extracted=extracted, request=weather_request(city, days))


DECIDERS = {"v1": decide_v1, "v2": decide_v2}


def observe(decision):
    """评分只核对这里列出的事实。

    查询的城市和天数取自将要发出的那条查询，是否说明了默认天数取自对用户说的话。
    这样，应用实际做的事和它自己打的标记不一致时，评分看的是实际做的事。
    """
    request = decision["weather_request"]
    return {
        "outcome": decision["outcome"],
        "decided_by": decision["decided_by"],
        "reason": decision["reason"],
        "extracted_city": decision["extracted_city"],
        "extracted_days": decision["extracted_days"],
        "city": request["city"] if request else None,
        "days": request["days"] if request else None,
        "sends_weather_request": request is not None,
        "default_disclosed": DEFAULT_NOTICE in decision["message"],
    }


# 期望行为的写法：每种结果必须写哪些项，可以多写哪些项。
EXPECTED_REQUIRED = {
    "query": {"city": str, "days": int, "default_disclosed": bool},
    "clarify": {},
    "unsupported": {},
}
EXPECTED_OPTIONAL = {
    "query": {},
    "clarify": {},
    "unsupported": {"decided_by": str, "reason": str, "extracted_city": str, "extracted_days": int},
}


def expected_problems(expected):
    """检查一条期望行为写得合不合规矩，返回问题的清单；没有问题时为空。"""
    if not isinstance(expected, dict):
        return ["需要写成一个对象"]
    outcome = expected.get("outcome")
    if outcome not in EXPECTED_REQUIRED:
        return ["outcome 只能是 query、clarify 或 unsupported"]
    required, optional = EXPECTED_REQUIRED[outcome], EXPECTED_OPTIONAL[outcome]
    problems = []
    for field, kind in required.items():
        if field not in expected:
            problems.append(f"少写了 {field}")
        elif type(expected[field]) is not kind:
            problems.append(f"{field} 的类型不对")
    for field, value in expected.items():
        if field == "outcome" or field in required:
            continue
        if field not in optional:
            problems.append(f"这种结果用不到 {field}")
        elif type(value) is not optional[field]:
            problems.append(f"{field} 的类型不对")
    if expected.get("decided_by", "model") not in ("model", "program"):
        problems.append("decided_by 只能是 model 或 program")
    if "reason" in expected and expected["reason"] not in REASONS:
        problems.append("reason 只能是 " + "、".join(REASONS))
    return problems


def implied(expected):
    """一条期望行为隐含的要求：只有期望查询时，才应当组装天气查询。"""
    return {**expected, "sends_weather_request": expected["outcome"] == "query"}


def same(left, right):
    """严格相等：类型也要一样。Python 里 True == 1，天数写成 true 不能算对。"""
    return type(left) is type(right) and left == right
