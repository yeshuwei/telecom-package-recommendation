"""
价位选择智能体 - 在已选套餐系列内推荐具体价位

优化内容：
1. 修复价位评分权重归一化（0.9 → 1.0）
2. 优化超预算惩罚机制（渐进式惩罚）
3. 调整系列分和价位分组合方式（相乘 → 加权平均）
4. 改进缺失值处理逻辑（根据用户需求动态调整）
5. 增加评分可解释性（score_breakdown）
6. 添加归一化和多样性控制
"""
import re
import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from agent.state import AgentState
from configs.config import PRICE_SHEET_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 从文本中提取数值（支持带单位的数值，如 "100GB" -> 100）
def _extract_number(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# 针对不同数据格式进行不同处理 1.None 2. 数值类型 3. 字符串类型
def _normalize_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return _extract_number(str(value))


def _sheet_to_series_name(name: str) -> str:
    return str(name).strip()


# 匹配套餐系列（支持模糊匹配）
# def _match_series(category: str, sheet_names: List[str]) -> List[str]:
#     c = category.strip()
#     exact = [n for n in sheet_names if c in n or n in c]
#     if exact:
#         return exact
#     tokens = [t for t in re.split(r"[\s/（）()·-]", c) if t]
#     matched = []
#     for n in sheet_names:
#         score = sum(1 for t in tokens if t and t in n)
#         if score > 0:
#             matched.append(n)
#     return matched or sheet_names

# 从明确需求文本中提取流量/通话量/数据量等数值或是根据描述进行映射
def _bucket_from_text(text: Optional[str], light: float, mid: float, heavy: float) -> Optional[float]:
    if not text:
        return None
    s = str(text)
    if re.search(r"轻度|偏低|不多|较少", s):
        return light
    if re.search(r"中度|一般|正常", s):
        return mid
    if re.search(r"重度|较多|很多|无限|不限", s):
        return heavy
    num = _extract_number(s)
    return num if num is not None else None


class PriceSelectionAgent:
    def __init__(self):
        self.frames: Dict[str, pd.DataFrame] = {}

    def load_tariff_excel(self, path: str) -> bool:
        try:
            book = pd.read_excel(path, sheet_name=None)
            frames = {}
            for sheet_name, df in book.items():
                df = df.copy()
                cols = {c: str(c).strip() for c in df.columns}
                # 对列进行重命名
                df.rename(columns=cols, inplace=True)
                name_col = None
                for cand in ["套餐名称", "名称", "资费名称", "产品名称"]:
                    if cand in df.columns:
                        name_col = cand
                        break
                price_col = None
                for cand in ["价格", "月费", "套餐价", "资费价格", "售价"]:
                    if cand in df.columns:
                        price_col = cand
                        break
                if not name_col or not price_col:
                    continue
                # astype(str) 强制转换为字符串类型，避免后续处理时类型错误
                df["plan_name"] = df[name_col].astype(str)
                # apply() 对 price_col 列中的每个元素应用 _normalize_price 函数
                df["price"] = df[price_col].apply(_normalize_price)
                df = df[df["price"].notna() & (df["plan_name"].str.len() > 0)]
                other_cols = [c for c in df.columns if c not in [name_col, price_col, "plan_name", "price"]]
                # 套餐独立的属性聚合为字典
                df["features"] = df[other_cols].to_dict(orient="records")
                # 字典结构：键为列名，值为对应DataFrame
                frames[_sheet_to_series_name(sheet_name)] = df[["plan_name", "price", "features"]]
            if not frames:
                logger.error("价位表解析为空")
                return False
            self.frames = frames
            logger.info(f"载入价位表：{len(frames)} 个系列")
            return True
        except Exception as e:
            logger.error(f"载入价位表失败: {e}")
            return False

    # 提取用户需求为数字（预算、数据量、通话量）
    def build_user_targets(self, state: AgentState) -> Dict[str, Optional[float]]:
        override = state.get("user_database_override")
        raw = state.get("raw_user_info") or {}
        budget = None
        data = None
        voice = None
        if override:
            budget = _bucket_from_text(getattr(override, "budget", None), 80, 180, 400)
            dn = getattr(override, "data_needs", None)
            if dn and re.search(r"不限|无限", str(dn)):
                data = 100.0
            else:
                data = _bucket_from_text(dn, 5, 30, 100)
            voice = _bucket_from_text(getattr(override, "call_minutes", None), 100, 500, 1000)
        if budget is None:
            for k, v in raw.items():
                if isinstance(k, str) and re.search(r"近三月平均消费\(套餐级\)", k):
                    budget = _normalize_price(v)
                    if budget is not None:
                        break
        if data is None:
            for k, v in raw.items():
                if isinstance(k, str) and re.search(r"前三月流量平均消耗", k):
                    num = _extract_number(v)
                    if num is not None:
                        data = num / 1024
                        break
        if voice is None:
            for k, v in raw.items():
                if isinstance(k, str) and re.search(r"语音|通话|分钟", k):
                    num = _extract_number(v)
                    if num is not None:
                        voice = num
                        break
        return {"budget": budget, "data": data, "voice": voice}

    # 提取套餐独立属性中的 “国内流量”和“国内通话”字段的值
    def _extract_feature_numbers(self, features: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """
        :param features: 接收每一行数据的"feature"属性中存储的字典，存储的是独特列名：对应的值
        :return:
        """
        data_val = None
        voice_val = None
        for k, v in (features or {}).items():
            # s = f"{k}{v}"
            if data_val is None and re.search(r"国内流量", k):
                # if re.search(r"不限|无限", s):
                #     data_val = 100.0
                # else:
                data_val = _extract_number(v)
            if voice_val is None and re.search(r"国内通话", k):
                voice_val = _extract_number(v)
        return data_val, voice_val

    def _preference_bonus(self, series: str, state: AgentState) -> float:
        needs = state.get("user_explicit_needs")
        bonus = 0.0
        s = series
        if needs:
            if getattr(needs, "video_needs", None) and re.search(r"视频|影视|iTV|畅玩", s):
                bonus += 0.15
            if getattr(needs, "education_needs", None) and re.search(r"教育|课堂|少儿", s):
                bonus += 0.15
            if getattr(needs, "smart_home_needs", None) and re.search(r"智慧家庭|家庭|WiFi|安防", s):
                bonus += 0.1
            if getattr(needs, "need_broadband", None) and re.search(r"宽带|融合", s):
                bonus += 0.1
        return min(bonus, 0.3)

    # 移除 _generate_reason 方法
    # 推荐理由将由 ResponseGenerationAgent 基于详细的评分数据自然生成

    def score_and_select(self, series: str, df: pd.DataFrame, targets: Dict[str, Optional[float]],
                         state: AgentState, series_score: Optional[float], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        对套餐系列内的不同价位进行评分和筛选
        
        评分机制：
        1. 预算匹配度（55%权重）：渐进式惩罚超预算套餐
        2. 流量匹配度（28%权重）：根据用户需求动态处理缺失值
        3. 通话匹配度（17%权重）：根据用户需求动态处理缺失值
        4. 最终得分 = 0.6 * 系列分 + 0.4 * 价位分（加权平均，避免系列分过度影响）
        
        :param series_score: 套餐系列得分（来自recommendation_agent）
        :param series: 套餐系列名称
        :param df: 存储该套餐系列的DataFrame
        :param targets: 提取的用户需求信息（budget/data/voice）
        :param state: 多智能体系统状态
        :param top_n: 返回前N个套餐
        :return: 评分后的套餐列表
        """
        # 提取用户需求
        budget = targets.get("budget")
        data_need = targets.get("data")
        voice_need = targets.get("voice")
        
        rows = []
        for _, row in df.iterrows():
            price = float(row["price"]) if row["price"] is not None else None
            # 提取套餐的流量和通话时长
            d_val, v_val = self._extract_feature_numbers(row["features"])
            
            # ========== 1. 预算匹配度（优化：渐进式惩罚） ==========
            aff = 0.8  # 默认分数
            if price is not None and budget and budget > 0:
                if price <= budget:
                    # 价格在预算内，根据接近程度给分
                    aff = 1.0 - 0.2 * abs(price - budget) / budget
                    aff = max(aff, 0.85)  # 预算内最低0.85
                else:
                    # 价格超预算，渐进式惩罚
                    over_ratio = (price - budget) / budget
                    if over_ratio <= 0.1:  # 超出10%以内
                        aff = 0.90
                    elif over_ratio <= 0.2:  # 超出10%-20%
                        aff = 0.80
                    elif over_ratio <= 0.3:  # 超出20%-30%
                        aff = 0.70
                    elif over_ratio <= 0.5:  # 超出30%-50%
                        aff = 0.60
                    else:  # 超出50%以上
                        aff = 0.45
            
            # ========== 2. 流量匹配度（优化：根据需求处理缺失值） ==========
            d_score = 0.5  # 默认分数
            if data_need:
                if d_val is None:
                    # 缺失值处理：根据用户需求大小调整
                    if data_need > 100:  # 重度用户
                        d_score = 0.3  # 缺失信息，降低分数
                    elif data_need > 50:  # 中度用户
                        d_score = 0.4
                    else:  # 轻度用户
                        d_score = 0.5
                elif d_val >= data_need:
                    # 流量充足
                    d_score = 1.0
                else:
                    # 流量不足，按比例给分
                    d_score = max(d_val / data_need, 0.0)
            
            # ========== 3. 通话匹配度（优化：根据需求处理缺失值） ==========
            v_score = 0.5  # 默认分数
            if voice_need:
                if v_val is None:
                    # 缺失值处理：根据用户需求大小调整
                    if voice_need > 1000:  # 重度用户
                        v_score = 0.3
                    elif voice_need > 500:  # 中度用户
                        v_score = 0.4
                    else:  # 轻度用户
                        v_score = 0.5
                elif v_val >= voice_need:
                    # 通话充足
                    v_score = 1.0
                else:
                    # 通话不足，按比例给分
                    v_score = max(v_val / voice_need, 0.0)
            
            # ========== 4. 价位评分（优化：权重归一化到1.0） ==========
            price_score = 0.55 * aff + 0.28 * d_score + 0.17 * v_score
            price_score = max(0.0, min(price_score, 1.0))
            
            # ========== 5. 系列评分 ==========
            # 如果recommendation_agent提供了系列分，使用它；否则使用偏好加分
            base_series_score = series_score if series_score is not None else self._preference_bonus(series, state)
            base_series_score = max(0.0, min(base_series_score, 1.0))
            
            # ========== 6. 最终得分（优化：加权平均而非相乘） ==========
            # 相乘模式：系列分过低会严重拖累价位分
            # 加权平均：更平衡，系列分占60%，价位分占40%
            total = 0.6 * base_series_score + 0.4 * price_score
            
            # ========== 7. 记录结果（包含详细评分数据，供ResponseGenerationAgent使用） ==========
            reason_parts = []
            if aff >= 0.85:
                reason_parts.append("价格合适")
            elif aff >= 0.7:
                reason_parts.append("价格稍高")
            else:
                reason_parts.append("超出预算较多")

            if d_score >= 0.8:
                reason_parts.append("流量充足")
            elif d_score >= 0.5:
                reason_parts.append("流量适中")
            else:
                reason_parts.append("流量偏少")

            if v_score >= 0.8:
                reason_parts.append("通话充足")

            simple_reason = "、".join(reason_parts) if reason_parts else "综合匹配"
            rows.append({
                "series": series,
                "plan_name": row["plan_name"],
                "price": price,
                "series_score": round(base_series_score, 4),
                "price_score": round(price_score, 4),
                "score": round(total, 4),
                "reason": simple_reason,
                "score_breakdown": {
                    "budget_match": round(aff, 3),
                    "data_match": round(d_score, 3),
                    "voice_match": round(v_score, 3),
                    "series_source": "llm" if series_score is not None else "preference"
                },
                # 原始套餐属性，供大模型参考
                "package_details": row.to_dict()
            })
        
        # ========== 9. 排序并返回Top-N ==========
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:top_n]

    def _normalize_and_diversify(self, results: List[Dict[str, Any]], max_per_series: int = 2) -> List[Dict[str, Any]]:
        """
        归一化评分并控制多样性（避免某个系列占据过多推荐位）
        
        :param results: 原始评分结果
        :param max_per_series: 每个系列最多保留几个套餐
        :return: 归一化并多样化后的结果
        """
        if not results:
            return results
        
        # 1. Min-Max归一化
        all_scores = [r["score"] for r in results]
        max_score = max(all_scores)
        min_score = min(all_scores)
        score_range = max_score - min_score
        
        for r in results:
            if score_range > 0:
                r["normalized_score"] = (r["score"] - min_score) / score_range
            else:
                r["normalized_score"] = 1.0
        
        # 2. 按归一化分数排序
        results.sort(key=lambda x: x["normalized_score"], reverse=True)
        
        # 3. 多样性控制：每个系列最多保留max_per_series个
        series_count = {}
        filtered_results = []
        for r in results:
            series = r["series"]
            count = series_count.get(series, 0)
            if count < max_per_series:
                filtered_results.append(r)
                series_count[series] = count + 1
        
        logger.info(f"归一化和多样性控制：{len(results)} → {len(filtered_results)} 个套餐")
        return filtered_results

    def process(self, state: AgentState) -> AgentState:
        """
        价位选择智能体主流程
        
        流程：
        1. 加载价位表
        2. 提取用户需求（预算、流量、通话）
        3. 对每个系列的套餐进行评分
        4. 归一化和多样性控制
        5. 生成推荐结果
        """
        if not self.frames and not self.load_tariff_excel(PRICE_SHEET_PATH):
            state["final_response"] = "价位表载入失败，暂时无法进行价位推荐。"
            state["next_node_to_call"] = "END"
            return state

        scored = state.get("scored_package_series") or []
        filtered = state.get("filtered_package_categories") or []
        sheet_names = list(self.frames.keys())
        targets = self.build_user_targets(state)
        
        logger.info(f"用户需求目标：预算={targets.get('budget')}元, 流量={targets.get('data')}GB, 通话={targets.get('voice')}分钟")
        
        results: List[Dict[str, Any]] = []
        # 存储系列及其对应的得分
        series_score_map = {item.get("series"): float(item.get("score", 0)) for item in scored if
                            isinstance(item, dict)}
        to_pick: List[Tuple[str, Optional[float]]] = []

        # 确定要评分的系列
        for name in filtered:
            if name in self.frames:
                to_pick.append((name, series_score_map.get(name)))
        if not to_pick:
            logger.warning("未找到匹配的套餐系列，使用所有系列")
            to_pick = [(s, series_score_map.get(s)) for s in sheet_names]

        # 对每个系列进行评分（每个系列选Top-2）
        for s, s_score in to_pick:
            df = self.frames[s]
            logger.info(f"正在评分系列【{s}】（系列分={s_score if s_score else '无'}，共{len(df)}个价位）")
            selected = self.score_and_select(s, df, targets, state, s_score, top_n=2)
            results.extend(selected)
            if selected:
                inner_list = [
                    '{}({:.3f})'.format(x['plan_name'], x['score'])
                    for x in selected
                ]
                logger.info(f"  选出{len(selected)}个价位：{inner_list}")
        
        # 归一化和多样性控制
        results = self._normalize_and_diversify(results, max_per_series=2)
        
        state["price_selection_results"] = results

        # 生成推荐结果文本
        if results:
            lines: List[str] = []
            by_series: Dict[str, List[Dict[str, Any]]] = {}
            for r in results:
                by_series.setdefault(r["series"], []).append(r)
            
            for s, items in by_series.items():
                lines.append(f"【{s}】")
                for i in items:
                    price_str = f"¥{int(i['price'])}" if i['price'] is not None else "价格未知"
                    score_str = f"综合评分{i['score']:.2f}"
                    lines.append(f"  • {i['plan_name']}（{price_str}）- {i['reason']} [{score_str}]")
            
            state["final_response"] = "✅ 已为您筛选出最合适的套餐价位：\n\n" + "\n".join(lines)
            logger.info(f"✅ 价位选择完成，共推荐{len(results)}个套餐")
        else:
            state["final_response"] = "⚠️ 暂未找到符合条件的具体价位，建议适度放宽预算或需求约束。"
            logger.warning("未找到符合条件的套餐")
        
        state["next_node_to_call"] = "response_generation"
        return state


price_selection_agent = PriceSelectionAgent()


def price_selection_node(state: AgentState) -> AgentState:
    return price_selection_agent.process(state)
