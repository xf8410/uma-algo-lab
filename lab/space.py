# -*- coding: utf-8 -*-
"""参数搜索空间：70 字段与 Rust ParamOverride 完全对齐。

center 取自 GA 全量轮（r5+r6 共 264 轮）最高适应度基因组：
round 8 / 赤心的驯鹿小姐 目白善信 / fitness 71415.6 / hash 9d8ce67d58616c83。
GA 在该参数面已收敛（同 HEAD 确定性复现已证），本实验室从该点出发做
更精细的连续优化（CMA-ES/TPE/SA/CEM 赛马）。f32 ±40%，i32 ±30%，
bool 与 GA 未覆盖字段固定，数组 null 槽位固定。
值域生成 2026-09-17。
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

@dataclass
class Field:
    name: str
    typ: str          # f32 | i32 | u32 | bool | arri32 | arrf32 | arrusize
    lo: Union[float, int, bool, list, None] = None
    hi: Union[float, int, None] = None

    def is_fixed(self) -> bool:
        if self.typ == "bool": return True
        if self.typ.startswith("arr"):
            return all(s is None for s in self.lo)
        return self.lo is None and self.hi is None

FREE_FIELDS: List[Field] = [
    Field("vital_rest", "i32", 25, 48),  # center=37
    Field("wisdom_vital_floor", "i32", 25, 46),  # center=36
    Field("motivation_outing", "i32"),  # preset 固定（GA 未覆盖）
    Field("status_rate", "f32", 0.6, 1.4),  # center=1
    Field("pt_tradeoff", "f32", 26.7331, 62.3773),  # center=44.55522
    Field("pt_tradeoff_shining", "f32", 13.1535, 30.6915),  # center=21.922499
    Field("pt_tradeoff_super", "f32", 17.4743, 40.7735),  # center=29.123894
    Field("cap_discount_weight", "f32", 0.7268, 1.6959),  # center=1.2113717
    Field("failure_penalty", "f32", 36.0, 84.0),  # center=60
    Field("effective_ramen_failure", "bool", False),  # GA 定型
    Field("shining_bonus", "f32", 36.0, 84.0),  # center=60
    Field("train_vital_value", "f32", 1.0149, 2.368),  # center=1.6914387
    Field("rest_base", "f32", 27.4333, 64.011),  # center=45.722157
    Field("rest_vital_value", "f32"),  # preset 固定（GA 未覆盖）
    Field("rest_target_vital", "i32", 38, 71),  # center=55
    Field("race_panel_discount", "f32"),  # preset 固定（GA 未覆盖）
    Field("race_free_urgency_weight", "f32", 1751.5002, 4086.8338),  # center=2919.167
    Field("race_gate_slack", "u32", 0, 1),  # center=1
    Field("outing_base", "f32"),  # preset 固定（GA 未覆盖）
    Field("friend_outing_bonus", "f32", 36.7888, 85.8405),  # center=61.314613
    Field("ramen_pt_weight", "f32", 2.7772, 6.4802),  # center=4.628715
    Field("ramen_effect_weight", "f32"),  # preset 固定（GA 未覆盖）
    Field("ramen_special_cost", "f32", 4.0477, 9.4445),  # center=6.746099
    Field("ramen_stock_cost", "f32", 0.24, 0.56),  # center=0.4
    Field("region_xunlian_weight", "f32", 8.9956, 20.9897),  # center=14.992668
    Field("region_hint_weight", "f32", 12.1432, 28.3342),  # center=20.238716
    Field("region_youqing_weight", "f32", -5.0, 5.0),  # center=0
    Field("region_weak_cover_weight", "f32"),  # preset 固定（GA 未覆盖）
    Field("event_vital_weight", "f32"),  # preset 固定（GA 未覆盖）
    Field("event_motivation_weight", "f32", 19.2729, 44.9701),  # center=32.12148
    Field("event_bad_flag_penalty", "f32"),  # preset 固定（GA 未覆盖）
    Field("early_bond_value", "f32", 4.1069, 9.5828),  # center=6.8448305
    Field("hint_bonus", "f32"),  # preset 固定（GA 未覆盖）
    Field("first_friend_click_value", "f32"),  # preset 固定（GA 未覆盖）
    Field("low_friend_bond_value", "f32", 5.0925, 11.8826),  # center=8.487566
    Field("active_friend_value", "f32"),  # preset 固定（GA 未覆盖）
    Field("feeling_overflow_threshold", "i32"),  # preset 固定（GA 未覆盖）
    Field("overflow_value", "f32", 0.8127, 1.8963),  # center=1.3544745
    Field("max_base_score_sacrifice", "f32", 230.7216, 538.3504),  # center=384.53598
    Field("status_reserve_max", "f32", 24.0, 56.0),  # center=40
    Field("dynamic_status_balance", "bool", False),  # GA 定型
    Field("status_gap_strength", "f32", 0.0843, 0.1967),  # center=0.14053416
    Field("status_overflow_strength", "f32"),  # preset 固定（GA 未覆盖）
    Field("dynamic_vital", "bool"),  # preset 固定（GA 未覆盖）
    Field("probabilistic_hint", "bool", False),  # GA 定型
    Field("expected_fail", "bool", False),  # GA 定型
    Field("checkpoint_scale", "f32"),  # preset 固定（GA 未覆盖）
    Field("rmj_cross_bonus", "f32"),  # preset 固定（GA 未覆盖）
    Field("great_cross_bonus", "f32"),  # preset 固定（GA 未覆盖）
    Field("ramen_window_weight", "f32", 0.0481, 0.1123),  # center=0.08024326
    Field("ramen_train_coupling_weight", "f32", 1.8369, 4.286),  # center=3.0614443
    Field("ramen_weak_train_boost", "f32", 0.8001, 1.8669),  # center=1.3334891
    Field("friend_hidden_starve_weight", "f32", 304.1589, 709.7042),  # center=506.93158
    Field("friend_future_hidden_weight", "f32"),  # preset 固定（GA 未覆盖）
    Field("friend_proactive_weight", "f32", 127.8243, 298.2568),  # center=213.04054
    Field("eat_guarantee_weight", "f32", 0.3144, 0.7336),  # center=0.5240228
    Field("cook2_stock_weight", "f32"),  # preset 固定（GA 未覆盖）
    Field("eat_requires_training", "bool"),  # preset 固定（GA 未覆盖）
    Field("eat_requires_covered_train", "bool", True),  # GA 定型
    Field("y3_pre_train_vital_target", "i32"),  # preset 固定（GA 未覆盖）
    Field("y3_post_train_vital_target", "i32", 16, 29),  # center=23
    Field("y3_vital_shortfall_weight", "f32", 0.0506, 0.1181),  # center=0.08438793
    Field("y3_post_train_hard_floor", "i32", 2, 3),  # center=3
    Field("y3_recovery_horizon", "bool", False),  # GA 定型
    Field("friend_outing_replaces_rest", "bool", True),  # GA 定型
    Field("friend_outing3_recovery_vital", "i32", 0, 30),  # center=0
    Field("dynamic_special_targets", "bool", True),  # GA 定型
]

ARRAY_FIELDS: List[Field] = [
    Field("vital_rest_eating", "arri32", [(25, 48), (25, 48), (6, 11)]),
    Field("pt_rate", "arrf32", [(11.2, 20.8), (59.58, 110.649), (44.8, 83.2)]),
    Field("friend_outing_cumulative_caps", "arrusize", [None, None, (3, 6)]),
]

ALL_FIELDS: List[Field] = FREE_FIELDS + ARRAY_FIELDS

def free_fields() -> List[Field]:
    return [f for f in ALL_FIELDS if not f.is_fixed()]

def free_dim() -> int:
    return sum(3 if f.typ.startswith("arr") else 1 for f in free_fields())

if __name__ == "__main__":
    import sys
    n_free = len(free_fields())
    print(f"字段总数 {len(ALL_FIELDS)}，可搜索字段 {n_free}，可搜索维度 {free_dim()}")
    for f in ALL_FIELDS:
        st = "FIXED" if f.is_fixed() else f"{f.lo}..{f.hi}"
        print(f"  {f.name:35s} {f.typ:8s} {st}")
