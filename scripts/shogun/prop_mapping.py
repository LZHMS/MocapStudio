"""Comprehensive prop mapping, theme recognition, and hierarchical matching for 4DHOI.

Provides exact bidirectional mapping between dialogue/script Chinese names,
Shogun mocap skeleton/rigid body prefixes, and CAD asset IDs.
Implements a 3-level hierarchical matcher:
  Level 1: Signature Prop Theme Detection (Orthogonal features for TEA, WRITE, INFO)
  Level 2: Theme-Directed Roster Priority Matching
  Level 3: Global Configuration Fallback (Excluding already-matched items)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# ==============================================================================
# 1. 25 Interactive Props + Global Background Fixtures Complete Mapping
# ==============================================================================

PROP_MAPPING: List[dict] = [
    # ----------------- 信息 (INFO) 主题交互道具 (10件) -----------------
    {
        "cn_name": "蓝色手机",
        "shogun_name": "blue_phone",
        "aliases": ["blue_phone_1", "blue_phone_2", "phone_blue"],
        "asset_id": "phone/phone/phone",
        "cad_glb": "3DObjects/meeting_room/phone/phone/phone.glb",
        "domain": "INFO",
        "is_signature": False,
        "is_global": False,
        "description": "深蓝色大屏智能手机"
    },
    {
        "cn_name": "笔记本电脑",
        "shogun_name": "white_laptop_base",
        "aliases": ["white_laptop_screen", "laptop", "white_laptop"],
        "asset_id": "laptop/laptop_base/laptop_base",
        "cad_glb": "3DObjects/meeting_room/laptop/laptop_base/laptop_base.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "白色便携笔记本电脑 (含底座 base 与屏幕 screen 两部分刚体)"
    },
    {
        "cn_name": "鼠标",
        "shogun_name": "white_mouse",
        "aliases": ["mouse"],
        "asset_id": "mouse/mouse",
        "cad_glb": "3DObjects/meeting_room/mouse/mouse.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "白色光学无线鼠标"
    },
    {
        "cn_name": "U盘",
        "shogun_name": "blue_udrive",
        "aliases": ["blue_u_drive", "u_drive", "usb_drive"],
        "asset_id": "usb_drive/u_drive",
        "cad_glb": "3DObjects/meeting_room/usb_drive/u_drive.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "蓝色便携金属U盘"
    },
    {
        "cn_name": "翻页笔",
        "shogun_name": "white_presentation_clicker",
        "aliases": ["white_clicker", "presentation_clicker", "clicker"],
        "asset_id": "presentation_clicker/presentation_clicker",
        "cad_glb": "3DObjects/meeting_room/presentation_clicker/presentation_clicker.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "白色激光演示翻页笔"
    },
    {
        "cn_name": "设备遥控器",
        "shogun_name": "white_remote_controller",
        "aliases": ["remote_controller", "remote"],
        "asset_id": "remote_controller/remote_controller",
        "cad_glb": "3DObjects/meeting_room/remote_controller/remote_controller.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "白色空调/多媒体设备遥控器"
    },
    {
        "cn_name": "话筒",
        "shogun_name": "white_huatong",
        "aliases": ["huatong", "handheld_microphone", "microphone_hand"],
        "asset_id": "handheld_microphone/microphone/microphone",
        "cad_glb": "3DObjects/meeting_room/handheld_microphone/microphone/microphone.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "手持无线动圈话筒 (与支架配套)"
    },
    {
        "cn_name": "话筒支架",
        "shogun_name": "white_support",
        "aliases": ["microphone_base", "support", "huatong_support"],
        "asset_id": "handheld_microphone/microphone_base/microphone_base",
        "cad_glb": "3DObjects/meeting_room/handheld_microphone/microphone_base/microphone_base.glb",
        "domain": "INFO",
        "is_signature": True,
        "is_global": False,
        "description": "话筒斜插底座/桌面支撑支架"
    },
    {
        "cn_name": "白色收纳盒",
        "shogun_name": "white_box",
        "aliases": ["storage_box_large", "white_storage_box"],
        "asset_id": "storage_box/storage_box_large/storage_box_large",
        "cad_glb": "3DObjects/meeting_room/storage_box/storage_box_large/storage_box_large.glb",
        "domain": "INFO",
        "is_signature": False,
        "is_global": False,
        "description": "白色大容量带盖杂物收纳盒"
    },
    {
        "cn_name": "马克杯",
        "shogun_name": "white_mug_cup",
        "aliases": ["white_mugcup", "mug_cup", "mug"],
        "asset_id": "cup/mug_cup/mug_cup",
        "cad_glb": "3DObjects/meeting_room/cup/mug_cup/mug_cup.glb",
        "domain": "INFO",
        "is_signature": False,
        "is_global": False,
        "description": "白色经典带耳陶瓷马克水杯"
    },

    # ----------------- 茶水 (TEA) 主题交互道具 (7件) -----------------
    {
        "cn_name": "托盘",
        "shogun_name": "blue_tray",
        "aliases": ["tray", "blue_tea_tray"],
        "asset_id": "tray/tray",
        "cad_glb": "3DObjects/meeting_room/tray/tray.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "蓝色长方形防滑茶水服务托盘"
    },
    {
        "cn_name": "茶壶",
        "shogun_name": "white_teaport",
        "aliases": ["white_teapot", "teapot", "tea_pot"],
        "asset_id": "teapot/teapot",
        "cad_glb": "3DObjects/meeting_room/teapot/teapot.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "白色耐热提梁冲泡茶壶"
    },
    {
        "cn_name": "水杯",
        "shogun_name": "white_cup",
        "aliases": ["water_cup"],
        "asset_id": "cup/water_cup/water_cup",
        "cad_glb": "3DObjects/meeting_room/cup/water_cup/water_cup.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "白色一次性双层隔热水杯"
    },
    {
        "cn_name": "矿泉水瓶",
        "shogun_name": "white_bottled_water",
        "aliases": ["high_bottled_water", "water_bottle"],
        "asset_id": "high_bottled_water/high_bottled_water",
        "cad_glb": "3DObjects/meeting_room/high_bottled_water/high_bottled_water.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "透明瓶身白色瓶盖高瓶身饮用纯净水"
    },
    {
        "cn_name": "饮料瓶",
        "shogun_name": "white_drink",
        "aliases": ["drink_bottle", "beverage_bottle"],
        "asset_id": "drink_bottle/drink_bottle",
        "cad_glb": "3DObjects/meeting_room/drink_bottle/drink_bottle.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "圆柱形茶饮/果汁饮料瓶"
    },
    {
        "cn_name": "易拉罐",
        "shogun_name": "white_can",
        "aliases": ["coke_can", "coke_can_half_open", "can"],
        "asset_id": "coke_can/coke_can_half_open",
        "cad_glb": "3DObjects/meeting_room/coke_can/coke_can_half_open.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "铝合金半开拉环易拉饮料罐"
    },
    {
        "cn_name": "垃圾桶",
        "shogun_name": "white_trash",
        "aliases": ["circle_trash", "trash_can"],
        "asset_id": "circle_trash/circle_trash",
        "cad_glb": "3DObjects/meeting_room/circle_trash/circle_trash.glb",
        "domain": "TEA",
        "is_signature": True,
        "is_global": False,
        "description": "圆形白色塑料会议室废纸篓"
    },

    # ----------------- 书写 (WRITE) 主题交互道具 (8件) -----------------
    {
        "cn_name": "书写笔记本",
        "shogun_name": "white_handbook_base",
        "aliases": ["white_handbook_cover", "files_book", "notebook"],
        "asset_id": "files_book/files_book_base/files_book_base",
        "cad_glb": "3DObjects/meeting_room/files_book/files_book_base/files_book_base.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "精装皮质会议纪要本 (含基座与封面两部分刚体)"
    },
    {
        "cn_name": "写字板",
        "shogun_name": "white_writing_board",
        "aliases": ["writing_board", "clipboard"],
        "asset_id": "writing_board/writing_board",
        "cad_glb": "3DObjects/meeting_room/writing_board/writing_board.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "白色便携板夹/会签硬垫板"
    },
    {
        "cn_name": "签字笔",
        "shogun_name": "white_sign_pen",
        "aliases": ["dark_sign_pen", "signing_pen", "sign_pen"],
        "asset_id": "signing_pen/signing_pen",
        "cad_glb": "3DObjects/meeting_room/signing_pen/signing_pen.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "中性黑色墨水商务签字笔"
    },
    {
        "cn_name": "笔筒",
        "shogun_name": "white_pen_box",
        "aliases": ["cat_pen_box", "pen_box", "pen_holder"],
        "asset_id": "cat_pen_box/cat_pen_box",
        "cad_glb": "3DObjects/meeting_room/cat_pen_box/cat_pen_box.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "白色猫耳造型案头多孔笔筒"
    },
    {
        "cn_name": "白色手机",
        "shogun_name": "white_phone",
        "aliases": ["phone_small", "white_phone_small"],
        "asset_id": "phone/phone_small/phone_small",
        "cad_glb": "3DObjects/meeting_room/phone/phone_small/phone_small.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "白色窄边框轻薄智能手机"
    },
    {
        "cn_name": "红色收纳盒",
        "shogun_name": "red_box",
        "aliases": ["storage_box", "red_storage_box"],
        "asset_id": "storage_box/storage_box/storage_box",
        "cad_glb": "3DObjects/meeting_room/storage_box/storage_box/storage_box.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "红色长方形文具整理小收纳盒"
    },
    {
        "cn_name": "麦克风",
        "shogun_name": "white_microphone",
        "aliases": ["gooseneck_microphone", "microphone_gooseneck"],
        "asset_id": "gooseneck_microphone/gooseneck_microphone",
        "cad_glb": "3DObjects/meeting_room/gooseneck_microphone/gooseneck_microphone.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "鹅颈式桌面发言拾音麦克风"
    },
    {
        "cn_name": "波纹杯",
        "shogun_name": "white_wavy_cup",
        "aliases": ["wavy_cup", "lan_cup"],
        "asset_id": "cup/wavy_cup/wavy_cup",
        "cad_glb": "3DObjects/meeting_room/cup/wavy_cup/wavy_cup.glb",
        "domain": "WRITE",
        "is_signature": True,
        "is_global": False,
        "description": "外壁带防烫防滑波纹的白色咖啡/冲饮水杯"
    },

    # ----------------- 全局常驻环境背景 (GLOBAL) (8件) -----------------
    {
        "cn_name": "内门把手",
        "shogun_name": "door_in",
        "aliases": ["door_hand", "door_inside"],
        "asset_id": "door_hand/door_hand",
        "cad_glb": "3DObjects/meeting_room/door_hand/door_hand.glb",
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "会议室木门室内推拉把手"
    },
    {
        "cn_name": "外门把手",
        "shogun_name": "door_out",
        "aliases": ["door_hand_outer", "door_outside"],
        "asset_id": "door_hand/door_hand_outer",
        "cad_glb": "3DObjects/meeting_room/door_hand/door_hand_outer.glb",
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "会议室木门走廊外侧推拉把手"
    },
    {
        "cn_name": "监控相机1",
        "shogun_name": "camera_1",
        "aliases": ["pocket_camera_1"],
        "asset_id": "pocket_camera/pocket_camera_1",
        "cad_glb": "3DObjects/pocket_camera/pocket_camera_1.glb",
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "动捕场地 1 号定点监控机位"
    },
    {
        "cn_name": "监控相机2",
        "shogun_name": "camera_2",
        "aliases": ["pocket_camera_2"],
        "asset_id": "pocket_camera/pocket_camera_2",
        "cad_glb": "3DObjects/pocket_camera/pocket_camera_2.glb",
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "动捕场地 2 号定点监控机位"
    },
    {
        "cn_name": "监控相机3",
        "shogun_name": "camera_3",
        "aliases": ["pocket_camera_3"],
        "asset_id": "pocket_camera/pocket_camera_3",
        "cad_glb": "3DObjects/pocket_camera/pocket_camera_3.glb",
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "动捕场地 3 号定点监控机位"
    },
    {
        "cn_name": "会议长桌",
        "shogun_name": "desk",
        "aliases": ["table", "conference_table"],
        "asset_id": "meeting_room_environment/table",
        "cad_glb": None,
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "会议室中央实木长会议桌"
    },
    {
        "cn_name": "橙色椅子",
        "shogun_name": "orange_chair_1",
        "aliases": ["orange_chair"],
        "asset_id": "meeting_room_environment/orange_chair_1",
        "cad_glb": None,
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "演员1专用橙色靠背办公椅"
    },
    {
        "cn_name": "白色椅子",
        "shogun_name": "white_chair_2",
        "aliases": ["white_chair_1", "chair"],
        "asset_id": "meeting_room_environment/white_chair_2",
        "cad_glb": None,
        "domain": "GLOBAL",
        "is_signature": False,
        "is_global": True,
        "description": "演员2专用白色靠背办公椅"
    },
]


# ==============================================================================
# 2. Lookup Indexes & Helper Dictionaries
# ==============================================================================

CN_TO_ITEM: Dict[str, dict] = {item["cn_name"]: item for item in PROP_MAPPING}
SHOGUN_TO_ITEM: Dict[str, dict] = {}

for item in PROP_MAPPING:
    SHOGUN_TO_ITEM[item["shogun_name"]] = item
    for alias in item["aliases"]:
        SHOGUN_TO_ITEM[alias] = item

GLOBAL_PROPS: Set[str] = {item["shogun_name"] for item in PROP_MAPPING if item["is_global"]}

THEME_SIGNATURE_PROPS: Dict[str, Set[str]] = {
    "TEA": {item["shogun_name"] for item in PROP_MAPPING if item["domain"] == "TEA" and item["is_signature"]},
    "WRITE": {item["shogun_name"] for item in PROP_MAPPING if item["domain"] == "WRITE" and item["is_signature"]},
    "INFO": {item["shogun_name"] for item in PROP_MAPPING if item["domain"] == "INFO" and item["is_signature"]},
}

# The standard theme pools (all domain items)
THEME_DOMAIN_PROPS: Dict[str, Set[str]] = {
    "TEA": {item["shogun_name"] for item in PROP_MAPPING if item["domain"] == "TEA"},
    "WRITE": {item["shogun_name"] for item in PROP_MAPPING if item["domain"] == "WRITE"},
    "INFO": {item["shogun_name"] for item in PROP_MAPPING if item["domain"] == "INFO"},
}


# ==============================================================================
# 3. Dynamic Hierarchical Matcher
# ==============================================================================

def identify_theme_from_detected_props(detected_props: List[str]) -> Optional[str]:
    """Identify the interaction theme (INFO_X_TEA, WRITE_X_INFO, TEA_X_WRITE) from detected props."""
    scores = {"TEA": 0, "WRITE": 0, "INFO": 0}
    for p in detected_props:
        item = SHOGUN_TO_ITEM.get(p)
        if item:
            domain = item["domain"]
            if domain in scores:
                scores[domain] += 2 if item["is_signature"] else 1

    sorted_domains = sorted(scores.items(), key=lambda x: -x[1])
    top1, score1 = sorted_domains[0]
    top2, score2 = sorted_domains[1]

    if score1 == 0:
        return None

    if score2 == 0:
        return top1

    active_domains = {top1, top2}
    if active_domains == {"INFO", "TEA"}:
        return "INFO_X_TEA"
    elif active_domains == {"WRITE", "INFO"}:
        return "WRITE_X_INFO"
    elif active_domains == {"TEA", "WRITE"}:
        return "TEA_X_WRITE"
    return f"{top1}_X_{top2}"

def build_priority_matching_plan(
    detected_props: List[str],
    take_theme_hint: Optional[str] = None
) -> Tuple[str, List[str], List[str]]:
    """Build a 3-level hierarchical matching plan.

    Returns:
      (resolved_theme, priority_candidate_props, remaining_global_props)

    Guarantees:
      - Already-matched props are excluded before fallback to avoid redundant compute.
      - Global props do NOT include cups or phones.
    """
    # 1. Level 1: Theme determination
    theme = take_theme_hint or identify_theme_from_detected_props(detected_props) or "INFO_X_TEA"

    # 2. Level 2: Assemble theme candidate roster
    theme_upper = theme.upper()
    priority_props: Set[str] = set()
    if "TEA" in theme_upper:
        priority_props.update(THEME_DOMAIN_PROPS["TEA"])
    if "WRITE" in theme_upper:
        priority_props.update(THEME_DOMAIN_PROPS["WRITE"])
    if "INFO" in theme_upper:
        priority_props.update(THEME_DOMAIN_PROPS["INFO"])

    # Expand aliases into the priority roster
    full_priority_roster: List[str] = []
    for p in sorted(priority_props):
        full_priority_roster.append(p)
        item = SHOGUN_TO_ITEM.get(p)
        if item:
            for alias in item["aliases"]:
                if alias not in full_priority_roster:
                    full_priority_roster.append(alias)

    # 3. Level 3: Global fallback pool, strictly excluding already-matched props
    global_fallback_pool = [
        p for p in sorted(GLOBAL_PROPS)
        if p not in priority_props and p not in detected_props
    ]

    return theme, full_priority_roster, global_fallback_pool
