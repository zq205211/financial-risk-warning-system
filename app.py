"""
多智能体AI+大数据因果推演的区域性地方金融隐性风险穿透预警系统
作者：全栈数据开发+金融风控可视化工程师
版本：v1.0.0
说明：本系统采用Streamlit框架开发，支持stlite静态编译部署到GitHub Pages
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import random

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="区域性金融风险穿透预警系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 全局样式配置 - 金融深色风控风格
# ============================================================================
def set_custom_theme():
    """设置自定义深色主题样式"""
    st.markdown("""
    <style>
        /* 全局背景色 */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }

        /* 侧边栏样式 */
        .css-1d391kg {
            background-color: #1a1d24;
        }

        /* 卡片样式 */
        .metric-card {
            background-color: #1a1d24;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* 指标卡片样式 */
        div[data-testid="stMetric"] {
            background-color: #1a1d24;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        /* 标题样式 */
        h1, h2, h3 {
            color: #fafafa !important;
        }

        /* 按钮样式 */
        .stButton>button {
            background-color: #2979ff;
            color: white;
            border-radius: 5px;
        }

        /* 选择框样式 */
        .stSelectbox>div>div {
            background-color: #1a1d24;
            color: #fafafa;
        }

        /* 数据表格样式 */
        .stDataFrame {
            background-color: #1a1d24;
        }

        /* 警告框样式 */
        .stAlert {
            background-color: #1a1d24;
        }

        /* Expander样式 */
        .streamlit-expanderHeader {
            background-color: #1a1d24;
            color: #fafafa;
        }

        /* 隐藏Streamlit默认页脚 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

set_custom_theme()

# ============================================================================
# 配色方案
# ============================================================================
COLORS = {
    "背景色": "#0e1117",
    "卡片背景": "#1a1d24",
    "文字主色": "#fafafa",
    "文字次色": "#a0a0a0",
    "低风险": "#00c853",
    "中风险": "#ffc107",
    "高风险": "#ff5722",
    "重度预警": "#d50000",
    "强调色": "#2979ff"
}

# ============================================================================
# 模拟数据生成函数
# ============================================================================
@st.cache_data
def generate_regional_data():
    """生成区域金融风险模拟数据"""
    np.random.seed(42)

    provinces = ["北京", "上海", "广东", "江苏", "浙江", "山东", "河南", "四川",
                 "湖北", "湖南", "福建", "安徽", "河北", "陕西", "辽宁", "江西",
                 "重庆", "广西", "云南", "贵州", "天津", "山西", "黑龙江", "吉林",
                 "甘肃", "内蒙古", "新疆", "海南", "宁夏", "青海", "西藏"]

    cities = {
        "北京": ["北京"],
        "上海": ["上海"],
        "广东": ["广州", "深圳", "东莞", "佛山", "珠海"],
        "江苏": ["南京", "苏州", "无锡", "常州", "南通"],
        "浙江": ["杭州", "宁波", "温州", "嘉兴", "绍兴"],
        "山东": ["济南", "青岛", "烟台", "潍坊", "临沂"],
        "河南": ["郑州", "洛阳", "开封", "新乡", "安阳"],
        "四川": ["成都", "绵阳", "德阳", "宜宾", "泸州"],
        "湖北": ["武汉", "宜昌", "襄阳", "荆州", "黄冈"],
        "湖南": ["长沙", "株洲", "湘潭", "衡阳", "岳阳"],
        "福建": ["福州", "厦门", "泉州", "漳州", "莆田"],
        "安徽": ["合肥", "芜湖", "蚌埠", "阜阳", "安庆"],
        "河北": ["石家庄", "唐山", "保定", "邯郸", "廊坊"],
        "陕西": ["西安", "宝鸡", "咸阳", "渭南", "延安"],
        "辽宁": ["沈阳", "大连", "鞍山", "抚顺", "锦州"],
        "江西": ["南昌", "九江", "赣州", "景德镇", "萍乡"],
        "重庆": ["重庆"],
        "广西": ["南宁", "柳州", "桂林", "梧州", "北海"],
        "云南": ["昆明", "大理", "丽江", "曲靖", "玉溪"],
        "贵州": ["贵阳", "遵义", "六盘水", "安顺", "毕节"],
        "天津": ["天津"],
        "山西": ["太原", "大同", "阳泉", "长治", "晋城"],
        "黑龙江": ["哈尔滨", "齐齐哈尔", "牡丹江", "佳木斯", "大庆"],
        "吉林": ["长春", "吉林", "四平", "辽源", "通化"],
        "甘肃": ["兰州", "嘉峪关", "金昌", "白银", "天水"],
        "内蒙古": ["呼和浩特", "包头", "乌海", "赤峰", "通辽"],
        "新疆": ["乌鲁木齐", "克拉玛依", "吐鲁番", "哈密"],
        "海南": ["海口", "三亚", "三沙", "儋州"],
        "宁夏": ["银川", "石嘴山", "吴忠", "固原", "中卫"],
        "青海": ["西宁", "海东", "海北", "黄南"],
        "西藏": ["拉萨", "日喀则", "昌都", "林芝"]
    }

    # 中国各省份大致经纬度坐标
    province_coords = {
        "北京": [116.4, 39.9], "上海": [121.5, 31.2], "广东": [113.3, 23.1],
        "江苏": [118.8, 32.1], "浙江": [120.2, 30.3], "山东": [117.0, 36.7],
        "河南": [113.7, 34.8], "四川": [104.1, 30.7], "湖北": [114.3, 30.6],
        "湖南": [113.0, 28.2], "福建": [119.3, 26.1], "安徽": [117.3, 31.9],
        "河北": [114.5, 38.0], "陕西": [108.9, 34.3], "辽宁": [123.4, 41.8],
        "江西": [115.9, 28.7], "重庆": [106.5, 29.6], "广西": [108.3, 22.8],
        "云南": [102.7, 25.0], "贵州": [106.7, 26.6], "天津": [117.2, 39.1],
        "山西": [112.5, 37.9], "黑龙江": [126.6, 45.8], "吉林": [125.3, 43.9],
        "甘肃": [103.8, 36.1], "内蒙古": [111.7, 40.8], "新疆": [87.6, 43.8],
        "海南": [110.3, 20.0], "宁夏": [106.3, 38.5], "青海": [101.8, 36.6],
        "西藏": [91.1, 29.6]
    }

    data = []
    for province in provinces:
        coords = province_coords.get(province, [0, 0])
        province_cities = cities.get(province, [province])

        for city in province_cities:
            # 生成随机风险指标
            risk_score = np.random.uniform(20, 95)
            if risk_score < 40:
                risk_level = "低风险"
                risk_color = COLORS["低风险"]
            elif risk_score < 60:
                risk_level = "中风险"
                risk_color = COLORS["中风险"]
            elif risk_score < 80:
                risk_level = "高风险"
                risk_color = COLORS["高风险"]
            else:
                risk_level = "重度预警"
                risk_color = COLORS["重度预警"]

            data.append({
                "省份": province,
                "城市": city,
                "风险等级": risk_level,
                "风险评分": round(risk_score, 2),
                "GDP增速": round(np.random.uniform(2, 8), 2),
                "债务率": round(np.random.uniform(60, 150), 2),
                "信贷规模": round(np.random.uniform(1000, 50000), 0),
                "不良贷款率": round(np.random.uniform(0.5, 5), 2),
                "经度": coords[0] + np.random.uniform(-2, 2),
                "纬度": coords[1] + np.random.uniform(-2, 2),
                "风险颜色": risk_color
            })

    return pd.DataFrame(data)

@st.cache_data
def generate_agent_data():
    """生成多智能体运算结果数据"""
    np.random.seed(42)

    agents = {
        "宏观经济智能体": {
            "icon": "🏛️",
            "metrics": {
                "GDP增速预测": {"value": 4.8, "change": -0.4, "unit": "%"},
                "通胀风险评估": {"value": 2.3, "level": "中等", "unit": "%"},
                "政策传导效应": {"value": 0.72, "level": "正向", "unit": ""},
                "财政赤字率": {"value": 3.2, "change": 0.3, "unit": "%"},
                "固定资产投资": {"value": 5.6, "change": 0.8, "unit": "%"},
                "消费增长率": {"value": 4.2, "change": -0.2, "unit": "%"}
            },
            "risk_assessment": "中风险",
            "confidence": 0.85
        },
        "信贷风控智能体": {
            "icon": "💳",
            "metrics": {
                "不良贷款率": {"value": 2.3, "change": 0.5, "unit": "%"},
                "信贷违约概率": {"value": 15.7, "level": "中等", "unit": "%"},
                "行业集中度风险": {"value": 0.68, "level": "高", "unit": ""},
                "拨备覆盖率": {"value": 185, "change": -12, "unit": "%"},
                "资本充足率": {"value": 14.2, "change": -0.3, "unit": "%"},
                "流动性比例": {"value": 52, "level": "正常", "unit": "%"}
            },
            "risk_assessment": "高风险",
            "confidence": 0.92
        },
        "非标债务智能体": {
            "icon": "📊",
            "metrics": {
                "非标债务规模": {"value": 3.2, "level": "万亿", "unit": "万亿元"},
                "到期压力指数": {"value": 78.5, "level": "高风险", "unit": ""},
                "期限错配率": {"value": 45, "level": "中等", "unit": "%"},
                "信托规模": {"value": 1.8, "change": -0.2, "unit": "万亿"},
                "委托贷款": {"value": 0.9, "change": -0.1, "unit": "万亿"},
                "融资租赁": {"value": 0.5, "change": 0.05, "unit": "万亿"}
            },
            "risk_assessment": "高风险",
            "confidence": 0.88
        },
        "舆情传导智能体": {
            "icon": "📰",
            "metrics": {
                "负面舆情指数": {"value": 62.3, "level": "中等风险", "unit": ""},
                "传导速度": {"value": 0.75, "level": "快速", "unit": ""},
                "影响范围": {"value": 0.62, "level": "区域性", "unit": ""},
                "舆情热度": {"value": 78, "change": 12, "unit": ""},
                "传播广度": {"value": 45, "level": "中等", "unit": "%"},
                "情绪倾向": {"value": -0.35, "level": "偏负面", "unit": ""}
            },
            "risk_assessment": "中风险",
            "confidence": 0.78
        }
    }

    return agents

@st.cache_data
def generate_causal_graph():
    """生成因果推演图谱数据"""
    np.random.seed(42)

    # 定义节点
    nodes = [
        {"id": "GDP增速下降", "group": "宏观", "weight": 0.9},
        {"id": "财政收入减少", "group": "财政", "weight": 0.85},
        {"id": "债务压力上升", "group": "债务", "weight": 0.95},
        {"id": "信贷收紧", "group": "信贷", "weight": 0.8},
        {"id": "企业违约风险", "group": "企业", "weight": 0.88},
        {"id": "不良贷款上升", "group": "银行", "weight": 0.82},
        {"id": "舆情恶化", "group": "舆情", "weight": 0.75},
        {"id": "资金链断裂", "group": "流动性", "weight": 0.9},
        {"id": "系统性风险", "group": "系统性", "weight": 0.98}
    ]

    # 定义边（因果关系）
    edges = [
        {"source": "GDP增速下降", "target": "财政收入减少", "weight": 0.85, "lag": 3},
        {"source": "GDP增速下降", "target": "信贷收紧", "weight": 0.7, "lag": 6},
        {"source": "财政收入减少", "target": "债务压力上升", "weight": 0.9, "lag": 2},
        {"source": "债务压力上升", "target": "信贷收紧", "weight": 0.75, "lag": 1},
        {"source": "信贷收紧", "target": "企业违约风险", "weight": 0.88, "lag": 4},
        {"source": "企业违约风险", "target": "不良贷款上升", "weight": 0.92, "lag": 2},
        {"source": "企业违约风险", "target": "舆情恶化", "weight": 0.65, "lag": 1},
        {"source": "舆情恶化", "target": "资金链断裂", "weight": 0.7, "lag": 3},
        {"source": "资金链断裂", "target": "系统性风险", "weight": 0.95, "lag": 2},
        {"source": "不良贷款上升", "target": "系统性风险", "weight": 0.85, "lag": 4}
    ]

    return nodes, edges

@st.cache_data
def generate_hidden_risk_data():
    """生成隐性风险穿透明细数据"""
    np.random.seed(42)

    risk_types = ["地方城投", "小微企业", "居民债务", "影子金融"]
    regions = ["北京", "上海", "广东", "江苏", "浙江", "山东", "河南", "四川",
               "湖北", "湖南", "福建", "安徽", "河北", "陕西", "辽宁", "江西"]

    entities = {
        "地方城投": ["XX城投集团", "XX建设投资", "XX城市发展", "XX交通投资", "XX水务集团"],
        "小微企业": ["XX科技公司", "XX贸易公司", "XX制造企业", "XX服务公司", "XX零售企业"],
        "居民债务": ["XX地区居民", "XX市居民", "XX县居民", "XX区居民", "XX镇居民"],
        "影子金融": ["XX信托计划", "XX资管产品", "XX私募基金", "XX担保公司", "XX小贷公司"]
    }

    main_risks = {
        "地方城投": ["债务逾期", "融资困难", "项目停工", "资金链紧张", "信用评级下调"],
        "小微企业": ["资金链紧张", "订单减少", "成本上升", "融资困难", "经营困难"],
        "居民债务": ["杠杆率上升", "收入下降", "还款压力", "消费降级", "信用违约"],
        "影子金融": ["兑付危机", "监管收紧", "资金池风险", "期限错配", "信用风险"]
    }

    data = []
    for _ in range(200):
        risk_type = np.random.choice(risk_types)
        region = np.random.choice(regions)

        risk_score = np.random.uniform(20, 98)
        if risk_score < 40:
            risk_level = "低风险"
            status = "正常"
        elif risk_score < 60:
            risk_level = "中风险"
            status = "关注"
        elif risk_score < 80:
            risk_level = "高风险"
            status = "预警"
        else:
            risk_level = "重度预警"
            status = "处置"

        data.append({
            "风险类型": risk_type,
            "主体名称": np.random.choice(entities[risk_type]),
            "所属区域": region,
            "风险等级": risk_level,
            "风险评分": round(risk_score, 2),
            "主要风险": np.random.choice(main_risks[risk_type]),
            "状态": status,
            "更新时间": (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime("%Y-%m-%d"),
            "涉及金额": round(np.random.uniform(100, 50000), 2),
            "影响人数": np.random.randint(100, 100000)
        })

    return pd.DataFrame(data)

@st.cache_data
def generate_time_series_data():
    """生成时间序列数据用于趋势分析"""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", end="2024-12-31", freq="M")

    data = {
        "日期": dates,
        "GDP增速": np.random.normal(5, 0.5, len(dates)).cumsum() / 10 + 5,
        "债务率": np.random.normal(80, 5, len(dates)).cumsum() / 10 + 80,
        "不良贷款率": np.random.normal(2, 0.3, len(dates)).cumsum() / 20 + 2,
        "风险指数": np.random.normal(50, 5, len(dates)).cumsum() / 5 + 50,
        "舆情指数": np.random.normal(40, 8, len(dates)).cumsum() / 8 + 40
    }

    df = pd.DataFrame(data)

    # 确保数据在合理范围内
    df["GDP增速"] = df["GDP增速"].clip(2, 8)
    df["债务率"] = df["债务率"].clip(50, 150)
    df["不良贷款率"] = df["不良贷款率"].clip(0.5, 5)
    df["风险指数"] = df["风险指数"].clip(0, 100)
    df["舆情指数"] = df["舆情指数"].clip(0, 100)

    return df

@st.cache_data
def generate_forecast_data():
    """生成未来风险预测数据"""
    np.random.seed(42)

    # 历史数据
    hist_dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="M")
    hist_values = np.random.normal(50, 5, len(hist_dates)).cumsum() / 10 + 50

    # 预测数据
    forecast_dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="M")
    forecast_values = hist_values[-1] + np.random.normal(2, 3, len(forecast_dates)).cumsum() / 5

    # 置信区间
    upper_bound = forecast_values + np.random.uniform(5, 10, len(forecast_dates))
    lower_bound = forecast_values - np.random.uniform(5, 10, len(forecast_dates))

    return {
        "历史日期": hist_dates,
        "历史值": hist_values,
        "预测日期": forecast_dates,
        "预测值": forecast_values,
        "上界": upper_bound,
        "下界": lower_bound
    }

# ============================================================================
# 可视化函数
# ============================================================================
def plot_risk_heatmap(df):
    """绘制风险热力地图"""
    fig = go.Figure()

    # 根据风险等级设置颜色
    color_map = {
        "低风险": COLORS["低风险"],
        "中风险": COLORS["中风险"],
        "高风险": COLORS["高风险"],
        "重度预警": COLORS["重度预警"]
    }

    for risk_level in ["低风险", "中风险", "高风险", "重度预警"]:
        df_level = df[df["风险等级"] == risk_level]
        fig.add_trace(go.Scattergeo(
            lon=df_level["经度"],
            lat=df_level["纬度"],
            text=df_level["城市"] + "<br>风险评分: " + df_level["风险评分"].astype(str),
            mode="markers",
            marker=dict(
                size=df_level["风险评分"] / 3,
                color=color_map[risk_level],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            name=risk_level,
            hovertemplate="<b>%{text}</b><extra></extra>"
        ))

    fig.update_layout(
        title=dict(text="区域性金融风险热力地图", font=dict(size=20, color=COLORS["文字主色"])),
        geo=dict(
            scope="asia",
            showland=True,
            landcolor="#2a2d35",
            countrycolor="#4a4d55",
            coastlinecolor="#4a4d55",
            showlakes=True,
            lakecolor="#1a1d24",
            showocean=True,
            oceancolor="#0e1117",
            center=dict(lat=35, lon=105),
            projection_scale=1.2
        ),
        paper_bgcolor=COLORS["背景色"],
        plot_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=COLORS["文字主色"])
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig

def plot_risk_level_gauge(risk_level):
    """绘制风险等级仪表盘"""
    color_map = {
        "低风险": COLORS["低风险"],
        "中风险": COLORS["中风险"],
        "高风险": COLORS["高风险"],
        "重度预警": COLORS["重度预警"]
    }

    value_map = {"低风险": 25, "中风险": 50, "高风险": 75, "重度预警": 95}

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_map.get(risk_level, 50),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"全域风险等级: {risk_level}", 'font': {'size': 16, 'color': COLORS["文字主色"]}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS["文字主色"]},
            'bar': {'color': color_map.get(risk_level, COLORS["中风险"])},
            'bgcolor': COLORS["卡片背景"],
            'borderwidth': 2,
            'bordercolor': COLORS["文字次色"],
            'steps': [
                {'range': [0, 40], 'color': COLORS["低风险"]},
                {'range': [40, 60], 'color': COLORS["中风险"]},
                {'range': [60, 80], 'color': COLORS["高风险"]},
                {'range': [80, 100], 'color': COLORS["重度预警"]}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value_map.get(risk_level, 50)
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["背景色"],
        font={'color': COLORS["文字主色"], 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def plot_agent_radar(agent_name, metrics):
    """绘制智能体雷达图"""
    categories = list(metrics.keys())
    values = [m["value"] if isinstance(m["value"], (int, float)) else 50 for m in metrics.values()]

    # 归一化到0-100
    max_vals = {"GDP增速预测": 10, "通胀风险评估": 10, "政策传导效应": 1, "财政赤字率": 10,
                "固定资产投资": 15, "消费增长率": 10, "不良贷款率": 10, "信贷违约概率": 100,
                "行业集中度风险": 1, "拨备覆盖率": 300, "资本充足率": 20, "流动性比例": 100,
                "非标债务规模": 10, "到期压力指数": 100, "期限错配率": 100, "信托规模": 5,
                "委托贷款": 5, "融资租赁": 5, "负面舆情指数": 100, "传导速度": 1,
                "影响范围": 1, "舆情热度": 100, "传播广度": 100, "情绪倾向": 1}

    normalized_values = []
    for cat, val in zip(categories, values):
        max_val = max_vals.get(cat, 100)
        normalized_values.append(min(100, (abs(val) / max_val) * 100))

    # 闭合雷达图
    normalized_values.append(normalized_values[0])
    categories.append(categories[0])

    fig = go.Figure(data=go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        line=dict(color=COLORS["强调色"], width=2),
        fillcolor='rgba(41, 121, 255, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color=COLORS["文字次色"], size=10),
                gridcolor=COLORS["文字次色"]
            ),
            angularaxis=dict(
                tickfont=dict(color=COLORS["文字主色"], size=10),
                gridcolor=COLORS["文字次色"]
            ),
            bgcolor=COLORS["背景色"]
        ),
        showlegend=False,
        paper_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        height=350,
        margin=dict(l=50, r=50, t=30, b=30)
    )

    return fig

def plot_agent_bar_chart(agent_name, metrics):
    """绘制智能体柱状图"""
    categories = list(metrics.keys())[:6]
    values = [m["value"] if isinstance(m["value"], (int, float)) else 50 for m in list(metrics.values())[:6]]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=values,
                colorscale=[[0, COLORS["低风险"]], [0.5, COLORS["中风险"]], [1, COLORS["高风险"]]],
                showscale=False
            ),
            text=values,
            textposition='outside',
            textfont=dict(color=COLORS["文字主色"])
        )
    ])

    fig.update_layout(
        xaxis=dict(tickfont=dict(color=COLORS["文字主色"], size=10), gridcolor=COLORS["文字次色"]),
        yaxis=dict(tickfont=dict(color=COLORS["文字主色"]), gridcolor=COLORS["文字次色"]),
        paper_bgcolor=COLORS["背景色"],
        plot_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        height=300,
        margin=dict(l=40, r=20, t=30, b=60),
        showlegend=False
    )

    return fig

def plot_causal_network(nodes, edges):
    """绘制因果推演网络图"""
    G = nx.DiGraph()

    # 添加节点
    for node in nodes:
        G.add_node(node["id"], group=node["group"], weight=node["weight"])

    # 添加边
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"], lag=edge["lag"])

    # 使用布局算法
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 创建边的trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color='#4a4d55'),
        hoverinfo='none',
        mode='lines'
    )

    # 创建节点的trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    group_colors = {
        "宏观": COLORS["强调色"],
        "财政": COLORS["低风险"],
        "债务": COLORS["中风险"],
        "信贷": COLORS["高风险"],
        "企业": COLORS["重度预警"],
        "银行": "#9c27b0",
        "舆情": "#ff9800",
        "流动性": "#00bcd4",
        "系统性": "#e91e63"
    }

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_data = next((n for n in nodes if n["id"] == node), None)
        if node_data:
            node_color.append(group_colors.get(node_data["group"], COLORS["强调色"]))
            node_size.append(node_data["weight"] * 40)
        else:
            node_color.append(COLORS["强调色"])
            node_size.append(30)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=11, color=COLORS["文字主色"]),
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=2, color='white'),
            opacity=0.9
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=dict(text="风险传导因果图谱", font=dict(size=18, color=COLORS["文字主色"])),
        showlegend=False,
        paper_bgcolor=COLORS["背景色"],
        plot_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig

def plot_waterfall_chart():
    """绘制指标冲击归因瀑布图"""
    categories = ["GDP增速下降", "财政收入减少", "债务压力上升", "信贷收紧",
                  "企业违约风险", "不良贷款上升", "舆情恶化", "系统性风险"]

    values = [5, 8, 12, 7, 15, 10, 6, -20]

    fig = go.Figure(go.Waterfall(
        name="风险归因",
        orientation="v",
        measure=["relative", "relative", "relative", "relative",
                 "relative", "relative", "relative", "total"],
        x=categories,
        textposition="outside",
        text=[str(v) for v in values],
        y=values,
        connector={"line": {"color": COLORS["文字次色"]}},
        decreasing={"marker": {"color": COLORS["低风险"]}},
        increasing={"marker": {"color": COLORS["高风险"]}},
        totals={"marker": {"color": COLORS["重度预警"]}}
    ))

    fig.update_layout(
        title=dict(text="风险指标冲击归因分析", font=dict(size=18, color=COLORS["文字主色"])),
        xaxis=dict(tickfont=dict(color=COLORS["文字主色"], size=10), gridcolor=COLORS["文字次色"]),
        yaxis=dict(tickfont=dict(color=COLORS["文字主色"]]), gridcolor=COLORS["文字次色"]),
        paper_bgcolor=COLORS["背景色"],
        plot_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        height=400,
        margin=dict(l=40, r=20, t=60, b=80),
        showlegend=False
    )

    return fig

def plot_forecast_chart(forecast_data):
    """绘制滞后性风险推演折线图"""
    fig = go.Figure()

    # 历史数据
    fig.add_trace(go.Scatter(
        x=forecast_data["历史日期"],
        y=forecast_data["历史值"],
        mode='lines',
        name='历史数据',
        line=dict(color=COLORS["强调色"], width=2)
    ))

    # 预测数据
    fig.add_trace(go.Scatter(
        x=forecast_data["预测日期"],
        y=forecast_data["预测值"],
        mode='lines',
        name='预测趋势',
        line=dict(color=COLORS["高风险"], width=2, dash='dash')
    ))

    # 置信区间
    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_data["预测日期"], forecast_data["预测日期"][::-1]]),
        y=np.concatenate([forecast_data["上界"], forecast_data["下界"][::-1]]),
        fill='toself',
        fillcolor='rgba(255, 87, 34, 0.2)',
        line=dict(color='rgba(255, 87, 34, 0)'),
        showlegend=True,
        name='置信区间'
    ))

    fig.update_layout(
        title=dict(text="滞后性风险推演预测", font=dict(size=18, color=COLORS["文字主色"])),
        xaxis=dict(tickfont=dict(color=COLORS["文字主色"]), gridcolor=COLORS["文字次色"]),
        yaxis=dict(tickfont=dict(color=COLORS["文字主色"]]), gridcolor=COLORS["文字次色"], title="风险指数"),
        paper_bgcolor=COLORS["背景色"],
        plot_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                   font=dict(color=COLORS["文字主色"]))
    )

    return fig

def plot_trend_chart(df, selected_metric):
    """绘制趋势折线图"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["日期"],
        y=df[selected_metric],
        mode='lines+markers',
        name=selected_metric,
        line=dict(color=COLORS["强调色"], width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=dict(text=f"{selected_metric}趋势分析", font=dict(size=18, color=COLORS["文字主色"])),
        xaxis=dict(tickfont=dict(color=COLORS["文字主色"]), gridcolor=COLORS["文字次色"]),
        yaxis=dict(tickfont=dict(color=COLORS["文字主色"]]), gridcolor=COLORS["文字次色"]),
        paper_bgcolor=COLORS["背景色"],
        plot_bgcolor=COLORS["背景色"],
        font=dict(color=COLORS["文字主色"]),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode='x unified'
    )

    return fig

# ============================================================================
# 页面模块
# ============================================================================
def page_overview():
    """模块1：区域金融总览大盘"""
    st.title("📊 区域金融风险总览大盘")
    st.markdown("---")

    # 加载数据
    df = generate_regional_data()

    # 顶部指标卡片
    st.markdown("### 🎯 全域风险等级看板")

    risk_counts = df["风险等级"].value_counts()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="低风险区域",
            value=f"{risk_counts.get('低风险', 0)}个",
            delta="较上月 -2",
            delta_color="normal"
        )

    with col2:
        st.metric(
            label="中风险区域",
            value=f"{risk_counts.get('中风险', 0)}个",
            delta="较上月 +3",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="高风险区域",
            value=f"{risk_counts.get('高风险', 0)}个",
            delta="较上月 +1",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="重度预警区域",
            value=f"{risk_counts.get('重度预警', 0)}个",
            delta="较上月 +1",
            delta_color="inverse"
        )

    st.markdown("---")

    # 筛选器
    col_filter1, col_filter2 = st.columns([1, 1])

    with col_filter1:
        selected_province = st.selectbox(
            "选择省份",
            options=["全部"] + list(df["省份"].unique()),
            index=0
        )

    with col_filter2:
        selected_risk = st.selectbox(
            "选择风险等级",
            options=["全部", "低风险", "中风险", "高风险", "重度预警"],
            index=0
        )

    # 数据筛选
    df_filtered = df.copy()
    if selected_province != "全部":
        df_filtered = df_filtered[df_filtered["省份"] == selected_province]
    if selected_risk != "全部":
        df_filtered = df_filtered[df_filtered["风险等级"] == selected_risk]

    # 风险热力地图
    st.markdown("### 🗺️ 分省市风险热力图")
    st.plotly_chart(plot_risk_heatmap(df_filtered), use_container_width=True)

    st.markdown("---")

    # 全域风险等级仪表盘
    col_gauge, col_table = st.columns([1, 1])

    with col_gauge:
        st.markdown("### 📈 全域风险等级")
        # 计算平均风险评分
        avg_risk_score = df_filtered["风险评分"].mean()
        if avg_risk_score < 40:
            overall_risk = "低风险"
        elif avg_risk_score < 60:
            overall_risk = "中风险"
        elif avg_risk_score < 80:
            overall_risk = "高风险"
        else:
            overall_risk = "重度预警"

        st.plotly_chart(plot_risk_level_gauge(overall_risk), use_container_width=True)

    with col_table:
        st.markdown("### 📋 区域风险排名 (Top 10)")
        df_top10 = df_filtered.nlargest(10, "风险评分")[["省份", "城市", "风险等级", "风险评分"]]
        st.dataframe(
            df_top10.style.background_gradient(cmap="RdYlGn_r", subset=["风险评分"]),
            use_container_width=True,
            height=300
        )

    st.markdown("---")

    # 关键指标趋势
    st.markdown("### 📊 关键指标趋势分析")

    trend_df = generate_time_series_data()

    col_trend1, col_trend2 = st.columns([1, 1])

    with col_trend1:
        selected_metric1 = st.selectbox(
            "选择指标1",
            options=["GDP增速", "债务率", "不良贷款率", "风险指数", "舆情指数"],
            index=0,
            key="metric1"
        )
        st.plotly_chart(plot_trend_chart(trend_df, selected_metric1), use_container_width=True)

    with col_trend2:
        selected_metric2 = st.selectbox(
            "选择指标2",
            options=["GDP增速", "债务率", "不良贷款率", "风险指数", "舆情指数"],
            index=1,
            key="metric2"
        )
        st.plotly_chart(plot_trend_chart(trend_df, selected_metric2), use_container_width=True)

def page_agents():
    """模块2：多智能体分工拆解面板"""
    st.title("🤖 多智能体协同运算结果")
    st.markdown("---")

    # 加载数据
    agents_data = generate_agent_data()

    # 智能体概览
    st.markdown("### 🎯 智能体协同概览")

    col_a1, col_a2, col_a3, col_a4 = st.columns(4)

    agent_icons = ["🏛️", "💳", "📊", "📰"]
    agent_names = list(agents_data.keys())

    for i, (col, name, icon) in enumerate(zip([col_a1, col_a2, col_a3, col_a4], agent_names, agent_icons)):
        with col:
            agent = agents_data[name]
            st.markdown(f"""
            <div style='background-color: {COLORS["卡片背景"]}; padding: 15px; border-radius: 10px; text-align: center;'>
                <h3>{icon} {name.replace('智能体', '')}</h3>
                <p style='color: {COLORS["文字次色"]};'>风险评估: <strong style='color: {COLORS["高风险"] if agent["risk_assessment"] == "高风险" else COLORS["中风险"]};'>{agent["risk_assessment"]}</strong></p>
                <p style='color: {COLORS["文字次色"]};'>置信度: <strong>{agent["confidence"]*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 各智能体详细分析
    for agent_name, agent_data in agents_data.items():
        with st.expander(f"{agent_data['icon']} {agent_name} - 详细分析", expanded=True):
            col_metrics, col_chart = st.columns([1, 1])

            with col_metrics:
                st.markdown("#### 📊 核心指标")
                for metric_name, metric_data in agent_data["metrics"].items():
                    value = metric_data["value"]
                    unit = metric_data.get("unit", "")
                    change = metric_data.get("change", None)
                    level = metric_data.get("level", None)

                    # 格式化显示
                    if isinstance(value, float):
                        if abs(value) < 1:
                            display_value = f"{value:.2f}{unit}"
                        else:
                            display_value = f"{value:.1f}{unit}"
                    else:
                        display_value = f"{value}{unit}"

                    # 变化显示
                    if change is not None:
                        delta = f"{change:+.2f}" if isinstance(change, float) else f"{change:+}"
                        st.metric(label=metric_name, value=display_value, delta=delta)
                    elif level is not None:
                        st.metric(label=metric_name, value=display_value, delta=level)
                    else:
                        st.metric(label=metric_name, value=display_value)

            with col_chart:
                st.markdown("#### 🕸️ 多维评估雷达图")
                st.plotly_chart(plot_agent_radar(agent_name, agent_data["metrics"]), use_container_width=True)

        st.markdown("")

def page_causal():
    """模块3：因果推演链路可视化"""
    st.title("🔗 风险传导因果推演")
    st.markdown("---")

    # 加载数据
    nodes, edges = generate_causal_graph()

    # 因果图谱
    st.markdown("### 🕸️ 风险传导因果图谱")
    st.markdown("""
    **图谱说明：**
    - 节点大小表示风险影响权重
    - 节点颜色表示风险类型分组
    - 连线表示风险传导路径
    """)

    st.plotly_chart(plot_causal_network(nodes, edges), use_container_width=True)

    st.markdown("---")

    # 传导路径详情
    st.markdown("### 📋 风险传导路径明细")

    df_edges = pd.DataFrame(edges)
    df_edges.columns = ["风险源头", "风险目标", "传导强度", "滞后时间(月)"]
    df_edges["传导强度"] = df_edges["传导强度"].round(2)

    st.dataframe(
        df_edges.style.background_gradient(cmap="RdYlGn_r", subset=["传导强度"]),
        use_container_width=True
    )

    st.markdown("---")

    # 指标冲击归因
    st.markdown("### 📊 指标冲击归因分析")

    col_attr1, col_attr2 = st.columns([1, 1])

    with col_attr1:
        selected_factor = st.selectbox(
            "选择分析视角",
            options=["系统性风险归因", "信贷风险归因", "债务风险归因", "舆情风险归因"],
            index=0
        )

    st.plotly_chart(plot_waterfall_chart(), use_container_width=True)

    st.markdown("---")

    # 滞后性风险推演
    st.markdown("### 📈 滞后性风险推演预测")

    forecast_data = generate_forecast_data()

    col_forecast1, col_forecast2 = st.columns([2, 1])

    with col_forecast1:
        st.plotly_chart(plot_forecast_chart(forecast_data), use_container_width=True)

    with col_forecast2:
        st.markdown("#### 🎯 预测参数设置")
        forecast_months = st.slider("预测时长(月)", min_value=3, max_value=24, value=12)
        confidence_level = st.slider("置信区间(%)", min_value=80, max_value=99, value=95)

        st.markdown("#### 📊 预测结果摘要")
        st.metric(
            label="预测风险指数",
            value=f"{forecast_data['预测值'][-1]:.1f}",
            delta=f"{forecast_data['预测值'][-1] - forecast_data['历史值'][-1]:.1f}"
        )
        st.metric(
            label="置信区间",
            value=f"[{forecast_data['下界'][-1]:.1f}, {forecast_data['上界'][-1]:.1f}]"
        )

def page_hidden_risks():
    """模块4：隐性风险穿透明细"""
    st.title("🔍 隐性风险穿透数据明细")
    st.markdown("---")

    # 加载数据
    df = generate_hidden_risk_data()

    # 筛选器
    st.markdown("### 🎯 数据筛选")

    col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])

    with col_f1:
        selected_type = st.selectbox(
            "风险类型",
            options=["全部"] + list(df["风险类型"].unique()),
            index=0
        )

    with col_f2:
        selected_region = st.selectbox(
            "所属区域",
            options=["全部"] + list(df["所属区域"].unique()),
            index=0
        )

    with col_f3:
        selected_level = st.selectbox(
            "风险等级",
            options=["全部", "低风险", "中风险", "高风险", "重度预警"],
            index=0
        )

    with col_f4:
        selected_status = st.selectbox(
            "状态",
            options=["全部", "正常", "关注", "预警", "处置"],
            index=0
        )

    # 数据筛选
    df_filtered = df.copy()

    if selected_type != "全部":
        df_filtered = df_filtered[df_filtered["风险类型"] == selected_type]
    if selected_region != "全部":
        df_filtered = df_filtered[df_filtered["所属区域"] == selected_region]
    if selected_level != "全部":
        df_filtered = df_filtered[df_filtered["风险等级"] == selected_level]
    if selected_status != "全部":
        df_filtered = df_filtered[df_filtered["状态"] == selected_status]

    st.markdown("---")

    # 统计概览
    st.markdown("### 📊 风险统计概览")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric("总记录数", f"{len(df_filtered)}条")

    with col_s2:
        st.metric("涉及金额", f"{df_filtered['涉及金额'].sum():.2f}亿元")

    with col_s3:
        st.metric("影响人数", f"{df_filtered['影响人数'].sum():,}人")

    with col_s4:
        high_risk_count = len(df_filtered[df_filtered["风险等级"].isin(["高风险", "重度预警"])])
        st.metric("高风险数量", f"{high_risk_count}条")

    st.markdown("---")

    # 数据表格
    st.markdown("### 📋 隐性风险明细表")

    # 格式化显示
    df_display = df_filtered[["风险类型", "主体名称", "所属区域", "风险等级", "风险评分", "主要风险", "状态", "更新时间"]].copy()

    # 风险等级颜色标记
    def highlight_risk_level(val):
        if val == "重度预警":
            return f'background-color: {COLORS["重度预警"]}; color: white; font-weight: bold;'
        elif val == "高风险":
            return f'background-color: {COLORS["高风险"]}; color: white; font-weight: bold;'
        elif val == "中风险":
            return f'background-color: {COLORS["中风险"]}; color: black; font-weight: bold;'
        elif val == "低风险":
            return f'background-color: {COLORS["低风险"]}; color: white; font-weight: bold;'
        return ''

    # 状态颜色标记
    def highlight_status(val):
        if val == "处置":
            return f'background-color: {COLORS["重度预警"]}; color: white;'
        elif val == "预警":
            return f'background-color: {COLORS["高风险"]}; color: white;'
        elif val == "关注":
            return f'background-color: {COLORS["中风险"]}; color: black;'
        return ''

    styled_df = df_display.style.applymap(highlight_risk_level, subset=["风险等级"]).applymap(highlight_status, subset=["状态"])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )

    st.markdown("---")

    # 导出功能
    col_export1, col_export2, col_export3 = st.columns([1, 1, 4])

    with col_export1:
        csv = df_filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 导出CSV",
            data=csv,
            file_name="隐性风险明细.csv",
            mime="text/csv"
        )

    with col_export2:
        # 简化的Excel导出（实际使用需要openpyxl）
        st.info("💡 提示：可使用CSV格式导出数据")

def page_warnings():
    """模块5：动态预警弹窗"""
    st.title("🚨 动态预警与风险处置")
    st.markdown("---")

    # 模拟预警数据
    warnings_data = [
        {"level": "重度", "region": "贵州省", "indicator": "债务率", "value": "125.3%", "threshold": "120%", "time": "2024-01-15 10:23:45"},
        {"level": "高", "region": "辽宁省", "indicator": "不良贷款率", "value": "4.2%", "threshold": "4%", "time": "2024-01-15 09:15:30"},
        {"level": "高", "region": "河南省", "indicator": "信贷违约率", "value": "18.5%", "threshold": "15%", "time": "2024-01-15 08:45:12"},
        {"level": "中", "region": "山东省", "indicator": "舆情指数", "value": "68.5", "threshold": "60", "time": "2024-01-15 08:30:00"},
        {"level": "中", "region": "四川省", "indicator": "非标债务压力", "value": "82.1", "threshold": "75", "time": "2024-01-15 07:50:25"},
    ]

    # 预警提示
    st.markdown("### ⚠️ 实时预警提示")

    for warning in warnings_data:
        level_color = COLORS["重度预警"] if warning["level"] == "重度" else COLORS["高风险"] if warning["level"] == "高" else COLORS["中风险"]

        st.markdown(f"""
        <div style='background-color: {COLORS["卡片背景"]}; padding: 15px; border-radius: 10px;
                    border-left: 5px solid {level_color}; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span style='background-color: {level_color}; color: white; padding: 3px 8px;
                                border-radius: 3px; font-weight: bold;'>{warning["level"]}预警</span>
                    <strong style='margin-left: 10px; color: {COLORS["文字主色"]};'>{warning["region"]}</strong>
                    <span style='color: {COLORS["文字次色"]}; margin-left: 10px;'>
                        {warning["indicator"]}超阈值
                    </span>
                </div>
                <div style='color: {COLORS["文字次色"]}; font-size: 12px;'>
                    {warning["time"]}
                </div>
            </div>
            <div style='margin-top: 10px; color: {COLORS["文字主色"]};'>
                当前值: <strong style='color: {level_color};'>{warning["value"]}</strong>
                <span style='margin-left: 20px; color: {COLORS["文字次色"]};'>
                    阈值: {warning["threshold"]}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 风险溯源路径
    st.markdown("### 🔍 风险溯源路径展示")

    col_source1, col_source2 = st.columns([2, 1])

    with col_source1:
        # 创建风险溯源流程图
        fig = go.Figure()

        # 定义节点位置
        nodes_trace = [
            {"name": "GDP增速下降", "x": 0, "y": 0.8},
            {"name": "财政收入减少", "x": 0.3, "y": 0.8},
            {"name": "债务压力上升", "x": 0.6, "y": 0.8},
            {"name": "信贷收紧", "x": 0.3, "y": 0.4},
            {"name": "企业违约风险", "x": 0.6, "y": 0.4},
            {"name": "不良贷款上升", "x": 0.9, "y": 0.4},
            {"name": "舆情恶化", "x": 0.6, "y": 0.0},
            {"name": "系统性风险", "x": 0.9, "y": 0.0}
        ]

        # 添加节点
        for node in nodes_trace:
            fig.add_trace(go.Scatter(
                x=[node["x"]],
                y=[node["y"]],
                mode='markers+text',
                marker=dict(size=40, color=COLORS["强调色"], line=dict(width=2, color='white')),
                text=[node["name"]],
                textposition='bottom center',
                textfont=dict(size=12, color=COLORS["文字主色"]),
                showlegend=False
            ))

        # 添加箭头连线
        arrows = [
            (0, 0.3, 0.8, 0.8),
            (0.3, 0.6, 0.8, 0.8),
            (0.6, 0.9, 0.8, 0.4),
            (0.3, 0.6, 0.4, 0.4),
            (0.6, 0.9, 0.4, 0.0),
            (0.6, 0.9, 0.4, 0.0),
            (0.6, 0.9, 0.0, 0.0)
        ]

        for x0, x1, y0, y1 in arrows:
            fig.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref='x', yref='y',
                axref='x', ayref='y',
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=COLORS["文字次色"],
                showarrow=True
            )

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.15, 0.95]),
            paper_bgcolor=COLORS["背景色"],
            plot_bgcolor=COLORS["背景色"],
            height=400,
            margin=dict(l=20, r=20, t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_source2:
        st.markdown("#### 📊 溯源统计")
        st.metric("传导深度", "4层", delta="较上次+1")
        st.metric("影响节点", "8个", delta="较上次+2")
        st.metric("传导时长", "12个月", delta="较上次+3")
        st.metric("风险强度", "85.3", delta="较上次+5.2")

    st.markdown("---")

    # 风险处置建议
    st.markdown("### 💡 风险处置建议")

    suggestions = [
        {
            "priority": "高",
            "title": "加强债务监管",
            "content": "建议对贵州省债务率超阈值问题启动专项监管，建立债务风险预警机制，优化债务结构，控制新增债务规模。",
            "department": "财政部门",
            "deadline": "2024-02-15"
        },
        {
            "priority": "高",
            "title": "优化信贷结构",
            "content": "建议辽宁省银行业机构优化信贷投放结构，加强贷后管理，提高拨备覆盖率，防范不良贷款率持续上升。",
            "department": "银保监局",
            "deadline": "2024-02-20"
        },
        {
            "priority": "中",
            "title": "舆情监测处置",
            "content": "建议山东省相关部门加强舆情监测，及时回应社会关切，防范舆情风险进一步传导至金融领域。",
            "department": "宣传部门",
            "deadline": "2024-02-10"
        },
        {
            "priority": "中",
            "title": "非标债务化解",
            "content": "建议四川省制定非标债务化解方案，通过债务重组、资产处置等方式降低非标债务压力。",
            "department": "金融监管部门",
            "deadline": "2024-03-01"
        }
    ]

    for suggestion in suggestions:
        priority_color = COLORS["重度预警"] if suggestion["priority"] == "高" else COLORS["中风险"]

        st.markdown(f"""
        <div style='background-color: {COLORS["卡片背景"]}; padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                <div>
                    <span style='background-color: {priority_color}; color: white; padding: 3px 8px;
                                border-radius: 3px; font-weight: bold;'>
                        {suggestion["priority"]}优先级
                    </span>
                    <strong style='margin-left: 10px; font-size: 16px; color: {COLORS["文字主色"]};'>
                        {suggestion["title"]}
                    </strong>
                </div>
                <div style='color: {COLORS["文字次色"]}; font-size: 12px;'>
                    截止日期: {suggestion["deadline"]}
                </div>
            </div>
            <div style='color: {COLORS["文字主色"]}; margin-bottom: 10px;'>
                {suggestion["content"]}
            </div>
            <div style='color: {COLORS["文字次色"]}; font-size: 12px;'>
                责任部门: {suggestion["department"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序入口"""
    # 侧边栏导航
    st.sidebar.title("🛡️ 金融风险预警系统")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "导航菜单",
        options=[
            "📊 区域金融总览",
            "🤖 多智能体分析",
            "🔗 因果推演链路",
            "🔍 隐性风险穿透",
            "🚨 动态预警处置"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ 系统信息")
    st.sidebar.info("""
    **版本**: v1.0.0
    **更新时间**: 2024-01-15
    **技术栈**: Streamlit + Plotly
    **部署方式**: GitHub Pages
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 使用说明")
    st.sidebar.markdown("""
    1. 选择左侧菜单查看不同模块
    2. 使用筛选器过滤数据
    3. 点击图表可交互查看详情
    4. 支持数据导出功能
    """)

    # 根据选择显示不同页面
    if page == "📊 区域金融总览":
        page_overview()
    elif page == "🤖 多智能体分析":
        page_agents()
    elif page == "🔗 因果推演链路":
        page_causal()
    elif page == "🔍 隐性风险穿透":
        page_hidden_risks()
    elif page == "🚨 动态预警处置":
        page_warnings()

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: {}; font-size: 12px;'>
        © 2024 多智能体AI+大数据因果推演的区域性地方金融隐性风险穿透预警系统 |
        技术支持: Streamlit + stlite | 部署平台: GitHub Pages
    </div>
    """.format(COLORS["文字次色"]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()