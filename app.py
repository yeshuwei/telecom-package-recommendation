"""
电信套餐智能推荐系统 - Streamlit 可视化界面

功能：
1. 接受用户输入（用户ID + 需求描述）
2. 实时展示系统"思考"过程（各智能体的执行状态）
3. 显示最终推荐结果
"""
import streamlit as st
import sys
from pathlib import Path
import json
import time

# 将项目根目录添加到Python路径中，以便导入模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.workflow_graph import app
from agent.state import AgentState

# --- 节点名称映射（中文显示）---
NODE_NAME_MAP = {
    "entry_router": "🏁 入口路由",
    "router": "🧭 意图识别路由",
    "slot_filling_agent": "📝 槽位填充智能体",
    "user_info_agent": "👤 用户信息查询",
    "recommendation_agent": "🎯 套餐推荐智能体",
    "price_selection_agent": "💰 价位选择智能体",
    "response_generation": "✍️ 回复生成智能体",
    "knowledge_agent": "📚 知识库查询",
    "comparison_agent": "⚖️ 套餐对比",
    "general_agent": "💬 通用对话"
}

# --- UI配置 ---
st.set_page_config(
    page_title="电信套餐智能推荐系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 自定义CSS样式 ---
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .thinking-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .final-result {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        color: #0b2e4f;
        font-size: 1.05rem;
        line-height: 1.7;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .final-result h3,
    .final-result p,
    .final-result li,
    .final-result strong,
    .final-result em {
        color: #0b2e4f;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🤖 电信套餐智能推荐系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">基于多智能体协作的个性化套餐推荐引擎</div>', unsafe_allow_html=True)

# --- 侧边栏输入 ---
with st.sidebar:
    st.header("👤 用户信息输入")
    
    # 提供一些预设场景方便测试
    scenario = st.selectbox(
        "💡 快速选择测试场景：",
        [
            "自定义输入",
            "场景1: 有明确需求的老年用户",
            "场景2: 仅有历史数据的用户",
            "场景3: 有预算和流量需求的年轻用户"
        ]
    )

    if scenario == "场景1: 有明确需求的老年用户":
        default_phone = "13800138001"
        default_input = "我是老年人，每月预算50元，想办5G和宽带，我经常看视频，流量用的不多，通话时长也要求不高"
    elif scenario == "场景2: 仅有历史数据的用户":
        default_phone = "13800138002"
        default_input = "给我推荐个套餐吧"
    elif scenario == "场景3: 有预算和流量需求的年轻用户":
        default_phone = "13800138003"
        default_input = "我想办个5G套餐，预算150元左右，流量要多一点"
    else:  # 自定义输入
        default_phone = "13800138000"
        default_input = ""

    user_phone_number = st.text_input(
        "📱 电话号码:",
        value=default_phone,
        help="输入您的电话号码（业务号码）"
    )
    
    user_input = st.text_area(
        "📝 需求描述:",
        value=default_input,
        height=150,
        placeholder="请输入您的套餐需求，例如：我想办个5G套餐，预算100元左右...",
        help="详细描述您的需求，包括预算、流量、通话等"
    )
    
    st.divider()
    
    submit_button = st.button("🚀 开始推荐", type="primary", use_container_width=True)
    
    if submit_button and not user_input.strip():
        st.warning("⚠️ 请输入您的需求描述！")
        submit_button = False

# --- 主界面布局 ---
if submit_button:
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.header("🧠 系统思考过程")
        thinking_container = st.container()
        
    with col2:
        st.header("💡 最终推荐结果")
        result_placeholder = st.empty()
        result_placeholder.info("⏳ 正在生成推荐，请稍候...")
        
        st.divider()
        st.subheader("📊 关键信息摘要")
        summary_placeholder = st.empty()

    # 准备初始状态
    initial_state: AgentState = {
        "phone_number": user_phone_number,
        "input": user_input,
        "chat_history": []
    }

    try:
        # 使用 .stream() 方法来实时获取工作流的每一步状态
        events = app.stream(initial_state)
        
        final_state = None
        step_count = 0

        for event in events:
            for node_name, state_update in event.items():
                step_count += 1
                
                # 获取节点的中文名称
                display_name = NODE_NAME_MAP.get(node_name, f"🔧 {node_name}")
                
                # 实时更新"思考"过程
                with thinking_container:
                    with st.expander(f"**步骤 {step_count}: {display_name}**", expanded=(step_count <= 3)):
                        # 提取关键信息进行展示
                        if isinstance(state_update, dict):
                            # 显示关键字段
                            key_fields = ["next_node_to_call", "final_response", "db_user_summary", 
                                         "user_explicit_needs", "recommendation_results", "price_selection_results"]
                            
                            for field in key_fields:
                                if field in state_update and state_update[field]:
                                    value = state_update[field]
                                    
                                    # 格式化显示
                                    if field == "next_node_to_call":
                                        next_node_display = NODE_NAME_MAP.get(value, value)
                                        st.info(f"**下一步**: {next_node_display}")
                                    elif field == "final_response":
                                        st.success(f"**生成回复**: {value[:100]}..." if len(str(value)) > 100 else f"**生成回复**: {value}")
                                    elif field == "db_user_summary":
                                        st.write(f"**用户画像**: {value}")
                                    elif field == "recommendation_results":
                                        if isinstance(value, list) and value:
                                            st.write(f"**推荐系列数**: {len(value)} 个")
                                            for i, rec in enumerate(value[:3], 1):  # 只显示前3个
                                                st.write(f"  {i}. {rec.get('series', 'N/A')} (评分: {rec.get('score', 0):.2f})")
                                    elif field == "price_selection_results":
                                        if isinstance(value, list) and value:
                                            st.write(f"**精选套餐数**: {len(value)} 个")
                                            for i, pkg in enumerate(value[:3], 1):
                                                st.write(f"  {i}. {pkg.get('plan_name', 'N/A')} - ¥{pkg.get('price', 0)}/月")
                            
                           # 显示完整状态数据
                            st.markdown("**完整状态数据：**")
                            clean_state = {k: v for k, v in state_update.items() if v is not None and not k.startswith('_')}
                            st.json(clean_state, expanded=False)
                        else:
                            st.write(state_update)
                
                final_state = state_update  # 持续保存最新状态
                time.sleep(0.1)  # 添加小延迟，让UI更新更流畅

        # 工作流结束后，显示最终结果
        if final_state:
            final_response = final_state.get("final_response", "抱歉，系统未能生成推荐结果，请稍后再试。")
            
            with col2:
                result_placeholder.markdown(f"""
                <div class="final-result">
                    <h3>✅ 推荐结果</h3>
                    <p>{final_response}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示关键信息摘要
                with summary_placeholder.container():
                    if final_state.get("price_selection_results"):
                        results = final_state["price_selection_results"]
                        st.metric("推荐套餐数", len(results))
                        
                        if results:
                            avg_price = sum(r.get("price", 0) for r in results) / len(results)
                            st.metric("平均价格", f"¥{avg_price:.0f}/月")
                            
                            avg_score = sum(r.get("score", 0) for r in results) / len(results)
                            st.metric("平均匹配度", f"{avg_score:.2f}")
        else:
            result_placeholder.error("❌ 工作流未能正确执行，无法获取最终结果。")

    except Exception as e:
        st.error(f"❌ 系统在运行过程中发生错误: {e}")
        with st.expander("查看详细错误信息"):
            import traceback
            st.code(traceback.format_exc())
else:
    # 欢迎界面
    st.info("👈 请在左侧输入您的信息，然后点击'开始推荐'按钮。")
    
    # 显示系统架构图
    st.subheader("📐 系统架构")
    st.markdown("""
    本系统采用**多智能体协作架构**，包含以下核心模块：
    
    1. **🧭 意图识别路由**: 识别用户意图，分发到对应智能体
    2. **📝 槽位填充智能体**: 提取用户需求，补充必要信息
    3. **🎯 套餐推荐智能体**: 基于用户画像进行系列推荐
    4. **💰 价位选择智能体**: 在推荐系列中选择最佳价位套餐
    5. **✍️ 回复生成智能体**: 生成自然、专业的推荐话术
    
    系统集成了：
    - 🗄️ **MySQL数据库**: 存储用户历史数据和套餐信息
    - [object Object]us向量数据库**: RAG知识检索
    - 🤖 **Gemini大模型**: 自然语言理解与生成
    """)
    
    # 显示示例
    st.subheader("💡 使用示例")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    with col_ex1:
        st.info("""
        **场景1: 老年用户**
        
        电话号码: 17354132409
        
        需求: "我是老年人，每月预算50元，想办5G和宽带，我经常看视频，流量用的不多，通话时长也要求不高"
        """)
    
    with col_ex2:
        st.info("""
        **场景2: 简单查询**
        
        电话号码: 17354132409
        
        需求: "给我推荐个套餐吧"
        """)
    
    with col_ex3:
        st.info("""
        **场景3: 年轻用户**
        
        电话号码: 17354132409
        
        需求: "我想办个5G套餐，预算150元左右..."
        """)
