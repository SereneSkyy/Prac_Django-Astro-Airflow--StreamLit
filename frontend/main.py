import sys, os, json, time, requests
import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts

# 1. PATH CONFIGURATION
# Preserving the logic to bridge the frontend with dataPipeline services
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_pipeline_path = os.path.join(project_root, 'dataPipeline')

if project_root not in sys.path: sys.path.append(project_root)
if data_pipeline_path not in sys.path: sys.path.append(data_pipeline_path)

from services.check_dag_status import get_tasks, check_dag_status
from services.retrieve_data import preview_data

# --- SYSTEM CONFIGURATION ---
API_BASE = "http://127.0.0.1:8000/api"
AIRFLOW_API = "http://127.0.0.1:8080/api/v1/dags"
DEFAULT_DAG_ID = "genz_dag" # Hardcoded to simplify user experience

st.set_page_config(page_title="Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- UI STYLING ---
st.markdown("""
    <style>
    /* Professional Metric Cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
    }
    /* Header Font Styling */
    h1 { color: #1e293b; font-weight: 800; }
    .stDataFrame { border-radius: 12px; }
    </style>
    """, unsafe_allow_html=True)

def render_styled_tree(topic):
    """
    Renders an interactive Knowledge Graph.
    Uses recursive mapping to show branching hierarchy and hover insights.
    """
    try:
        res = requests.get(f"{API_BASE}/retrieve/tree/{topic}")
        if res.status_code != 200: return st.info("Synchronizing taxonomy...")
        flat_nodes = res.json()
        if not flat_nodes or 'error' in str(flat_nodes): return st.info("Analysis in progress...")

        def build_nested(p_id):
            children = []
            for n in flat_nodes:
                if str(n.get('parent_id')) == str(p_id):
                    l_val = n.get('lstm_val', 0.5)
                    imp = n.get('imp_val', 0)
                    
                    # Professional Color Palette: Emerald (Pos), Rose (Neg), Slate (Neu)
                    color = "#10b981" if l_val > 0.6 else ("#f43f5e" if l_val < 0.4 else "#64748b")
                    sent_text = "Positive" if l_val > 0.6 else ("Negative" if l_val < 0.4 else "Neutral")
                    
                    # Symbol Size correlates to Importance (min 12, max 35)
                    size = 12 + (imp * 80)
                    size = min(max(size, 12), 35)

                    children.append({
                        "name": str(n.get('text', '')).upper(),
                        "value": f"{round(imp * 100, 1)}%", # Mapped to {c} in tooltip
                        "sentiment": sent_text,
                        "symbolSize": size,
                        "itemStyle": {"color": color, "borderColor": color, "borderWidth": 2},
                        "children": build_nested(n.get('id'))
                    })
            return children

        # Primary concept node
        chart_data = {
            "name": topic.upper(), 
            "symbolSize": 20, 
            "itemStyle": {"color": "#1e293b"}, 
            "children": build_nested(None)
        }
        
        opts = {
            "tooltip": {
                "trigger": "item",
                "triggerOn": "mousemove",
                "formatter": "<b>{b}</b><br/>Importance: {c}<br/>"
            },
            "series": [{
                "type": "tree",
                "data": [chart_data],
                "top": "5%", "left": "15%", "bottom": "5%", "right": "20%",
                "symbol": "circle",
                "edgeShape": "curve", 
                "lineStyle": {"width": 2, "curveness": 0.5, "color": "#cbd5e1"},
                "label": {"position": "top", "fontSize": 12, "fontWeight": "600", "distance": 8},
                "leaves": {"label": {"position": "right", "align": "left"}},
                "emphasis": {"focus": "descendant"},
                "expandAndCollapse": True,
                "animationDuration": 600
            }]
        }
        st_echarts(opts, height="500px")
    except Exception as e: st.error(f"Visualization Error: {e}")

def monitor_dag_progress(dag_id, run_id):
    """Wait for all tasks in a DAG to finish success."""
    queue = get_tasks(dag_id, run_id)[::-1]
    progress_bar = st.progress(0)
    total, done = len(queue), 0
    while queue:
        status = check_dag_status(queue[-1], dag_id, run_id)
        if status == "success":
            queue.pop(); done += 1
            progress_bar.progress(done / total)
        elif status in ("failed", "skipped"): return status
        time.sleep(2)
    return "success"

# --- HEADER ---
st.title("Sentiment Analyzer")
st.caption("AI Engine • BERT Semantic Taxonomy • Bi-LSTM Opinion Classification")

# --- ALIGNED SEARCH BAR ---
# Using vertical_alignment="bottom" ensures the button and text input are on the same line
search_container = st.container(border=True)
with search_container:
    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    topic_input = col1.text_input("Analysis Topic", placeholder="Enter a topic (e.g. Nepal politics)")
    trigger_btn = col2.button("Run Analysis", use_container_width=True, type="primary")

if trigger_btn and topic_input:
    try:
        # EXECUTION PHASES
        with st.status("Initializing Analysis Pipeline...", expanded=True) as status_box:
            status_box.write("Phase 1: Scraping data and validating sources...")
            init_res = requests.post(f"{API_BASE}/ingest/{DEFAULT_DAG_ID}/{topic_input}").json()
            
            if monitor_dag_progress(DEFAULT_DAG_ID, init_res["dag_run_id"]) == "success":
                status_box.write("Phase 2: Semantic Mapping & Sentiment Inference...")
                time.sleep(5) # Registration buffer
                res = requests.get(f"{AIRFLOW_API}/embed_dag/dagRuns?limit=1&order_by=-execution_date", auth=("airflow", "airflow")).json()
                
                if res["dag_runs"] and monitor_dag_progress("embed_dag", res["dag_runs"][0]["dag_run_id"]) == "success":
                    status_box.update(label="Analysis Sequence Complete!", state="complete", expanded=False)
                    st.toast("Intelligence Synchronized", icon="✅")
                else: st.error("AI Phase Failed")
            else: st.error("Data Collection Failed")

        # --- VISUALIZATION LAYER ---
        result_rows = preview_data(topic_input)
        if result_rows and isinstance(result_rows, list):
            df = pd.DataFrame(result_rows)
            
            # Row 1: Dashboard Metrics
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sample Size", len(df))
            if 'sentiment' in df.columns:
                m2.metric("Positive Opinions", len(df[df['sentiment'] == 'Positive']))
                m3.metric("Negative Opinions", len(df[df['sentiment'] == 'Negative']))
                m4.metric("Neutral / Others", len(df[df['sentiment'] == 'Neutral']))

            # Row 2: Knowledge Graph
            st.subheader("Semantic Taxonomy Tree")
            st.caption("Interactive graph showing hierarchical concept relationships. Hover nodes for weight and context.")
            render_styled_tree(topic_input)

            # Row 3: Public Sentiment Feed (Cleaned Data Hidden)
            st.subheader("Public Opinion Feed")
            mapping = {
                'author': 'Author',
                'comment': 'YouTube Comment',
                'sentiment': 'AI Sentiment',
                'p_timestamp': 'Date Posted'
            }
            # Rename existing columns for viewer readability
            existing = [c for c in mapping.keys() if c in df.columns]
            clean_view_df = df[existing].copy().rename(columns=mapping)
            
            st.dataframe(clean_view_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Analysis complete, but no valid data matches the specified filters.")

    except Exception as e:
        st.error(f"System Operational Error: {e}")