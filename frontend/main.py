import sys, os, json, time, requests
import streamlit as st
import pandas as pd
from streamlit_echarts import st_echarts

# 1. PATH CONFIGURATION
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_pipeline_path = os.path.join(project_root, 'dataPipeline')

if project_root not in sys.path: sys.path.append(project_root)
if data_pipeline_path not in sys.path: sys.path.append(data_pipeline_path)

from services.check_dag_status import get_tasks, check_dag_status
from services.retrieve_data import preview_data

# --- SYSTEM SETTINGS ---
API_BASE = "http://127.0.0.1:8000/api"
AIRFLOW_API = "http://127.0.0.1:8080/api/v1/dags"
DEFAULT_DAG_ID = "genz_dag"

st.set_page_config(page_title="Sentiment Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- UI STYLE ---
st.markdown("""
    <style>
    [data-testid="stMetric"] { background-color: #ffffff; border: 1px solid #e2e8f0; padding: 15px; border-radius: 12px; }
    h1 { color: #1e293b; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

def render_styled_tree(topic):
    """Knowledge Graph that filters out 0% importance nodes and shows rich tooltips."""
    try:
        res = requests.get(f"{API_BASE}/retrieve/tree/{topic}")
        if res.status_code != 200: return st.info("Intelligence map synchronizing...")
        flat_nodes = res.json()
        if not flat_nodes or 'error' in str(flat_nodes): return st.info("Tree is being prepared...")

        def build_nested(p_id):
            children = []
            for n in flat_nodes:
                # Ensure we compare UUIDs as strings
                if str(n.get('parent_id')) == str(p_id):
                    score = n.get('lstm_val', 0.5)
                    imp = n.get('imp_val', 0)
                    
                    # --- REQUIREMENT: HIDE 0% IMPORTANCE NODES ---
                    if imp <= 0:
                        continue 
                    
                    # Sentiment Labels
                    if score > 0.6: label, color = "Positive", "#10b981"
                    elif score < 0.4: label, color = "Negative", "#f43f5e"
                    else: label, color = "Neutral", "#64748b"
                    
                    # Visual Sizing
                    size = 12 + (imp * 80)
                    size = min(max(size, 12), 35)

                    children.append({
                        "name": str(n.get('text', '')).upper(),
                        "value": f"Imp: {round(imp*100, 1)}% | Sentiment: {label}",
                        "symbolSize": size,
                        "itemStyle": {"color": color, "borderColor": color, "borderWidth": 2},
                        "children": build_nested(n.get('id'))
                    })
            return children

        chart_data = {
            "name": topic.upper(), 
            "symbolSize": 20, 
            "itemStyle": {"color": "#1e293b"}, 
            "children": build_nested(None)
        }
        
        opts = {
            "tooltip": {
                "trigger": "item", "triggerOn": "mousemove",
                "formatter": "<b>{b}</b><br/>{c}" 
            },
            "series": [{
                "type": "tree", "data": [chart_data], "top": "5%", "left": "15%", "bottom": "5%", "right": "20%",
                "symbol": "circle", "edgeShape": "curve", 
                "lineStyle": {"width": 2, "curveness": 0.5, "color": "#cbd5e1"},
                "label": {"position": "top", "fontSize": 12, "fontWeight": "600"},
                "leaves": {"label": {"position": "right", "align": "left"}},
                "expandAndCollapse": True, "initialTreeDepth": 2, "animationDuration": 600
            }]
        }
        st_echarts(opts, height="550px")
    except Exception as e: st.error(f"Visualization Logic Error: {e}")

def monitor_dag_progress(dag_id, run_id):
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

# --- UI ---
st.title("Sentiment Analyzer")
st.caption("AI Engine • BERT Semantic Taxonomy • Bi-LSTM Classification")

search_container = st.container(border=True)
with search_container:
    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    topic_input = col1.text_input("Analysis Concept", placeholder="Enter a topic...")
    trigger_btn = col2.button("Run Analysis", use_container_width=True, type="primary")

if trigger_btn and topic_input:
    try:
        with st.status("Initializing Analysis Pipeline...", expanded=True) as status_box:
            status_box.write("Step 1: Collecting data...")
            init_res = requests.post(f"{API_BASE}/ingest/{DEFAULT_DAG_ID}/{topic_input}").json()
            if monitor_dag_progress(DEFAULT_DAG_ID, init_res["dag_run_id"]) == "success":
                status_box.write("Step 2: Processing BERT & LSTM Inference...")
                time.sleep(5)
                res = requests.get(f"{AIRFLOW_API}/embed_dag/dagRuns?limit=1&order_by=-execution_date", auth=("airflow", "airflow")).json()
                if res["dag_runs"] and monitor_dag_progress("embed_dag", res["dag_runs"][0]["dag_run_id"]) == "success":
                    status_box.update(label="Analysis Sequence Complete!", state="complete", expanded=False)
                else: st.error("AI Phase Failed")
            else: st.error("Data Collection Failed")

        # --- RESULTS ---
        result_rows = preview_data(topic_input)
        if result_rows and isinstance(result_rows, list):
            df = pd.DataFrame(result_rows)
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sample Size", len(df))
            if 'sentiment' in df.columns:
                m2.metric("Positive", len(df[df['sentiment'] == 'Positive']))
                m3.metric("Negative", len(df[df['sentiment'] == 'Negative']))
                m4.metric("Neutral", len(df[df['sentiment'] == 'Neutral']))

            st.subheader("Semantic Taxonomy Tree")
            render_styled_tree(topic_input)

            st.subheader("Public Opinion Feed")
            # --- REQUIREMENT: ADD LANGUAGE COLUMN ---
            mapping = {
                'author': 'Author', 
                'language': 'Lang', # Added Language
                'comment': 'YouTube Comment', 
                'sentiment': 'AI Sentiment', 
                'p_timestamp': 'Date Posted'
            }
            existing = [c for c in mapping.keys() if c in df.columns]
            clean_df = df[existing].copy().rename(columns=mapping)
            
            # Format language to uppercase for professional look
            if 'Lang' in clean_df.columns:
                clean_df['Lang'] = clean_df['Lang'].str.upper()

            st.dataframe(clean_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No data found for this search.")
    except Exception as e: st.error(f"Handshake Error: {e}")