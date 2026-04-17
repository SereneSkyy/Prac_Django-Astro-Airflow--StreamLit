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

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# --- UI STYLE (Optimized for Dark/Light Mode) ---
st.markdown("""<style>[data-testid="stMetric"] { background-color: rgba(0,0,0,0); border: 1px solid rgba(128, 128, 128, 0.3); padding: 15px; border-radius: 12px; }
    h1 { color: #1e293b; font-weight: 800; }</style>""", unsafe_allow_html=True)

if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'current_topic' not in st.session_state: st.session_state.current_topic = ""
if 'pruned' not in st.session_state: st.session_state.pruned = False

def render_styled_tree(topic):
    try:
        threshold = 0.05 if st.session_state.pruned else 0.0
        res = requests.get(f"{API_BASE}/retrieve/tree/{topic}")
        if res.status_code != 200: return st.info("Intelligence map synchronizing...")
        flat_nodes = res.json()

        def build_nested(p_id):
            children = []
            for n in flat_nodes:
                if str(n.get('parent_id')) == str(p_id):
                    score, imp = n.get('lstm_val', 0.5), n.get('imp_val', 0)
                    if imp < threshold: continue 
                    # Vibrant Color logic
                    if score > 0.51: label, color = "Positive", "#00FF7F"
                    elif score < 0.49: label, color = "Negative", "#FF3131"
                    else: label, color = "Neutral", "#94a3b8"
                    
                    children.append({
                        "name": str(n.get('text', '')).upper(), # Node Name Only
                        "value": f"Sentiment: {label} | Significance: {round(imp*100, 1)}%", # Hover Text
                        "symbolSize": min(max(15 + (imp * 120), 15), 45),
                        "itemStyle": {"color": color, "borderColor": "#fff", "borderWidth": 1.5},
                        "children": build_nested(n.get('id'))
                    })
            return children

        chart_data = {"name": topic.upper(), "symbolSize": 25, "itemStyle": {"color": "#1e293b"}, "children": build_nested(None)}
        if st.button("Show Full Tree" if st.session_state.pruned else "Prune Noise"):
            st.session_state.pruned = not st.session_state.pruned; st.rerun()

        st_echarts({"tooltip": {"trigger": "item", "formatter": "<b>{b}</b><br/>{c}"},
            "series": [{"type": "tree", "data": [chart_data], "initialTreeDepth": -1, "expandAndCollapse": True, "label": {"position": "top", "fontSize": 10}}]}, height="550px")
    except Exception as e: st.error(f"Visualization Logic Error: {e}")

# ... (Monitor DAG logic remains same) ...
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

# MAIN SEARCH
st.title("Sentiment Analyzer")
search_container = st.container(border=True)
with search_container:
    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    topic_input = col1.text_input("Analysis Concept", placeholder="Enter topic...")
    trigger_btn = col2.button("Run", use_container_width=True, type="primary")

if trigger_btn and topic_input:
    st.session_state.analysis_done = False
    try:
        with st.status("Analyzing...", expanded=True) as status_box:
            init_res = requests.post(f"{API_BASE}/ingest/{DEFAULT_DAG_ID}/{topic_input}").json()
            if monitor_dag_progress(DEFAULT_DAG_ID, init_res["dag_run_id"]) == "success":
                time.sleep(5)
                res = requests.get(f"{AIRFLOW_API}/embed_dag/dagRuns?limit=1&order_by=-execution_date", auth=("airflow", "airflow")).json()
                if res["dag_runs"] and monitor_dag_progress("embed_dag", res["dag_runs"][0]["dag_run_id"]) == "success":
                    st.session_state.analysis_done = True; st.session_state.current_topic = topic_input
                    status_box.update(label="Complete!", state="complete")
    except Exception as e: st.error(f"Error: {e}")

if st.session_state.analysis_done:
    topic = st.session_state.current_topic
    result_rows = preview_data(topic)
    if result_rows and isinstance(result_rows, list):
        df = pd.DataFrame(result_rows)
        st.divider(); m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Samples", len(df))
        df_v = df[df['sentiment'] != 'Pending']
        m2.metric("Positive", len(df_v[df_v['sentiment'] == 'Positive']))
        m3.metric("Neutral", len(df_v[df_v['sentiment'] == 'Neutral']))
        m4.metric("Negative", len(df_v[df_v['sentiment'] == 'Negative']))
        acc_res = requests.get(f"{API_BASE}/retrieve/accuracy/{topic}")
        m5.metric("Model MAC", f"{acc_res.json() if acc_res.status_code == 200 else 0.0}%")
        
        st.subheader("Semantic Taxonomy Tree"); render_styled_tree(topic)
        st.subheader("Public Opinion Feed")
        # --- REQUIREMENT: SHOW DATE POSTED, HIDE PROCESSED ---
        mapping = {'author': 'Author', 'language': 'Lang', 'comment': 'YouTube Comment', 'sentiment': 'AI Sentiment', 'p_timestamp': 'Date Posted'}
        st.dataframe(df[[c for c in mapping.keys() if c in df.columns]].rename(columns=mapping), use_container_width=True, hide_index=True)