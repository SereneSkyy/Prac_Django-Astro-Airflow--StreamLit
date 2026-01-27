import sys, os, json, time, requests
import streamlit as st
import pandas as pd
from services.check_dag_status import get_tasks, check_dag_status
from services.retrieve_data import preview_data

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DP_PATH = os.path.join(BASE_DIR, 'dataPipeline')
for p in [BASE_DIR, DP_PATH]:
    if p not in sys.path: sys.path.append(p)

# --- CONSTANTS ---
API_INGEST = "http://127.0.0.1:8000/api/ingest"
API_AIRFLOW = "http://127.0.0.1:8080/api/v1/dags"
TREE_FILE = os.path.join(DP_PATH, "hierarchy_tree.json")

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# --- UI HELPER: TREE GENERATOR ---
def build_tree_string(node, tree_data, prefix="", is_last=True, is_root=False):
    """
    Recursively builds a directory-style tree string.
    """
    lines = []
    # Current node formatting
    marker = "" if is_root else ("└── " if is_last else "├── ")
    lines.append(f"{prefix}{marker}{node.upper()}")
    
    # Prepare prefix for children
    new_prefix = prefix + ("    " if is_last else "│   ")
    
    children = tree_data.get(node, [])
    for i, child in enumerate(children):
        last_child = (i == len(children) - 1)
        lines.extend(build_tree_string(child, tree_data, new_prefix, last_child))
        
    return lines

# --- CORE LOGIC ---
def run_monitor(dag_id, run_id):
    """Tracks DAG progress with a progress bar."""
    tasks = get_tasks(dag_id, run_id)[::-1]
    progress = st.progress(0)
    total = len(tasks)
    done = 0
    
    while tasks:
        state = check_dag_status(tasks[-1], dag_id, run_id)
        if state == "success":
            tasks.pop()
            done += 1
            progress.progress(done / total)
        elif state in ("failed", "skipped"):
            return False
        time.sleep(2)
    return True

# --- SIDEBAR ---
with st.sidebar:
    st.header("Taxonomy Tree")
    if st.button("Generate Structure"):
        if os.path.exists(TREE_FILE):
            with open(TREE_FILE, "r") as f:
                data = json.load(f)
                tree_map = data.get("tree", {})
                roots = data.get("roots", [])
                
                all_lines = []
                for root in roots:
                    all_lines.extend(build_tree_string(root, tree_map, is_root=True))
                
                # Displaying as a code block keeps the lines perfectly aligned
                st.code("\n".join(all_lines), language="text")
        else:
            st.error("Missing hierarchy_tree.json")

# --- MAIN INTERFACE ---
st.title("Sentiment Analyzer")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    dag_input = col1.text_input("Pipeline ID", value="genz_dag")
    topic_input = col2.text_input("Analysis Topic")
    run_btn = st.form_submit_button("Start Analysis")

if run_btn and topic_input:
    try:
        # Step 1: Ingestion
        st.subheader("Progress")
        st.write(f"Phase 1: Collecting data for '{topic_input}'")
        init_res = requests.post(f"{API_INGEST}/{dag_input}/{topic_input}").json()
        
        if run_monitor(dag_input, init_res["dag_run_id"]):
            # Step 2: AI Processing
            st.write("Phase 2: BERT Embedding and LSTM Sentiment Analysis")
            time.sleep(5) # Brief wait for DagRun registration
            
            dag_list = requests.get(f"{API_AIRFLOW}/embed_dag/dagRuns?limit=1&order_by=-execution_date", auth=("airflow", "airflow")).json()
            
            if dag_list["dag_runs"] and run_monitor("embed_dag", dag_list["dag_runs"][0]["dag_run_id"]):
                st.success("Analysis sequence complete")
                
                # Step 3: Retrieval
                results = preview_data(topic_input)
                if results and isinstance(results, list):
                    df = pd.DataFrame(results)
                    
                    # Sentiment Breakdown
                    if 'sentiment' in df.columns:
                        st.divider()
                        st.subheader("Sentiment Metrics")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Positive", len(df[df['sentiment'] == 'Positive']))
                        m2.metric("Negative", len(df[df['sentiment'] == 'Negative']))
                        m3.metric("Neutral", len(df[df['sentiment'] == 'Neutral']))
                    
                    # Data Display
                    st.subheader("Processed Data")
                    target_cols = ['author', 'comment', 'sentiment', 'language']
                    st.dataframe(
                        df[[c for c in target_cols if c in df.columns]], 
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.warning("No data found for this topic.")
            else:
                st.error("AI Analysis failed or timed out.")
        else:
            st.error("Data collection phase failed.")
            
    except Exception as e:
        st.error(f"Operational error: {e}")