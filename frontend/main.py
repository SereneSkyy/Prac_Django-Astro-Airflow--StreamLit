import sys
import os
import streamlit as st
import requests
import pandas as pd

# Minimal addition for importing the AI Engine
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path: sys.path.append(project_root)

from services.check_dag_status import check_dag_status, get_skip_reason
from services.retrieve_data import preview_data
from dataPipeline.services.nlp_engine import NLPEngine # Import Engine

DJANGO_INGEST_API = "http://127.0.0.1:8000/api/ingest"

st.title("Start Fetching and Analyzing Data")

with st.form("dag_form"):
    dag_id = st.text_input(label="Enter Dag ID", max_chars=255)
    topic = st.text_input(label="Enter Topic", max_chars=255)
    trigger_btn = st.form_submit_button("Trigger Tasks")


def trigger_dag(dag_id, topic):
    response = requests.post(
        f"{DJANGO_INGEST_API}/{dag_id}/{topic}",
        timeout=10
    )

    if response.status_code != 200:
        raise RuntimeError(
            response.json().get("error", "Failed to trigger DAG")
        )

    return response.json()


if trigger_btn and topic:
    try:
        # Trigger DAG
        data = trigger_dag(dag_id, topic)
        dag_run_id = data["dag_run_id"]
        status = data["state"]

        # Poll DAG status
        while status in ("queued", "running"):
            status = check_dag_status(dag_id, dag_run_id)

        # Final state handling
        if status == "failed":
            st.error("DAG run FAILED!")

        elif status == "skipped":
            skip_reason = get_skip_reason(
                dag_id,
                dag_run_id,
                task_id="extract_data"
            )
            st.error(skip_reason)

        else:
            st.toast(f":green[DAG finished with status: {status}]", duration=2)
            
            # 1. Fetch data and filter strictly for English
            result = preview_data(topic)
            english_data = [item for item in result if item.get('language') == 'en']

            if not english_data:
                st.warning(f"No English comments found for topic: {topic}")
            else:
                # 2. Run AI Engine Accuracy Testing
                with st.spinner("Running Topic Modeling..."):
                    winner, scores = NLPEngine.run_full_pipeline(english_data)
                
                # 3. Display Accuracy Dashboard
                st.success(f"Best Topic Model: {winner}")
                acc_df = pd.DataFrame(list(scores.items()), columns=['Model', 'Coherence'])
                st.bar_chart(data=acc_df, x='Model', y='Coherence')

                # 4. Display Raw English Data only
                st.subheader(f"English Comments for: {topic}")
                df = pd.DataFrame(english_data)
                st.dataframe(df[['author', 'comment', 'p_timestamp']], use_container_width=True)

    except requests.RequestException as e:
        st.error(f"Request error: {e}")

    except KeyError as e:
        st.error(f"Missing expected response field: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")