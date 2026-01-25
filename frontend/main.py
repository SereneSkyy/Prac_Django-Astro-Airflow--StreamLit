import sys
import os
import streamlit as st
import requests
import pandas as pd
from services.check_dag_status import check_dag_status, get_skip_reason
from services.retrieve_data import preview_data

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
            
            result = preview_data(topic)
            if result:
                df = pd.DataFrame(result)
                st.dataframe(df, use_container_width=True)
            else:
                st.write("API ERROR, PLS TRY AGAIN LATER!")
                        
    except requests.RequestException as e:
        st.error(f"Request error: {e}")

    except KeyError as e:
        st.error(f"Missing expected response field: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")