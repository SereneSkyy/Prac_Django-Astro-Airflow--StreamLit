import requests
from requests.auth import HTTPBasicAuth
import streamlit as st
import time

AIRFLOW_USERNAME, AIRFLOW_PASSWORD = "airflow", "airflow"

DAG_STATUS_CHECK_API = "http://127.0.0.1:8080/api/v1/dags"

def get_tasks(dag_id, dag_run_id):
    url = f"{DAG_STATUS_CHECK_API}/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    response = requests.get(url, auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
    response.raise_for_status()
    task_instances = response.json().get("task_instances", [])
    tasks = []
    for task in task_instances:
        tasks.append(task["task_id"])
    return tasks

def check_dag_status(cur_task, dag_id, dag_run_id):
    state = "queued"
    st.toast(f"Current status({cur_task}): :green[{state}]", duration=2)  
    time.sleep(1.5)
    
    response = requests.get(
    f"{DAG_STATUS_CHECK_API}/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{cur_task}",
    auth=("airflow", "airflow"),
    timeout=10)
    response.raise_for_status()
    time.sleep(1.5)
    task_info = response.json()
    state = task_info.get("state", "failed")
    if state in ("failed", "skipped"):
        return state

    return state

def get_skip_reason(dag_id, dag_run_id, task_id, key="skip_reason"):
    url = f"{DAG_STATUS_CHECK_API}/{dag_id}/dagRuns/{dag_run_id}/taskInstances/extract_data/xcomEntries/{key}"
    resp = requests.get(url, auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD))
    if resp.status_code == 200:
        return resp.json().get("value")
    return None
