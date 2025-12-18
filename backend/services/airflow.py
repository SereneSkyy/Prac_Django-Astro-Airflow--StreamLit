# services/airflow.py
import requests
from requests.auth import HTTPBasicAuth

AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"
AIRFLOW_USERNAME = "airflow"
AIRFLOW_PASSWORD = "airflow"


def trigger_dag(dag_id, conf=None, run_id=None):
    url = f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns"

    payload = {}
    if conf:
        payload["conf"] = conf
    if run_id:
        payload["dag_run_id"] = run_id

    response = requests.post(
        url,
        json=payload,
        auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        timeout=10, # responding time, but the whole process time
    )

    response.raise_for_status()
    return response.json()
