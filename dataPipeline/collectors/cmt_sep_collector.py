import re
from psycopg2.extras import execute_values
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # makes results stable

def detect_language(text):
    try:
        lang = detect(text)
    except:
        return 'rne'
    
    if lang == 'ne':
        return 'ne'
    elif lang == 'en':
        return 'en'
    else:
        return 'rne'

def cmt_sep_collector(cursor, cmt: dict):
    # cmt is a dict of {comment_id: comment_text}
    rows = [(cid, detect_language(txt)) for cid, txt in cmt.items()]

    # Upsert logic: Insert language, or update it if the row exists
    execute_values(
        cursor,
        """
        INSERT INTO airflow.processed_comments (comment_id, language)
        VALUES %s
        ON CONFLICT (comment_id) DO UPDATE SET language = EXCLUDED.language
        """,
        rows,
        page_size=1000
    )