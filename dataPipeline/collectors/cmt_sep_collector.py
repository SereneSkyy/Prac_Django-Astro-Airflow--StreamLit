import re
from psycopg2.extras import execute_values
from langdetect import detect, DetectorFactory
from schemas.etl_schema import insert_comment_lang

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
        insert_comment_lang,
        rows,
        page_size=1000
    )