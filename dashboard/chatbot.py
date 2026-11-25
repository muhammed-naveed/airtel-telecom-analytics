"""AI Data Analyst: built-in aggregate answers, plus optional Gemini SQL with strict read-only checks."""
import json
import os
import re
import sqlite3
import time
import urllib.request
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st

TABLE = "customer_features"
BLOCKED_COLUMNS = {"customer_id"}
COLUMNS = """activation_date tenure_days tenure_years observed_days circle_name region plan_name plan_type
price_inr validity_days monthly_gb quota_basis age_band gender device_type os acquisition_channel connection_type
ab_group is_churned churn_date days_to_churn usage_days total_data_mb total_voice_min total_sms roaming_days
mb_per_active_day mb_per_observed_day last_usage_date recency_days recharge_count recharge_total_6m
actual_arpu_6m stated_arpu arpu_gap complaint_count unresolved_count resolved_count avg_resolution_days""".split()
MAX_ROWS = 500

# Completed years of tenure (text labels are kept as they are)
TENURE_YEARS = "CASE WHEN typeof(tenure_years) = 'text' THEN tenure_years ELSE CAST(tenure_years AS INTEGER) END"
# Built-in questions work without an API key: question -> grouping column (None = overall)
SUGGESTED = {
    "Overall churn": None,
    "Churn by circle": "circle_name",
    "Prepaid vs postpaid": "connection_type",
    "Churn by tenure": TENURE_YEARS,
    "Unresolved complaints": "CASE WHEN unresolved_count > 0 THEN 'Has unresolved' ELSE 'None unresolved' END",
}


# ---------------------------------------------------------------- safety
def validate_sql(sql):
    sql = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^select\b", sql):
        raise ValueError("Only SELECT statements are allowed.")
    if ";" in sql or "--" in sql or "/*" in sql:
        raise ValueError("Multiple statements and comments are blocked.")
    if re.search(r"(?i)\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|trigger|with|union)\b", sql):
        raise ValueError("Blocked SQL keyword.")
    if not re.search(r"(?i)\b(count|sum|avg|min|max|total)\s*\(", sql):
        raise ValueError("Only aggregated queries are allowed.")
    limit = re.search(r"(?i)\blimit\s+(\d+)\s*$", sql)
    if limit and int(limit.group(1)) > MAX_ROWS:
        raise ValueError(f"Limit cannot exceed {MAX_ROWS}.")
    return sql if limit else sql + " LIMIT 100"


def authorizer(action, arg1, arg2, db_name, source):
    """SQLite-level allow-list: this is the real guard; the text checks above are a first filter."""
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_DENY if arg2 == "load_extension" else sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if source == TABLE:  # reads made internally when customer_features is a view
            return sqlite3.SQLITE_OK
        if arg1 == TABLE and arg2 not in BLOCKED_COLUMNS:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def execute_readonly(db, sql, params=()):
    uri = Path(db).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        deadline = time.monotonic() + 10
        con.set_progress_handler(lambda: time.monotonic() > deadline, 10_000)
        con.set_authorizer(authorizer)
        cursor = con.execute(sql, params)
        return pd.DataFrame(cursor.fetchmany(MAX_ROWS), columns=[c[0] for c in cursor.description])


# ---------------------------------------------------------------- answers
def built_in_answer(db, question, connection, region, circle):
    where, params = [], []
    for column, value in (("connection_type", connection), ("region", region), ("circle_name", circle)):
        if value != "All":
            where.append(f"{column} = ?")
            params.append(value)
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    group = SUGGESTED[question]
    select = (f"{group} AS segment, " if group else "") + (
        "COUNT(*) AS subscribers, SUM(is_churned) AS churned, "
        "ROUND(100.0 * SUM(is_churned) / COUNT(*), 2) AS churn_rate_pct")
    tail = ""
    if group:
        tail = " GROUP BY segment ORDER BY " + ("segment" if group == TENURE_YEARS else "churn_rate_pct DESC")
    sql = f"SELECT {select} FROM {TABLE}{clause}{tail} LIMIT {MAX_ROWS}"
    result = execute_readonly(db, sql, params)
    if result.empty or not result.subscribers.sum():
        return "No subscribers match this selection.", sql, result
    if group:
        top = result.loc[result.churn_rate_pct.idxmax()]
        text = (f"The highest churn rate is in **{result.at[result.churn_rate_pct.idxmax(), 'segment']}**: {top.churn_rate_pct:.2f}% "
                f"({int(top.churned):,} of {int(top.subscribers):,}). Check group sizes before comparing.")
    else:
        row = result.iloc[0]
        text = (f"Churn is **{row.churn_rate_pct:.2f}%** ({int(row.churned):,} of {int(row.subscribers):,}). "
                "The supplied national reference is 23.75%; it is not a target.")
    return text, sql, result


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except FileNotFoundError:
        key = None
    return key or os.getenv("GEMINI_API_KEY")


def gemini_call(prompt, api_key):
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                         "generationConfig": {"temperature": 0.1}}).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    with urllib.request.urlopen(request, timeout=40) as response:
        data = json.loads(response.read().decode())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def ai_answer(db, question, connection, region, circle, api_key):
    prompt = f"""You are an Airtel retention data analyst. Write exactly one SQLite SELECT query for the question.
Only table: {TABLE}. Columns: {", ".join(COLUMNS)}.
Rules: aggregated results only; no customer identifiers; no joins, subqueries, CTEs or unions;
churn rate = SUM(is_churned)*1.0/COUNT(*); end with LIMIT 100.
Current filters (apply unless the user asks for all data): connection_type={connection}; region={region}; circle_name={circle}.
Return only SQL.
Question: {question}"""
    sql = validate_sql(gemini_call(prompt, api_key).replace("```sql", "").replace("```", ""))
    result = execute_readonly(db, sql)
    text = gemini_call(
        "Explain this query result briefly and simply. Use only the numbers shown. Make no causal claims. "
        "23.75% is a national churn reference, not a target.\n"
        f"Question: {question}\nSQL: {sql}\nResult:\n{result.to_string(index=False)}", api_key)
    return text, sql, result


# ---------------------------------------------------------------- UI
def render_chatbot(db, connection_filter="All", region_filter="All", selected_circle="All"):
    api_key = get_api_key()
    context = (connection_filter, region_filter, selected_circle)
    st.caption(f"Read-only · Connection: {context[0]} · Region: {context[1]} · Circle: {context[2]}")
    question, quick = None, False
    for column, label in zip(st.columns(len(SUGGESTED)), SUGGESTED):
        if column.button(label, key=f"suggest_{label}", width="stretch"):
            question, quick = label, True
    with st.form("ask_form", clear_on_submit=True, border=False):
        typed = st.text_input("Your question", placeholder="Ask about retention data…",
                              disabled=not api_key)
        if st.form_submit_button("Ask", disabled=not api_key) and typed.strip():
            question, quick = typed.strip(), False
    if not api_key:
        st.caption("Buttons above work offline. Set GEMINI_API_KEY to type your own questions.")
    if question:
        try:
            with st.spinner("Querying the database…"):
                answer = (built_in_answer(db, question, *context) if quick
                          else ai_answer(db, question, *context, api_key))
            st.session_state.last_answer = (question, context, *answer)
        except Exception as error:  # network, API, validation or SQL failure
            st.session_state.pop("last_answer", None)
            st.error(f"I couldn't safely complete that query: {error}")
    if "last_answer" in st.session_state:
        asked, used, text, sql, result = st.session_state.last_answer
        if used == context:
            st.markdown(f"**{asked}**\n\n{text}")
            st.dataframe(result, width="stretch", hide_index=True, height=220)
            with st.expander("Validated SQL"):
                st.code(sql, language="sql")
