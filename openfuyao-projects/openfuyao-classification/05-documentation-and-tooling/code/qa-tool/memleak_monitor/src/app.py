import os
import logging
import sqlite3
from datetime import datetime
from typing import Any

from flask import Flask, request, jsonify, g

DB_PATH = os.getenv("DB_PATH", "/metrics.db")
ALERT_PATH = os.getenv("ALERT_PATH", "/alert.log")
PORT = os.getenv("APP_PORT", "5000")
app = Flask(__name__)

alert_logger = logging.getLogger("core")
alert_logger.setLevel(logging.INFO)
alert_logger.propagate = False
fh = logging.FileHandler(ALERT_PATH, encoding="utf-8")
fh.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(message)s'))
alert_logger.addHandler(fh)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc) -> None:
    db = g.pop("db", None)
    if db:
        db.close()


def insert_event(table: str, payload: dict):
    db = get_db()

    keys = ",".join(payload.keys())
    placeholders = ",".join(["?"] * len(payload))

    sql = f"INSERT INTO {table} ({keys}) VALUES ({placeholders})"
    db.execute(sql, list(payload.values()))
    db.commit()


def log_alert(alert_type: str, message: str) -> None:
    alert_logger.warning(f"[{alert_type.capitalize()} Alert]: {message}")


def generate_oom_alert(data: dict) -> str:
    return (
        "OOMkilled event detected in "
        f"node={data['node']}, "
        f"namespace={data['namespace']}, "
        f"pod={data['pod']}, "
        f"container={data['container']}"
    )


@app.route("/memory/stat", methods=["POST"])
def memory_stat() -> tuple[Any, int]:
    data = request.get_json(force=True)
    required = ["timestamp", "node", "mem_workingset", "mem_available", "mem_rss"]
    
    is_valid, error_msg = validate_required_fields(data, required)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    insert_event("memory", data)
    return jsonify({"status": "ok"}), 200


@app.route("/memory/alert", methods=["POST"])
def memory_alert() -> tuple[Any, int]:
    data = request.get_json(force=True)
    required = ["message"]
    
    is_valid, error_msg = validate_required_fields(data, required)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    message = data.get("message", "")
    log_alert("memory", message)
    return jsonify({"status": "ok"}), 200


@app.route("/oom", methods=["POST"])
def oom_api() -> tuple[Any, int]:
    data = request.get_json(force=True)
    required = ["timestamp", "node", "namespace", "pod", "container"]

    is_valid, error_msg = validate_required_fields(data, required)
    if not is_valid:
        return jsonify({"error": error_msg}), 400
    
    message = generate_oom_alert(data)
    log_alert("oomkilled", message)

    return jsonify({"status": "ok"}), 200


def validate_required_fields(data: dict, required_fields: list[str]) -> tuple[bool, str]:
    """验证请求数据中是否包含所有必需字段"""
    for field in required_fields:
        if field not in data:
            return False, f"missing field: {field}"
    return True, ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(PORT))
