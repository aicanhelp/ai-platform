import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DB_PATH = os.getenv('DB_PATH', '/data/metrics.db')
AGGREGATE_URL = os.getenv('AGGREGATE_URL', 'http://localhost:8080/memory/alert')
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '300'))
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '60'))
MIN_SAMPLES = 5


def check_linear_slope(data, threshold_ratio=0.001):
    if len(data) < 2:
        return False
    
    total_mem = data[-1]['mem_workingset'] + data[-1]['mem_rss'] + data[-1]['mem_available']

    timestamps = [datetime.fromtimestamp(d['timestamp']) for d in data]
    mem_values = [d['mem_workingset'] + d['mem_rss'] for d in data]
    base_time = timestamps[0]
    time_offsets = [(t - base_time).total_seconds() / 60 for t in timestamps]
    x = np.array(time_offsets)
    y = np.array(mem_values)
    matrix = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(matrix, y, rcond=None)[0]
    
    return slope > (total_mem * threshold_ratio)


def check_continuous_growth(data, threshold=5):
    if len(data) < threshold:
        return False
    
    recent_data = data[-threshold:]
    mem_values = [d['mem_workingset'] + d['mem_rss'] for d in recent_data]
    
    for i in range(1, len(mem_values)):
        if mem_values[i] < mem_values[i - 1]:
            return False
    
    return True


def check_over_threshold(data, threshold=0.8):
    if not data:
        return False

    current_usage = data[-1]['usage_ratio']        
    return current_usage >= threshold


class MemoryAnalyzer:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.conn = None
        self.node_list = self.get_all_nodes()
        self.check_funcs = {
            "slope_threshold": check_linear_slope,
            "continuous_growth": check_continuous_growth,
            "over_threshold": check_over_threshold,
        }

    def connect_db(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            return True
        except sqlite3.Error:
            return False
    
    def close_db(self):
        if self.conn:
            self.conn.close()
    
    def get_all_nodes(self):
        if not self.conn and not self.connect_db():
            return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DISTINCT node FROM memory")
            return [row[0] for row in cursor.fetchall()]
        except sqlite3.Error:
            return []
    
    def get_memory_trend_data(self, node, window_minutes=WINDOW_SIZE):
        if not self.conn and not self.connect_db():
            return []
        
        try:
            cursor = self.conn.cursor()
            since_time = int(
                (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).timestamp()
            )
            
            query = """
                SELECT node, timestamp, mem_workingset, mem_available, mem_rss,
                      CAST(mem_workingset + mem_rss AS REAL) / (mem_workingset + mem_rss + mem_available) as usage_ratio
                FROM memory 
                WHERE node = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """
            cursor.execute(query, (node, since_time))
            rows = cursor.fetchall()
            logger.info(f"Retrieved {len(rows)} memory records for node={node}")
            
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []
    
    def check_memory_leak(self, node):
        data = self.get_memory_trend_data(node)
        if len(data) < MIN_SAMPLES:
            return None
        
        conditions = []
        for check_name, check_func in self.check_funcs.items():
            if check_func(data):
                conditions.append(check_name)
        
        if conditions:
            start_mem = data[0]['mem_workingset'] + data[0]['mem_rss']
            end_mem = data[-1]['mem_workingset'] + data[-1]['mem_rss']
            change_percent = ((end_mem - start_mem) / start_mem) * 100
            start_gb = start_mem / 1024 / 1024 / 1024
            end_gb = end_mem / 1024 / 1024 / 1024
            message = (
                "[NODE MEMORY LEAK SUSPECTED],"
                f"Node: {node},"
                f"Window: {WINDOW_SIZE}min,"
                f"Usage: {start_gb:.1f}GB → {end_gb:.1f}GB ({change_percent:+.1f}%)"
            )
            return message
        
        return None
    
    @staticmethod
    def send_alert(message):
        try:
            response = requests.post(AGGREGATE_URL, json={'message': message}, timeout=5)
            logger.info(f"Memory alert sent, url={AGGREGATE_URL}, status={response.status_code}")
        except requests.RequestException:
            logger.exception("Error sending memory alert")

    def analyze_all_nodes(self):
        for node in self.node_list:
            alert_message = self.check_memory_leak(node)
            if alert_message:
                self.send_alert(alert_message)


if __name__ == '__main__':
    analyzer = MemoryAnalyzer()
    while True:
        logger.info("Starting memory analysis...")
        analyzer.analyze_all_nodes()
        time.sleep(CHECK_INTERVAL)
