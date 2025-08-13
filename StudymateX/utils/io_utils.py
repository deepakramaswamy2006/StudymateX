from pathlib import Path
from datetime import datetime
from typing import List, Dict

def save_session_log(session_list: List[Dict], out_path: Path):
    lines = []
    for s in session_list:
        t = s.get("timestamp", datetime.utcnow().isoformat())
        q = s.get("question", "")
        a = s.get("answer", "")
        lines.append(f"---\nTime: {t}\nQ: {q}\nA: {a}\n")
    out_path.write_text("\n\n".join(lines), encoding="utf8")
    return out_path