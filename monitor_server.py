"""
Minimal monitoring API server for the SMS Spam Detection dashboard.
Run: python monitor_server.py
Then open: monitor.html in a browser.
"""
import json, os, re, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BASE = Path(__file__).parent
LOG_FILE      = BASE / "logs" / "sms_spam.log"
FEED_LOG_FILE = BASE / "logs" / "data_feed.log"
METRICS_FILE  = BASE / "results" / "metrics" / "svm_results.json"
RAW_DATA      = BASE / "data" / "raw" / "spam.csv"
PROCESSED     = BASE / "data" / "processed" / "processed.pkl"
MODEL_FILE    = BASE / "models" / "svm.pkl"

ANSI = re.compile(r'\x1b\[[0-9;]*m')

def _clean(line): return ANSI.sub('', line).strip()

def get_logs():
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()
    parsed = []
    for raw in lines[-200:]:
        line = _clean(raw)
        if not line: continue
        level = 'INFO'
        for lvl in ('ERROR','WARNING','DEBUG','INFO'):
            if lvl in line: level = lvl; break
        parsed.append({'raw': line, 'level': level})
    return parsed

def get_metrics():
    if not METRICS_FILE.exists():
        return {}
    return json.loads(METRICS_FILE.read_text())

def get_pipeline_stats():
    logs = [_clean(l) for l in LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines() if l.strip()]

    # pipeline runs — count "Pipeline complete" lines
    runs = [l for l in logs if 'Pipeline complete' in l]
    last_run = runs[-1] if runs else None
    exec_time = None
    if last_run:
        m = re.search(r'(\d+\.?\d*)s', last_run)
        if m: exec_time = float(m.group(1))

    # data ingestion
    downloads = [l for l in logs if 'Download complete' in l]
    download_ok = len(downloads)
    download_fail = len([l for l in logs if 'ERROR' in l and 'download' in l.lower()])

    # dataset stats from log
    loaded = next((l for l in reversed(logs) if 'Loaded' in l and 'messages' in l), None)
    total_msgs = spam_count = ham_count = None
    if loaded:
        m = re.search(r'(\d+) messages.*spam=(\d+).*ham=(\d+)', loaded)
        if m: total_msgs, spam_count, ham_count = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # predictions from log
    preds = [l for l in logs if 'Prediction:' in l]
    spam_preds = [l for l in preds if 'SPAM' in l]
    ham_preds  = [l for l in preds if ': HAM' in l]

    # confidence values
    confidences = []
    for p in preds:
        m = re.search(r'spam=(\d+\.?\d*)%', p)
        if m: confidences.append(float(m.group(1)))

    avg_conf = round(sum(confidences)/len(confidences), 1) if confidences else None

    # data freshness
    freshness_secs = None
    if RAW_DATA.exists():
        freshness_secs = int(time.time() - RAW_DATA.stat().st_mtime)

    # errors / warnings
    errors   = [_clean(l) for l in LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines() if 'ERROR' in l]
    warnings = [_clean(l) for l in LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines() if 'WARNING' in l]

    # model file age
    model_age_secs = None
    if MODEL_FILE.exists():
        model_age_secs = int(time.time() - MODEL_FILE.stat().st_mtime)

    # simple drift proxy: spam ratio in recent predictions vs training baseline
    baseline_spam_ratio = 0.134  # 747/5572 from dataset
    recent_spam_ratio = len(spam_preds) / len(preds) if preds else None
    drift_score = None
    if recent_spam_ratio is not None:
        drift_score = round(abs(recent_spam_ratio - baseline_spam_ratio), 3)

    return {
        'pipeline_runs': len(runs),
        'last_run_time': last_run.split('  ')[0] if last_run else None,
        'exec_time_secs': exec_time,
        'download_success': download_ok,
        'download_fail': download_fail,
        'total_messages': total_msgs,
        'spam_count': spam_count,
        'ham_count': ham_count,
        'total_predictions': len(preds),
        'spam_predictions': len(spam_preds),
        'ham_predictions': len(ham_preds),
        'avg_confidence': avg_conf,
        'confidences': confidences[-50:],
        'data_freshness_secs': freshness_secs,
        'model_age_secs': model_age_secs,
        'error_count': len(errors),
        'warning_count': len(warnings),
        'recent_errors': errors[-5:],
        'recent_warnings': warnings[-5:],
        'drift_score': drift_score,
        'baseline_spam_ratio': baseline_spam_ratio,
        'recent_spam_ratio': round(recent_spam_ratio, 3) if recent_spam_ratio is not None else None,
    }

def get_mlflow_runs():
    mlruns = BASE / 'mlruns'
    runs = []
    for exp_dir in mlruns.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in ('.trash', 'models'): continue
        exp_meta = exp_dir / 'meta.yaml'
        exp_name = exp_dir.name
        if exp_meta.exists():
            for line in exp_meta.read_text().splitlines():
                if line.startswith('name:'): exp_name = line.split(':', 1)[1].strip(); break
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == 'models': continue
            meta = run_dir / 'meta.yaml'
            if not meta.exists(): continue
            info = {'run_id': run_dir.name, 'experiment': exp_name, 'params': {}, 'metrics': {}, 'tags': {}}
            for line in meta.read_text().splitlines():
                for key in ('run_name', 'status', 'start_time', 'end_time', 'user_id'):
                    if line.startswith(key + ':'):
                        info[key] = line.split(':', 1)[1].strip()
            status_map = {'1': 'RUNNING', '2': 'SCHEDULED', '3': 'FINISHED', '4': 'FAILED', '5': 'KILLED'}
            info['status'] = status_map.get(str(info.get('status', '')), info.get('status', 'UNKNOWN'))
            try:
                s, e = int(info.get('start_time', 0)), int(info.get('end_time', 0))
                info['duration_s'] = round((e - s) / 1000, 1) if e > s else None
                info['start_time'] = s
            except: pass
            # metric files: each line is "timestamp value step" — take last non-empty line
            for mfile in (run_dir / 'metrics').iterdir() if (run_dir / 'metrics').exists() else []:
                try:
                    last = [l for l in mfile.read_text().splitlines() if l.strip()][-1]
                    info['metrics'][mfile.name] = round(float(last.split()[1]), 4)
                except: pass
            for pfile in (run_dir / 'params').iterdir() if (run_dir / 'params').exists() else []:
                try: info['params'][pfile.name] = pfile.read_text().strip()
                except: pass
            # tags — mlflow.runName overrides run_name from meta.yaml
            for tfile in (run_dir / 'tags').iterdir() if (run_dir / 'tags').exists() else []:
                try: info['tags'][tfile.name] = tfile.read_text().strip()
                except: pass
            if 'mlflow.runName' in info['tags']:
                info['run_name'] = info['tags']['mlflow.runName']
            # only keep SVM runs
            run_name_lower = info.get('run_name', '').lower()
            model_type = info['tags'].get('model_type', '').lower()
            if not run_name_lower.startswith('svm') and model_type not in ('svm', ''):
                continue
            if run_name_lower and not run_name_lower.startswith('svm') and model_type == '':
                continue
            runs.append(info)
    runs.sort(key=lambda r: r.get('start_time', 0), reverse=True)
    return runs


def get_dvc_data():
    import yaml as _yaml
    result = {'stages': {}, 'params': {}, 'metrics': {}}
    dvc_yaml  = BASE / 'dvc.yaml'
    dvc_lock  = BASE / 'dvc.lock'
    params_yaml = BASE / 'params.yaml'

    if dvc_yaml.exists():
        doc = _yaml.safe_load(dvc_yaml.read_text())
        for name, body in (doc.get('stages') or {}).items():
            result['stages'][name] = {
                'cmd':  body.get('cmd', ''),
                'deps': [d if isinstance(d, str) else list(d.keys())[0] for d in (body.get('deps') or [])],
                'outs': [o if isinstance(o, str) else list(o.keys())[0] for o in (body.get('outs') or [])],
            }

    if dvc_lock.exists():
        lock = _yaml.safe_load(dvc_lock.read_text())
        for name, body in (lock.get('stages') or {}).items():
            if name not in result['stages']: continue
            # deps with real md5 + size from lock
            result['stages'][name]['locked_deps'] = [
                {'path': d.get('path', ''), 'md5': d.get('md5', ''), 'size': d.get('size', 0)}
                for d in (body.get('deps') or []) if isinstance(d, dict)
            ]
            # outs with real md5 + size from lock
            result['stages'][name]['locked_outs'] = [
                {'path': o.get('path', ''), 'md5': o.get('md5', ''), 'size': o.get('size', 0)}
                for o in (body.get('outs') or []) if isinstance(o, dict)
            ]
            # locked params snapshot
            result['stages'][name]['locked_params'] = body.get('params', {})

    if params_yaml.exists():
        raw = _yaml.safe_load(params_yaml.read_text())
        # exclude mlflow section — it's tracking config, not model hyperparams
        result['params'] = {k: v for k, v in raw.items() if k != 'mlflow'}

    if (BASE / 'results' / 'metrics' / 'svm_results.json').exists():
        result['metrics'] = json.loads((BASE / 'results' / 'metrics' / 'svm_results.json').read_text())
    return result


def get_feed_status():
    """Return recent data-feed log lines and row count of raw CSV."""
    lines = []
    if FEED_LOG_FILE.exists():
        raw_lines = FEED_LOG_FILE.read_text(encoding='utf-8', errors='ignore').splitlines()
        lines = [_clean(l) for l in raw_lines[-100:] if l.strip()]
    row_count = 0
    if RAW_DATA.exists():
        with open(RAW_DATA, encoding='utf-8', errors='ignore') as f:
            row_count = sum(1 for _ in f) - 1
    last_append = None
    for l in reversed(lines):
        if 'Appended' in l:
            last_append = l
            break
    return {
        'csv_rows': row_count,
        'last_append': last_append,
        'recent_feed_logs': lines[-20:],
        'feed_log_exists': FEED_LOG_FILE.exists(),
    }


ROUTES = {
    '/api/logs':    lambda: get_logs(),
    '/api/metrics': lambda: get_metrics(),
    '/api/stats':   lambda: get_pipeline_stats(),
    '/api/mlflow':  lambda: get_mlflow_runs(),
    '/api/dvc':     lambda: get_dvc_data(),
    '/api/feed':    lambda: get_feed_status(),
}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ROUTES:
            data = json.dumps(ROUTES[path]()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
        elif path == '/':
            html = (BASE / 'monitor.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_): pass  # silence request logs

if __name__ == '__main__':
    port = int(os.environ.get('MONITOR_PORT', 8765))
    host = os.environ.get('MONITOR_HOST', '0.0.0.0')
    print(f"Dashboard → http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()
