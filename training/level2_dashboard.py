"""Separate localhost approval UI; does not use Flask or app.py."""
import argparse
import json
import os
import re
import secrets
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from level2_review import approve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs_training_v5_level2"
ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")

PAGE = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>plant2CO2｜Codex 審核中心</title>
<style>
:root{font-family:system-ui,"Microsoft JhengHei",sans-serif;color-scheme:dark}
body{background:#0d1828;color:#f1f6fd;margin:0}
header{padding:25px max(22px,calc((100% - 1100px)/2));background:#182b41}
main{max-width:1100px;margin:auto;padding:20px;display:grid;gap:18px}
section{background:#17283d;border:1px solid #314760;border-radius:14px;padding:22px}
h1{margin:0 0 8px;font-size:26px}h2{margin:0 0 12px;font-size:18px}
small,.hint{color:#bacbdc}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.metric{border:1px solid #35506b;border-radius:10px;padding:17px;background:#20364e}
.metric strong{display:block;font-size:24px;margin-top:9px}
.bar{height:10px;background:#34485c;margin-top:12px;border-radius:12px;overflow:hidden}
.fill{height:100%;background:#68bafa;width:0%}
.proposal{display:flex;gap:12px;align-items:flex-start;padding:15px;margin:12px 0;border:1px solid #3d536a;border-radius:10px}
.proposal input{width:22px;height:22px;accent-color:#61b4ec}
.proposal b{display:block}code{background:#2b415c;border-radius:4px;padding:4px;word-break:break-all}
button{padding:12px 20px;color:white;background:#288bdb;border:0;border-radius:9px;cursor:pointer}
button:disabled{background:#566375;cursor:default}
#feedback{white-space:pre-wrap}
</style></head><body>
<header><h1>plant2CO2 · Codex Level 2 審核中心</h1>
<small>獨立 localhost 管理頁面｜不占用 app.py 網站連接埠｜每 10 秒更新</small></header>
<main>
<section><h2>Codex Token 與剩餘額度</h2><div class="metrics">
<div class="metric">最近一回合 Token<strong id="turn">—</strong><small id="details"></small></div>
<div class="metric">最近工作階段累計<strong id="session">—</strong><small>不是帳號全部消耗</small></div>
<div class="metric" id="shortcard">短期額度剩餘<strong id="short">無資料</strong><div class="bar"><div id="shortbar" class="fill"></div></div></div>
<div class="metric" id="longcard">長期額度剩餘<strong id="long">無資料</strong><div class="bar"><div id="longbar" class="fill"></div></div></div>
</div><p class="hint" id="usageinfo">讀取本機 Codex 日誌中...</p></section>
<section><h2>待人工審核提案</h2>
<p class="hint">勾選想核准的提案，按下按鈕後只會建立 approved JSON，不會修改正在運行的訓練。</p>
<div id="proposals">讀取中...</div>
<button id="submit" disabled>確認核准勾選提案</button>
<button id="refresh">重新整理</button>
<p id="feedback"></p></section>
<section><h2>分析證據與最佳模型</h2>
<div class="metrics">
<div class="metric">最佳驗證 Epoch<strong id="bestepoch">—</strong></div>
<div class="metric">已讀混淆矩陣<strong id="cmcount">—</strong></div>
<div class="metric">固定影像覆蓋率 MAE<strong id="fixedmae">—</strong></div>
</div>
<p class="hint" id="evidencenote">等待 evidence.json，沒有人工標註及預測 mask 時不會計算 MAE。</p></section>
<section><h2>DeepLabV3+ 訓練狀態</h2>
<div class="metrics">
<div class="metric">最新 Epoch<strong id="epoch">—</strong></div>
<div class="metric">最佳植被 mIoU<strong id="miou">—</strong></div>
<div class="metric">警告數量<strong id="alerts">—</strong></div>
</div><p class="hint">由監控程式的 status.json 提供，並非直接控制訓練程序。</p></section>
<p class="hint">額度百分比是 Codex 本機日誌的最新可得快照，不保證與官方帳戶即時餘額完全同步。沒有資料時不推估。</p>
</main>
<script>
const csrf="__CSRF__";
const $=id=>document.getElementById(id);
const selected=new Set();
function text(id,value){$(id).textContent=value===undefined||value===null?"—":String(value)}
function rate(prefix,entry){
if(!entry){text(prefix,"無資料");$(prefix+"bar").style.width="0%";return}
text(prefix,(100-entry.used_percent).toFixed(1)+"%");
$(prefix+"bar").style.width=Math.max(0,100-entry.used_percent)+"%";
if(entry.window_minutes)$(prefix+"card").firstChild.textContent=entry.window_minutes+" 分鐘額度剩餘";
}
function makeProposal(p){
const label=document.createElement("label");label.className="proposal";
const checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.value=p.id;
checkbox.checked=selected.has(p.id);checkbox.onchange=()=>{
if(checkbox.checked)selected.add(p.id);else selected.delete(p.id);
$("submit").disabled=!selected.size;};
const wrapper=document.createElement("div");
const title=document.createElement("b");title.textContent=(p.kind==="review_task"?"離線評估審核 ":"訓練參數提案 ")+p.id+" ｜ Epoch "+p.base_epoch;
const reason=document.createElement("p");reason.textContent=p.reason||"無說明";
const code=document.createElement("code");code.textContent=p.kind==="review_task"?"僅核准調查項目，不修改訓練參數":JSON.stringify(p.changes);
const sub=document.createElement("p");sub.className="hint";sub.textContent="依據："+JSON.stringify(p.evidence||"未知")+(p.kind==="review_task"?"；核准後需人工完成離線評估":"；僅在下一次手動啟動/續訓時生效");
wrapper.append(title,reason,code,sub);label.append(checkbox,wrapper);return label;
}
async function refresh(){
try{
const response=await fetch("/api/state",{cache:"no-store"});if(!response.ok)throw Error("HTTP "+response.status);
const d=await response.json(),u=d.usage||{},t=u.tokens||{},m=d.monitor||{};
text("turn",t.last_total_tokens?.toLocaleString()||"—");
text("session",t.session_total_tokens?.toLocaleString()||"—");
text("details","輸入 "+(t.last_input_tokens??"—")+" / 輸出 "+(t.last_output_tokens??"—"));
rate("short",u.short);rate("long",u.long);text("usageinfo",u.message||"尚未取得 Codex 日誌");
text("epoch",m.latest_epoch);text("miou",typeof m.best_vegetation_miou==="number"?(m.best_vegetation_miou*100).toFixed(2)+"%":"—");
text("alerts",Array.isArray(m.alerts)?m.alerts.length:"—");
const e=d.evidence||{},h=e.history||{},c=e.confusion||{},fixed=e.fixed_images||{};
text("bestepoch",h.best_epoch);text("cmcount",(c.epochs||[]).length);
const best=fixed.aggregates?.best,latest=fixed.aggregates?.latest;
text("fixedmae",best&&latest?"最佳 "+best.coverage_mae_pp+" / 最後 "+latest.coverage_mae_pp+" pp":"無資料");
text("evidencenote",fixed.available?"固定影像逐張覆蓋率誤差（百分點），詳見 monitor/evidence.json":
"固定測試資料尚未齊全："+(fixed.reason||"未取得")+"；最佳模型資訊與混淆方向見 evidence.json");
const p=$("proposals");p.replaceChildren();
const existing=new Set((d.proposals||[]).map(x=>x.id));
for(const id of selected)if(!existing.has(id))selected.delete(id);
for(const item of d.proposals||[])p.append(makeProposal(item));
if(!p.children.length)p.textContent="目前沒有待核准的提案";
$("submit").disabled=!selected.size;
}catch(e){text("feedback","更新失敗："+e.message)}
}
$("refresh").onclick=refresh;
$("submit").onclick=async()=>{
const ids=[...selected];if(!ids.length)return;
if(!confirm("確認核准 "+ids.length+" 個提案或離線評估項目？只會寫入核准檔，不會開始訓練。"))return;
$("submit").disabled=true;
try{
const response=await fetch("/api/approve",{method:"POST",headers:{"Content-Type":"application/json","X-CSRF-Token":csrf},body:JSON.stringify({ids:ids})});
const result=await response.json();text("feedback",JSON.stringify(result,null,2));
if(response.ok)selected.clear();await refresh();
}catch(e){text("feedback","核准失敗："+e.message)}
};
refresh();setInterval(refresh,10000);
</script></body></html>"""


def codex_usage(home=None):
    """Only inspect token_count events, never credentials/auth.json."""
    base = Path(home) if home else Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions = base / "sessions"
    if not sessions.is_dir():
        return {"message": "沒有 Codex sessions 日誌；先在 Codex CLI 執行一次分析。"}
    paths = sorted(sessions.glob("*/*/*/rollout-*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    latest_info, latest_rate = None, None
    info_time, rate_time = "", ""
    for path in paths:
        try:
            with path.open("rb") as fd:
                fd.seek(0, 2)
                size = fd.tell()
                fd.seek(max(0, size - 4 * 1024 * 1024))
                raw = fd.read()
            if size > len(raw):
                raw = raw.split(b"\n", 1)[-1]
            for line in raw.splitlines():
                if b'"token_count"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except (UnicodeError, ValueError):
                    continue
                payload = event.get("payload", {})
                if event.get("type") != "event_msg" or not isinstance(payload, dict) or payload.get("type") != "token_count":
                    continue
                timestamp = event.get("timestamp", "")
                info = payload.get("info")
                if isinstance(info, dict) and info.get("total_token_usage") and timestamp >= info_time:
                    latest_info, info_time = info, timestamp
                limits = payload.get("rate_limits")
                if not isinstance(limits, dict) and isinstance(info, dict):
                    limits = info.get("rate_limits")
                if isinstance(limits, dict) and timestamp >= rate_time:
                    latest_rate, rate_time = limits, timestamp
        except (OSError, ValueError):
            continue
    info = latest_info or {}
    last = info.get("last_token_usage") or {}
    total = info.get("total_token_usage") or {}
    def numeric(v):
        return v if isinstance(v, (float, int)) and not isinstance(v, bool) else None
    windows = []
    for entry in (latest_rate or {}).values():
        if not isinstance(entry, dict) or numeric(entry.get("used_percent")) is None:
            continue
        used = min(100, max(0, float(entry["used_percent"])))
        windows.append({"used_percent": used, "window_minutes": numeric(entry.get("window_minutes")),
                        "resets_at": entry.get("resets_at")})
    windows.sort(key=lambda x: x.get("window_minutes") or 0)
    return {
        "tokens": {
            "last_total_tokens": numeric(last.get("total_tokens")),
            "last_input_tokens": numeric(last.get("input_tokens")),
            "last_output_tokens": numeric(last.get("output_tokens")),
            "session_total_tokens": numeric(total.get("total_tokens")),
        },
        "short": windows[0] if windows else None,
        "long": windows[-1] if len(windows) > 1 else None,
        "message": "本機快照；Token 更新：" + (info_time or "無")
                   + "，額度更新：" + (rate_time or "無；未提供剩餘百分比"),
    }


def proposals_in(folder):
    if not folder.is_dir():
        return []
    result = []
    for path in sorted(folder.glob("proposal_*.json")):
        id_ = path.stem[9:]
        if not ID_PATTERN.fullmatch(id_) or (folder / ("approved_" + id_ + ".json")).exists():
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("id") == id_ and item.get("status") == "pending":
                result.append({key: item.get(key) for key in ("id", "kind", "task_code", "base_epoch", "reason", "changes", "evidence")})
        except (OSError, ValueError):
            continue
    return result[-100:]


def monitor_state(output):
    try:
        status = json.loads((output / "monitor" / "status.json").read_text(encoding="utf-8"))
        return {key: status.get(key) for key in ("latest_epoch", "best_vegetation_miou", "alerts")}
    except (OSError, ValueError):
        return {}


def evidence_state(output):
    try:
        p = output / "monitor" / "evidence.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        return {
            "history": {k: d.get("history", {}).get(k) for k in
                        ("rows", "best_epoch", "best_vegetation_miou", "latest_valid_epoch")},
            "confusion": {"epochs": [
                {"epoch": x.get("epoch")} for x in d.get("confusion", {}).get("epochs", [])
            ]},
            "fixed_images": {"available": d.get("fixed_images", {}).get("available"),
                             "aggregates": d.get("fixed_images", {}).get("aggregates"),
                             "reason": d.get("fixed_images", {}).get("reason")},
        }
    except (OSError, ValueError, TypeError):
        return {}


def handler_factory(output, csrf):
    approvals = output / "approvals"

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, data, status=200, mime="application/json; charset=utf-8"):
            content = data.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/":
                self.send_body(PAGE.replace("__CSRF__", csrf),
                               mime="text/html; charset=utf-8")
            elif self.path == "/api/state":
                self.send_body(json.dumps({
                    "usage": codex_usage(),
                    "proposals": proposals_in(approvals),
                    "monitor": monitor_state(output),
                    "evidence": evidence_state(output),
                }, ensure_ascii=False))
            else:
                self.send_body('{"error":"not found"}', 404)

        def do_POST(self):
            if self.path != "/api/approve":
                return self.send_body('{"error":"not found"}', 404)
            origin = self.headers.get("Origin")
            if origin and origin != "http://127.0.0.1:" + str(self.server.server_port):
                return self.send_body('{"error":"forbidden origin"}', 403)
            if self.headers.get("X-CSRF-Token") != csrf:
                return self.send_body('{"error":"csrf failed"}', 403)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send_body('{"error":"expected json"}', 415)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 1 <= n <= 4096:
                    raise ValueError("request too large")
                ids = json.loads(self.rfile.read(n)).get("ids")
                if not isinstance(ids, list) or not 1 <= len(ids) <= 50 or len(ids) != len(set(ids)):
                    raise ValueError("需選擇 1–50 個不同方案")
                if not all(isinstance(x, str) and ID_PATTERN.fullmatch(x) for x in ids):
                    raise ValueError("方案編號格式錯誤")
            except (ValueError, TypeError) as exc:
                return self.send_body(json.dumps({"error": str(exc)}, ensure_ascii=False), 400)
            results = []
            for id_ in ids:
                try:
                    approve(id_, "APPROVE", approvals)
                    results.append({"id": id_, "result": "approved"})
                except (OSError, ValueError) as exc:
                    results.append({"id": id_, "result": "failed", "message": str(exc)})
            status = 200 if all(x["result"] == "approved" for x in results) else 409
            self.send_body(json.dumps({"results": results,
                "notice": "僅儲存核准 JSON；未修改或重新啟動訓練"}, ensure_ascii=False), status)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("連接埠必須在 1024–65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port),
                                 handler_factory(args.output, secrets.token_urlsafe(32)))
    url = "http://127.0.0.1:" + str(args.port) + "/"
    print("Codex Level 2 本機審核網頁：" + url)
    print("只監督與核准提案，不會影響 app.py。Ctrl+C 結束。")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n審核網頁已關閉，訓練不受影響。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
