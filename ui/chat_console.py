import os
import json
from datetime import datetime
import gradio as gr
import pandas as pd
from typing import List, Optional

from agent.agent_bridge import evaluate, list_strategies, place_order
from config.settings import SETTINGS

ALL_STRATS = list_strategies()
DEFAULT_TFS = list(SETTINGS.default_timeframes)

HELP = """
**Examples**
- `analyze EURUSD.a` — uses default TFs and confidence filter  
- `analyze USDJPY.a M5,M15,H1` — explicit timeframes  
- `analyze GBPUSD.a minconf=0.25 lookback=180` — tweak filters  
- `analyze XAUUSD.a which="Trend Following, Breakout Trading, Fair Value Gap (FVG) Strategy"`
"""

def parse_freeform(msg: str):
    """
    Tiny parser for freeform `analyze` commands.
    """
    msg = (msg or "").strip()
    if not msg.lower().startswith("analyze"):
        return None
    symbol: Optional[str] = None
    tfs: List[str] = list(DEFAULT_TFS)
    minconf = SETTINGS.default_min_confidence
    lookback = SETTINGS.default_lookback_days
    which = None

    try:
        parts = msg.split()
        if len(parts) >= 2:
            symbol = parts[1]

        if "minconf=" in msg:
            minconf = float(msg.split("minconf=")[1].split()[0].replace("%", ""))
            if minconf > 1:
                minconf = minconf / 100.0
        if "lookback=" in msg:
            lookback = int(msg.split("lookback=")[1].split()[0])
        if '"' in msg and "which=" in msg:
            after = msg.split("which=")[1]
            quote_str = after[after.index('"') + 1 : after.index('"', after.index('"') + 1)]
            which = [s.strip() for s in quote_str.split(",") if s.strip()]

        for token in parts[2:]:
            up = token.upper()
            if up[0] in ("M", "H", "D", "W") or up.startswith("MN"):
                tfs = [x.strip().upper() for x in token.split(",") if x.strip()]
                break
    except Exception:
        pass

    return dict(symbol=symbol, timeframes=tfs, min_conf=minconf, lookback=lookback, which=which)

def run(symbol, tfs, which, min_conf, lookback):
    res = evaluate(
        symbol=symbol,
        timeframes=tfs,
        which=(None if not which else which),
        lookback_days=int(lookback),
        min_confidence=float(min_conf),
    )
    if not res.get("ok"):
        return None, f"❌ {res.get('error','Unknown error')}", None

    rows = res.get("rows", [])
    if not rows:
        msg = "No qualifying signals (filtered by confidence and no-trade). Try lowering min confidence or adding timeframes."
        if res.get("errors"):
            msg += f"\nEngine notes: {res['errors']}"
        return None, msg, None

    top_buy = next((r for r in rows if r["decision"] == "buy"), None)
    top_sell = next((r for r in rows if r["decision"] == "sell"), None)
    summary = []
    if top_buy:
        summary.append(f"**Top Buy**: {top_buy['strategy']} @ {top_buy['timeframe']} ({round(top_buy['confidence']*100,1)}%)")
    if top_sell:
        summary.append(f"**Top Sell**: {top_sell['strategy']} @ {top_sell['timeframe']} ({round(top_sell['confidence']*100,1)}%)")
    if not summary:
        summary.append("_Signals found but none categorized as buy/sell after sorting_")

    df = pd.DataFrame(rows)[["strategy","decision","confidence","timeframe","as_of_utc","extras"]]
    df["confidence"] = (df["confidence"]*100.0).round(2)

    md = "### Results\n"
    for r in rows[:10]:
        md += f"- **{r['strategy']}** → **{r['decision']}**, {round(r['confidence']*100,1)}%, {r['timeframe']}\n"
    if res.get("errors"):
        md += f"\n_Engine notes_: {res['errors']}"

    return df, "<br/>".join(summary) + "<br/><br/>" + md, rows

# ---------- Export helpers ----------
def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def export_csv(rows, symbol):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fn = f"{symbol or 'results'}_{_timestamp()}.csv"
    path = os.path.join(SETTINGS.export_dir, fn)
    df.to_csv(path, index=False)
    return path

def save_session(history, rows, symbol):
    payload = {
        "saved_at_utc": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "messages": history or [],
        "results": rows or [],
        "version": "console-v1",
    }
    fn = f"session_{symbol or 'NA'}_{_timestamp()}.json"
    path = os.path.join(SETTINGS.sessions_dir, fn)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path

# ---------- UI ----------
with gr.Blocks(title="FX Agent – Chat Console") as demo:
    gr.Markdown("# 💬 FX Agent – Chat Console")
    gr.Markdown(
        "Chat with your MT5-backed analysis agent. Use the controls below **or** type commands like:\n\n" + HELP
    )

    with gr.Row():
        symbol = gr.Textbox(label="Symbol", placeholder="e.g. USDJPY.a", scale=2)
        tfs = gr.CheckboxGroup(
            choices=["M1","M5","M15","M30","H1","H4","D1","W1"],
            value=list(SETTINGS.default_timeframes),
            label="Timeframes",
            scale=2
        )
        min_conf = gr.Slider(0, 1, value=SETTINGS.default_min_confidence, step=0.01, label="Min Confidence")
        lookback = gr.Number(value=SETTINGS.default_lookback_days, precision=0, label="Lookback Days")

    which = gr.CheckboxGroup(choices=ALL_STRATS, label="Strategies (leave empty for ALL)")

    run_btn = gr.Button("⚙️ Fetch & Analyze", variant="primary")

    # Chatbot uses OpenAI-style message dicts
    chatbot = gr.Chatbot(height=400, type="messages")
    cmd = gr.Textbox(label="Command (e.g. analyze USDJPY.a M15,H1 minconf=0.25)", placeholder="Type 'analyze SYMBOL ...' and press Enter")

    table = gr.Dataframe(
        headers=["strategy","decision","confidence","timeframe","as_of_utc","extras"],
        wrap=True,
        interactive=False
    )
    rows_state = gr.State([])
    selected_row_idx = gr.State(value=None)  # keep track of user selection
    selection_info = gr.Markdown("**Selected row**: _none_")

    # ---- Export & Session controls
    with gr.Row():
        export_btn = gr.Button("💾 Export CSV")
        export_file = gr.File(label="Last export", interactive=False)
        save_btn = gr.Button("🧷 Save Session")
        session_file = gr.File(label="Saved session", interactive=False)

    # ---- Place Order panel (guarded)
    gr.Markdown("## ⚠️ Place Order (Guarded)")
    gr.Markdown(
        "Select a **row in the table** above. Then choose side & volume. "
        "You must tick **Confirm** to send a live market order via MT5."
    )
    with gr.Row():
        po_side = gr.Radio(choices=["buy","sell"], label="Side")
        po_volume = gr.Number(value=SETTINGS.default_volume, precision=3, label="Volume (lots)")
        po_sl = gr.Number(value=SETTINGS.default_sl_points, precision=0, label="SL (points)", interactive=True)
        po_tp = gr.Number(value=SETTINGS.default_tp_points, precision=0, label="TP (points)", interactive=True)
        po_confirm = gr.Checkbox(label="I understand this places a REAL market order in MT5.")

    place_btn = gr.Button("🚀 Place Order", variant="secondary")

    # Button-driven Fetch & Analyze
    def on_click(symbol, tfs, which, min_conf, lookback, history):
        if not symbol:
            history = history + [
                {"role":"user","content":"Fetch & Analyze"},
                {"role":"assistant","content":"Please enter a symbol (e.g., USDJPY.a)."},
            ]
            return history, None, None
        df, summary_md, rows = run(symbol, tfs, which, min_conf, lookback)
        if df is None:
            history = history + [
                {"role":"user","content":f"Analyze {symbol}"},
                {"role":"assistant","content":summary_md},
            ]
            return history, None, None
        history = history + [
            {"role":"user","content":f"Analyze {symbol} ({', '.join(tfs)})"},
            {"role":"assistant","content":summary_md},
        ]
        return history, df, rows

    run_btn.click(
        fn=on_click,
        inputs=[symbol, tfs, which, min_conf, lookback, chatbot],
        outputs=[chatbot, table, rows_state],
    )

    # Freeform command box
    def on_cmd(cmd_text, history):
        parsed = parse_freeform(cmd_text)
        if not parsed:
            return "", history + [
                {"role":"user","content":cmd_text},
                {"role":"assistant","content":"Type `analyze SYMBOL` or use the controls above."},
            ]
        df, summary_md, rows = run(
            symbol=parsed["symbol"],
            tfs=parsed["timeframes"],
            which=parsed["which"],
            min_conf=parsed["min_conf"],
            lookback=parsed["lookback"],
        )
        if df is None:
            return "", history + [
                {"role":"user","content":cmd_text},
                {"role":"assistant","content":summary_md},
            ]
        return "", history + [
            {"role":"user","content":cmd_text},
            {"role":"assistant","content":summary_md},
        ]

    cmd.submit(on_cmd, inputs=[cmd, chatbot], outputs=[cmd, chatbot])

    # Table row selection → update selected_row_idx, selection_info and side
    def on_table_select(evt: gr.SelectData, rows):
        try:
            # evt.index can be (row, col) for Dataframe; take row
            ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        except Exception:
            ridx = None
        if ridx is None or rows is None or ridx >= len(rows):
            return None, "**Selected row**: _none_", gr.update()
        sel = rows[ridx]
        msg = f"**Selected row**: #{ridx} — {sel['strategy']} → **{sel['decision']}**, {round(sel['confidence']*100,1)}%, {sel['timeframe']}"
        # Default side from row decision (if valid)
        default_side = sel["decision"] if sel["decision"] in ("buy","sell") else None
        return ridx, msg, gr.update(value=default_side)

    table.select(
        on_table_select,
        inputs=[rows_state],
        outputs=[selected_row_idx, selection_info, po_side],
    )

    # Export CSV
    def on_export(rows, symbol, history):
        path = export_csv(rows, symbol)
        if not path:
            history = history + [{"role":"assistant","content":"Nothing to export yet."}]
            return history, None
        history = history + [{"role":"assistant","content":f"Exported CSV to `{path}`"}]
        return history, path

    export_btn.click(on_export, inputs=[rows_state, symbol, chatbot], outputs=[chatbot, export_file])

    # Save Session
    def on_save(history, rows, symbol):
        path = save_session(history, rows, symbol)
        history = history + [{"role":"assistant","content":f"Saved session to `{path}`"}]
        return history, path

    save_btn.click(on_save, inputs=[chatbot, rows_state, symbol], outputs=[chatbot, session_file])

    # Place Order (guarded)
    def on_place(symbol, rows, ridx, side, volume, sl, tp, confirmed, history):
        if not confirmed:
            history = history + [{"role":"assistant","content":"⚠️ Please tick **Confirm** to place the order."}]
            return history
        if symbol is None or symbol == "":
            history = history + [{"role":"assistant","content":"Please enter a **Symbol** first."}]
            return history
        if ridx is None or rows is None or ridx >= len(rows):
            history = history + [{"role":"assistant","content":"Select a **row** in the table first."}]
            return history
        sel = rows[int(ridx)]
        # If side not chosen, use row decision (must be buy/sell)
        side_final = (side or sel.get("decision") or "").lower()
        if side_final not in ("buy","sell"):
            history = history + [{"role":"assistant","content":"Selected row is not a buy/sell signal. Choose a **Side** explicitly."}]
            return history
        vol = float(volume or SETTINGS.default_volume)
        sl_points = None if sl in ("", None) else int(sl)
        tp_points = None if tp in ("", None) else int(tp)

        res = place_order(symbol=symbol, side=side_final, volume=vol, sl_points=sl_points, tp_points=tp_points)
        if not res.get("ok"):
            history = history + [{"role":"assistant","content":f"❌ Order failed: {res.get('error','unknown')}"}]
            return history
        # Compose summary
        summary = f"✅ Order sent: **{side_final.upper()} {symbol}** @ vol {vol}"
        if sl_points: summary += f", SL {sl_points}p"
        if tp_points: summary += f", TP {tp_points}p"
        history = history + [{"role":"assistant","content":summary}]
        return history

    place_btn.click(
        on_place,
        inputs=[symbol, rows_state, selected_row_idx, po_side, po_volume, po_sl, po_tp, po_confirm, chatbot],
        outputs=[chatbot],
    )

    gr.Markdown("———")
    gr.Markdown(
        "_Safety note_: Market orders are **live** on your logged-in MT5 terminal. Use small volumes while testing."
    )

if __name__ == "__main__":
    demo.launch()
