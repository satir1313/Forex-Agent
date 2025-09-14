import os
import sys
import json
from datetime import datetime
from agent.agent_bridge import evaluate as agent_evaluate
from agent.agent_bridge import train_pipeline as agent_train_pipeline

# Ensure project root on path so imports work when running directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
import pandas as pd
from typing import List, Optional

from agent.agent_bridge import (
    evaluate,
    list_strategies,
    place_order,
    list_positions,
    close_position,
    chat as agent_chat,
)
from config.settings import SETTINGS

ALL_STRATS = list_strategies()
DEFAULT_TFS = list(SETTINGS.default_timeframes)
pos_refresh_counter = 0  # server-side counter for debugging refreshes

# Feature toggle for showing the training UI (set FX_ENABLE_TRAINING_UI=true in .env)
_FEATURE_TRAINING_UI = os.getenv("FX_ENABLE_TRAINING_UI", "").strip().lower() in ("1", "true", "yes", "on")

# ---------- Scoped CSS (layout only; no fundamental UI changes) ----------
CSS = """
/* Keep the top row strictly two columns (no wrapping into a second line) */
.fx-top { gap: 12px !important; display: flex; flex-wrap: nowrap !important; align-items: flex-start; }

/* Make left column narrower to free space for Model panel */
.fx-left  { min-width: 440px; max-width: 640px; }

/* Give Model panel more room */
.fx-right { min-width: 760px; }

/* Keep the small Model table tidy; prevent content from pushing layout */
.fx-small-table table { table-layout: fixed; width: 100%; }
.fx-small-table td, .fx-small-table th { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
"""

# ---------- Helpers ----------
def parse_freeform(msg: str):
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
        return None, f"❌ {res.get('error','Unknown error')}", None, None

    rows = res.get("rows", [])
    if not rows:
        msg = "No qualifying signals (filtered by confidence and no-trade). Try lowering min confidence or adding timeframes."
        if res.get("errors"):
            msg += f"\nEngine notes: {res['errors']}"
        return None, msg, None, None

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

    # small table: only ML predictions (Arvid v1)
    ml_rows = [r for r in rows if str(r.get("strategy")) == "Arvid v1"]
    ml_df = (pd.DataFrame(ml_rows)[["strategy","decision","confidence","timeframe","as_of_utc","extras"]]
             if ml_rows else pd.DataFrame(columns=["strategy","decision","confidence","timeframe","as_of_utc","extras"]))
    if not ml_df.empty:
        ml_df["confidence"] = (ml_df["confidence"]*100.0).round(2)

    md = "### Results\n"
    for r in rows[:10]:
        md += f"- **{r['strategy']}** → **{r['decision']}**, {round(r['confidence']*100,1)}%, {r['timeframe']}\n"
    if res.get("errors"):
        md += f"\n_Engine notes_: {res['errors']}"

    return df, "<br/>".join(summary) + "<br/><br/>" + md, rows, ml_rows

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
with gr.Blocks(title="FX Agent – Chat Console", css=CSS) as demo:
    gr.Markdown("# 💬 FX Agent – Chat Console")
    gr.Markdown("Chat with your MT5-backed analysis agent. Use the controls below **or** type commands like:\n\n")

    # Top control panel: strictly two columns (no wrap)
    with gr.Row(elem_classes=["fx-top"]):
        # LEFT COLUMN (narrower): symbol, timeframes, min conf, lookback
        with gr.Column(scale=2, min_width=440, elem_classes=["fx-left"]):
            symbol = gr.Textbox(label="Symbol", placeholder="e.g. USDJPY.a", scale=1, container=True)

            tfs = gr.CheckboxGroup(
                choices=["M1","M5","M15","M30","H1","H4","D1","W1"],
                value=list(SETTINGS.default_timeframes),
                label="Timeframes",
                scale=1
            )

            with gr.Row():
                min_conf = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=SETTINGS.default_min_confidence,
                    step=0.01,
                    label="Min Confidence",
                    scale=2
                )
                lookback = gr.Number(
                    value=SETTINGS.default_lookback_days,
                    precision=0,
                    label="Lookback Days",
                    scale=1
                )

        # RIGHT COLUMN (wider): Model panel
        with gr.Column(scale=5, min_width=760, elem_classes=["fx-right"]):
            if _FEATURE_TRAINING_UI:
                gr.Markdown("## 🧪 Model")
                train_btn = gr.Button("Train Model", variant="secondary")
                gr.Markdown("#### Model Predictions")
                model_pred_table = gr.Dataframe(
                    headers=["strategy","decision","confidence","timeframe","as_of_utc","extras"],
                    wrap=True,
                    interactive=False,
                    row_count=6,
                    elem_classes=["fx-small-table"]
                )
                model_pred_rows_state = gr.State([])
                model_train_log = gr.Textbox(label="Training Console", value="", max_lines=8, show_copy_button=True)
            else:
                gr.Markdown("### ")  # placeholder to keep layout balanced

    # Strategies filter (unchanged behavior)
    which = gr.CheckboxGroup(choices=ALL_STRATS, label="Strategies (leave empty for ALL)")

    run_btn = gr.Button("⚙️ Fetch & Analyze", variant="primary")

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
    table = gr.Dataframe(
        headers=["strategy","decision","confidence","timeframe","as_of_utc","extras"],
        wrap=True,
        interactive=False
    )
    rows_state = gr.State([])
    selected_row_idx = gr.State(value=None)
    selection_info = gr.Markdown("**Selected row**: _none_")

    # Chatbot + freeform command
    chatbot = gr.Chatbot(height=400, type="messages")
    cmd = gr.Textbox(
        label="Command (e.g. analyze USDJPY.a M15,H1 minconf=0.25)",
        placeholder="Type 'analyze SYMBOL ...' or chat freely and press Enter"
    )

    # -------- Handlers --------
    def on_click(symbol, tfs, which, min_conf, lookback, history):
        if not symbol:
            history = history + [
                {"role":"user","content":"Fetch & Analyze"},
                {"role":"assistant","content":"Please enter a symbol (e.g., USDJPY.a)."},
            ]
            return history, None, None, None, None
        df, summary_md, rows, ml_rows = run(symbol, tfs, which, min_conf, lookback)
        if df is None:
            history = history + [
                {"role":"user","content":f"Analyze {symbol}"},
                {"role":"assistant","content":summary_md},
            ]
            return history, None, None, None
        history = history + [
            {"role":"user","content":f"Analyze {symbol} ({', '.join(tfs)})"},
            {"role":"assistant","content":summary_md},
        ]
        import pandas as _pd
        ml_df = (_pd.DataFrame(ml_rows)[["strategy","decision","confidence","timeframe","as_of_utc","extras"]]
                 if ml_rows else _pd.DataFrame(columns=["strategy","decision","confidence","timeframe","as_of_utc","extras"]))
        if not ml_df.empty:
            ml_df["confidence"] = (ml_df["confidence"]*100.0).round(2)
        return history, df, rows, ml_df, ml_rows

    run_btn.click(
        fn=on_click,
        inputs=[symbol, tfs, which, min_conf, lookback, chatbot],
        outputs=[chatbot, table, rows_state, model_pred_table, model_pred_rows_state],
    )

    if _FEATURE_TRAINING_UI:
        def on_train(symbol, tfs, lookback):
            syms = [symbol] if symbol else []
            tfsu = [tf.upper() for tf in (tfs or [])]
            res = agent_train_pipeline(syms, tfsu, int(lookback))
            if not res.get("ok"):
                return f"❌ {res.get('error','Unknown error')}"
            return (res.get("log") or "").strip() or "✅ Done."

        train_btn.click(fn=on_train, inputs=[symbol, tfs, lookback], outputs=[model_train_log])

    def on_cmd2(cmd_text, history):
        parsed = parse_freeform(cmd_text)
        if parsed:
            df, summary_md, rows = run(
                symbol=parsed["symbol"],
                tfs=parsed["timeframes"],
                which=parsed["which"],
                min_conf=parsed["min_conf"],
                lookback=parsed["lookback"],
            )
            sym_update = gr.update(value=parsed.get("symbol")) if parsed.get("symbol") else gr.update()
            if df is None:
                return "", ((history or []) + [{"role":"user","content":cmd_text},{"role":"assistant","content":summary_md}]), None, None, sym_update
            return "", ((history or []) + [{"role":"user","content":cmd_text},{"role":"assistant","content":summary_md}]), df, rows, sym_update

        # freeform chat path
        try:
            existing_msgs = [{"role": m.get("role"), "content": m.get("content")} for m in (history or []) if isinstance(m, dict) and m.get("role") and m.get("content")]
            existing_msgs.append({"role":"user","content":cmd_text})
            res = agent_chat(messages=existing_msgs)
            if not res.get("ok"):
                return "", ((history or []) + [{"role":"user","content":cmd_text},{"role":"assistant","content":f"�?O Error: {res.get('error','unknown') }"}]), gr.update(), gr.update(), gr.update()
            reply = res.get("reply") or ""
            return "", ((history or []) + [{"role":"user","content":cmd_text},{"role":"assistant","content":reply}]), gr.update(), gr.update(), gr.update()
        except Exception as e:
            return "", ((history or []) + [{"role":"user","content":cmd_text},{"role":"assistant","content":f"�?O Exception: {str(e)}"}]), gr.update(), gr.update(), gr.update()

    cmd.submit(on_cmd2, inputs=[cmd, chatbot], outputs=[cmd, chatbot, table, rows_state, symbol])

    def on_table_select(evt: gr.SelectData, rows):
        try:
            ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        except Exception:
            ridx = None
        if ridx is None or rows is None or ridx >= len(rows):
            return None, "**Selected row**: _none_", gr.update()
        sel = rows[ridx]
        msg = f"**Selected row**: #{ridx} — {sel['strategy']} → **{sel['decision']}**, {round(sel['confidence']*100,1)}%, {sel['timeframe']}"
        default_side = sel["decision"] if sel["decision"] in ("buy","sell") else None
        return ridx, msg, gr.update(value=default_side)

    place_btn.click(
        lambda symbol, rows, ridx, side, volume, sl, tp, confirmed, history: on_place(symbol, rows, ridx, side, volume, sl, tp, confirmed, history),
        inputs=[symbol, rows_state, selected_row_idx, po_side, po_volume, po_sl, po_tp, po_confirm, chatbot],
        outputs=[chatbot],
    )

    def on_place(symbol, rows, ridx, side, volume, sl, tp, confirmed, history):
        if not confirmed:
            return history + [{"role":"assistant","content":"⚠️ Please tick **Confirm** to place the order."}]
        if not symbol:
            return history + [{"role":"assistant","content":"Please enter a **Symbol** first."}]
        if ridx is None or rows is None or ridx >= len(rows):
            return history + [{"role":"assistant","content":"Select a **row** in the table first."}]
        sel = rows[int(ridx)]
        side_final = (side or sel.get("decision") or "").lower()
        if side_final not in ("buy","sell"):
            return history + [{"role":"assistant","content":"Selected row is not a buy/sell signal. Choose a **Side** explicitly."}]
        vol = float(volume or SETTINGS.default_volume)
        sl_points = None if sl in ("", None) else int(sl)
        tp_points = None if tp in ("", None) else int(tp)
        res = place_order(symbol=symbol, side=side_final, volume=vol, sl_points=sl_points, tp_points=tp_points)
        if not res.get("ok"):
            return history + [{"role":"assistant","content":f"❌ Order failed: {res.get('error','unknown')}"}]
        summary = f"✅ Order sent: **{side_final.upper()} {symbol}** @ vol {vol}"
        if sl_points: summary += f", SL {sl_points}p"
        if tp_points: summary += f", TP {tp_points}p"
        return history + [{"role":"assistant","content":summary}]

    table.select(on_table_select, inputs=[rows_state], outputs=[selected_row_idx, selection_info, po_side])

    # ---- Positions (guarded)
    gr.Markdown("## Close Position (Guarded)")
    gr.Markdown("Refresh open positions for the given symbol, select one, and confirm to close it.")
    with gr.Row():
        refresh_pos_btn = gr.Button("Refresh Positions", elem_id="refresh-positions-btn")
        filter_by_symbol = gr.Checkbox(label="Filter by symbol", value=False)
        close_vol = gr.Number(value=None, precision=3, label="Close Volume (optional)")
        close_confirm = gr.Checkbox(label="I confirm closing the selected position.")
        close_btn = gr.Button("Close Position", variant="secondary")

    pos_table = gr.Dataframe(
        headers=["ticket","symbol","type","volume","price_open","profit"],
        wrap=True,
        interactive=False,
        row_count=7,
    )
    pos_rows_state = gr.State([])
    selected_pos_idx = gr.State(value=None)
    pos_selection_info = gr.Markdown("**Selected position**: _none_")

    def on_refresh_positions_silent(symbol, filter_by_symbol):
        global pos_refresh_counter
        pos_refresh_counter += 1
        res = list_positions(symbol=(symbol if (filter_by_symbol and symbol) else None))
        if not res.get("ok"):
            return None, [], None
        rows = res.get("positions", [])
        def _ptype(v):
            try:
                i = int(v)
                return "BUY" if i == 0 else ("SELL" if i == 1 else str(i))
            except Exception:
                return str(v)
        view = [{
            "ticket": r.get("ticket"),
            "symbol": r.get("symbol"),
            "type": _ptype(r.get("type")),
            "volume": r.get("volume"),
            "price_open": r.get("price_open"),
            "profit": r.get("profit"),
        } for r in rows]
        df = pd.DataFrame(view) if view else None
        return df, view, None

    refresh_pos_btn.click(on_refresh_positions_silent, inputs=[symbol, filter_by_symbol], outputs=[pos_table, pos_rows_state, selected_pos_idx])

    def on_pos_select(evt: gr.SelectData, rows):
        try:
            ridx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        except Exception:
            ridx = None
        if ridx is None or rows is None or ridx >= len(rows):
            return None, "**Selected position**: _none_"
        sel = rows[int(ridx)]
        msg = f"**Selected position**: ticket {sel.get('ticket')} {sel.get('symbol')} {sel.get('type')} vol {sel.get('volume')}"
        return ridx, msg

    pos_table.select(on_pos_select, inputs=[pos_rows_state], outputs=[selected_pos_idx, pos_selection_info])

    def on_close(symbol, rows, ridx, volume, confirmed, history):
        if not confirmed:
            return history + [{"role":"assistant","content":"Please tick **Confirm** to close the position."}]
        if ridx is None or rows is None or ridx >= len(rows):
            return history + [{"role":"assistant","content":"Select a **position** first and refresh if needed."}]
        sel = rows[int(ridx)]
        ticket = sel.get("ticket")
        if ticket is None:
            return history + [{"role":"assistant","content":"Selected row has no ticket."}]
        v = None if volume in (None, "") else float(volume)
        res = close_position(ticket=int(ticket), volume=v)
        if not res.get("ok"):
            return history + [{"role":"assistant","content":f"Close failed: {res.get('error','unknown')}"}]
        summary = f"Closed position {ticket} on {sel.get('symbol')}"
        if v:
            summary += f" (volume {v})"
        return history + [{"role":"assistant","content":summary}]

    close_btn.click(on_close, inputs=[symbol, pos_rows_state, selected_pos_idx, close_vol, close_confirm, chatbot], outputs=[chatbot])

    gr.Markdown("———")
    gr.Markdown("_Safety note_: Market orders are **live** on your logged-in MT5 terminal. Use small volumes while testing.")

    with gr.Row():
        export_btn = gr.Button("💾 Export CSV")
        save_btn = gr.Button("🧷 Save Session")

    with gr.Column(visible=False) as export_modal:
        export_modal_title = gr.Markdown("### Result")
        export_modal_msg = gr.Markdown("")
        export_modal_close = gr.Button("Close")

    export_btn.click(lambda rows, symbol, history: on_export(rows, symbol, history), inputs=[rows_state, symbol, chatbot], outputs=[chatbot, export_modal, export_modal_msg])
    save_btn.click(lambda history, rows, symbol: on_save(history, rows, symbol), inputs=[chatbot, rows_state, symbol], outputs=[chatbot, export_modal, export_modal_msg])

    def _hide_modal():
        return gr.update(visible=False), ""

    export_modal_close.click(fn=_hide_modal, inputs=[], outputs=[export_modal, export_modal_msg])

    # Auto-refresh positions (kept intact)
    auto_refresh_checkbox = gr.Checkbox(label="Auto-refresh positions", value=True)
    refresh_timer = gr.Timer(1.0, active=True)

    def toggle_timer(val: bool):
        return gr.update(active=val)

    auto_refresh_checkbox.change(fn=toggle_timer, inputs=auto_refresh_checkbox, outputs=refresh_timer)
    refresh_timer.tick(fn=on_refresh_positions_silent, inputs=[symbol, filter_by_symbol], outputs=[pos_table, pos_rows_state, selected_pos_idx])

if __name__ == "__main__":
    demo.launch()
