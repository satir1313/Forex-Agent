import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
import gradio as gr
import pandas as pd
from typing import List, Optional
from agent.agent_bridge import evaluate, list_strategies
from config.settings import SETTINGS

ALL_STRATS = list_strategies()
DEFAULT_TFS = list(SETTINGS.default_timeframes)

HELP = """
**Examples**
- `analyze EURUSD.a` — uses default TFs and confidence filter
- `analyze USDJPY.a M5,M15,H1` — explicit timeframes
- `analyze GBPUSD.a minconf=0.25 lookback=180` — tweak filters
- `analyze XAUUSD.a which="Trend Following, Breakout Trading, SMC: Liquidity Grab / Stop Hunt"`
"""

def parse_freeform(msg: str):
    """
    Tiny parser for freeform `analyze` commands.
    """
    msg = (msg or "").strip()
    if not msg.lower().startswith("analyze"):
        return None
    # Defaults
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

        # Explicit timeframes token like M5,M15
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

    # Chatbot now uses OpenAI-style message dicts
    chatbot = gr.Chatbot(height=400, type="messages")
    cmd = gr.Textbox(label="Command (e.g. analyze USDJPY.a M15,H1 minconf=0.25)", placeholder="Type 'analyze SYMBOL ...' and press Enter")

    table = gr.Dataframe(
        headers=["strategy","decision","confidence","timeframe","as_of_utc","extras"],
        wrap=True,
        interactive=False
    )
    rows_state = gr.State([])

    # Button-driven flow
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
        df, summary_md, _ = run(
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

    gr.Markdown("———")
    gr.Markdown(
        "_Safety note_: Trading actions are disabled in this console. If you want one-click orders, we can add a guarded "
        "“Place Order” panel that calls `mt5_place_order()` with confirmations."
    )

if __name__ == "__main__":
    demo.launch()
