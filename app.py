"""
Bull vs Bear: a multi-agent stock debate.

Agents:
  1. Document analyst - reads uploaded PDFs (concalls, results) once and writes a digest
  2. Bull analyst     - argues the stock is attractive
  3. Bear analyst     - argues the stock is risky
  4. Judge            - a portfolio manager who weighs the debate and returns a JSON verdict

Run:  streamlit run app.py
Educational project only. Not financial advice.
"""

import base64
import json
import os
import re

import anthropic
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bull vs Bear", page_icon="🐂", layout="wide")

AVATAR = {"bull": "🐂", "bear": "🐻"}
LABEL = {"bull": "Bull", "bear": "Bear"}


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        help="Or set the ANTHROPIC_API_KEY environment variable.",
    )
    model = st.text_input("Model", value="claude-sonnet-5-5")
    rounds = st.slider("Debate rounds", 1, 4, 3)
    max_words = st.slider("Max words per turn", 80, 300, 150, step=10)
    st.caption("Educational tool. Not financial advice.")


# ---------------------------------------------------------------- Market data
@st.cache_data(ttl=3600, show_spinner=False)
def get_fact_sheet(ticker: str):
    """Fetch price history and key ratios from Yahoo Finance and format a fact sheet."""
    t = yf.Ticker(ticker)
    hist = t.history(period="1y")
    if hist.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. "
            "For Indian stocks add .NS (NSE) or .BO (BSE), e.g. INFY.NS."
        )
    try:
        info = t.info or {}
    except Exception:
        info = {}

    close = hist["Close"]

    def change(days):
        if len(close) > days:
            return f"{(close.iloc[-1] / close.iloc[-days - 1] - 1) * 100:+.1f}%"
        return "n/a"

    def val(key, as_pct=False):
        v = info.get(key)
        if v is None:
            return "n/a"
        if as_pct and isinstance(v, (int, float)):
            return f"{v * 100:.1f}%"
        if isinstance(v, (int, float)):
            return f"{v:,.2f}"
        return str(v)

    name = info.get("longName") or info.get("shortName") or ticker
    sheet = f"""Company: {name} ({ticker})
Sector / industry: {val('sector')} / {val('industry')}
Currency: {val('currency')}
Last price: {close.iloc[-1]:,.2f}
Price change: 1 month {change(21)} | 6 months {change(126)} | 1 year {change(len(close) - 1)}
1-year range: {close.min():,.2f} to {close.max():,.2f}
Market cap: {val('marketCap')}
Trailing P/E: {val('trailingPE')} | Forward P/E: {val('forwardPE')} | Price/Book: {val('priceToBook')}
Revenue growth (yoy): {val('revenueGrowth', True)} | Earnings growth (yoy): {val('earningsGrowth', True)}
Profit margin: {val('profitMargins', True)} | Return on equity: {val('returnOnEquity', True)}
Debt/Equity (as reported by Yahoo): {val('debtToEquity')}"""
    return name, sheet, close


# ---------------------------------------------------------------- Agents
def digest_pdfs(client, files, company):
    """Document analyst: read all PDFs once and produce a shared digest."""
    content = []
    for f in files:
        content.append({"type": "text", "text": f"Document: {f.name}"})
        content.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(f.getvalue()).decode(),
            },
        })
    content.append({"type": "text", "text": f"""You are a sell-side research analyst covering {company}.
Write a neutral digest of these documents for an investment debate. Use these sections:
1. Financial performance (revenue, profit, margins vs last year and last quarter, with numbers)
2. Management guidance and outlook
3. Positives highlighted by management
4. Risks, weaknesses, or concerns (including anything management avoided or deflected)
5. Notable analyst Q&A moments
Cite the source for each point as [document name, p.X]. Do not add opinions or outside information."""})

    msg = client.messages.create(
        model=model, max_tokens=2500,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def debater_system(side, company, evidence, focus):
    stance = (
        "argue that this stock is an attractive investment over the next 1-3 years"
        if side == "bull"
        else "argue that this stock is risky or unattractive over the next 1-3 years"
    )
    return f"""You are the {LABEL[side].upper()} analyst in a structured debate about {company}.
Your job: {stance}.
{f'The debate question is: {focus}' if focus else ''}

Rules:
- Use ONLY the evidence below. Never invent numbers, events, or quotes.
- Cite your evidence briefly, e.g. [Facts] or [document name, p.X].
- After the opening, directly rebut your opponent's strongest points.
- Stay under {max_words} words. Plain paragraphs, no headers.

EVIDENCE:
{evidence}"""


JUDGE_SYSTEM = """You are an experienced, impartial portfolio manager judging a Bull vs Bear debate.
Judge the quality of the arguments, not your own view of the stock.
Reward claims backed by the evidence and strong rebuttals. Penalise any claim not supported by the evidence.
Do not favour a side for speaking first or writing more.

Respond with ONLY a JSON object, no other text, with these keys:
{
  "winner": "bull" | "bear" | "tie",
  "bull_score": 0-10,
  "bear_score": 0-10,
  "stance": "Bullish" | "Cautiously bullish" | "Neutral" | "Cautiously bearish" | "Bearish",
  "confidence": 1-10,
  "strongest_bull_point": "...",
  "strongest_bear_point": "...",
  "key_risk_to_watch": "...",
  "unsupported_claims": ["..."],
  "reasoning": "3-5 sentences"
}"""


def stream_turn(client, system, user):
    """Yield text chunks so Streamlit can show the argument as it is written."""
    with client.messages.stream(
        model=model,
        max_tokens=max_words * 2 + 200,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def round_name(i, total):
    if i == 0:
        return "Opening"
    if i == total - 1:
        return "Closing"
    return f"Rebuttal {i}" if total > 3 else "Rebuttal"


def transcript_text(turns):
    return "\n\n".join(f"[{t['round']}] {LABEL[t['side']].upper()}: {t['text']}" for t in turns)


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- Rendering helpers
def show_market(sheet, close):
    left, right = st.columns([3, 2])
    left.line_chart(close, height=260)
    right.text(sheet)


def show_verdict(verdict, raw):
    st.subheader("Verdict")
    if not verdict:
        st.warning("The judge didn't return valid JSON. Raw response:")
        st.write(raw)
        return
    winner = verdict.get("winner", "tie")
    if winner in AVATAR:
        st.success(f"{AVATAR[winner]} {LABEL[winner]} wins the debate")
    else:
        st.info("The debate is a tie")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bull score", f"{verdict.get('bull_score', '?')}/10")
    c2.metric("Bear score", f"{verdict.get('bear_score', '?')}/10")
    c3.metric("Stance", verdict.get("stance", "?"))
    c4.metric("Confidence", f"{verdict.get('confidence', '?')}/10")
    st.markdown(f"**Strongest bull point:** {verdict.get('strongest_bull_point', '')}")
    st.markdown(f"**Strongest bear point:** {verdict.get('strongest_bear_point', '')}")
    st.markdown(f"**Key risk to watch:** {verdict.get('key_risk_to_watch', '')}")
    st.markdown(f"**Reasoning:** {verdict.get('reasoning', '')}")
    unsupported = verdict.get("unsupported_claims") or []
    if unsupported:
        with st.expander(f"Unsupported claims flagged by the judge ({len(unsupported)})"):
            for claim in unsupported:
                st.markdown(f"- {claim}")


def download_button(result):
    md = f"# Bull vs Bear: {result['company']}\n\n## Market facts\n```\n{result['sheet']}\n```\n\n"
    if result["digest"]:
        md += f"## Document digest\n{result['digest']}\n\n"
    md += "## Debate\n\n" + "\n\n".join(
        f"**{LABEL[t['side']]} ({t['round']}):** {t['text']}" for t in result["turns"]
    )
    md += f"\n\n## Verdict\n```json\n{json.dumps(result['verdict'], indent=2) if result['verdict'] else result['raw']}\n```\n"
    st.download_button(
        "Download transcript",
        md,
        file_name=f"bull_bear_{result['ticker']}.md",
        mime="text/markdown",
    )


def replay(result):
    """Redraw a finished debate (Streamlit reruns the script on every click)."""
    st.subheader(f"Market data: {result['company']}")
    show_market(result["sheet"], result["close"])
    if result["digest"]:
        with st.expander("Document digest"):
            st.markdown(result["digest"])
    st.subheader("The debate")
    for t in result["turns"]:
        with st.chat_message(t["side"], avatar=AVATAR[t["side"]]):
            st.markdown(f"**{LABEL[t['side']]}: {t['round']}**")
            st.markdown(t["text"])
    show_verdict(result["verdict"], result["raw"])
    download_button(result)


# ---------------------------------------------------------------- Main page
st.title("🐂 Bull vs Bear 🐻")
st.write("Two AI analysts debate a stock using live market data and your documents. A third AI judges.")

col1, col2 = st.columns([1, 2])
ticker = col1.text_input("Ticker", value="INFY.NS", help="NSE: add .NS, BSE: add .BO, US: plain ticker").strip().upper()
pdfs = col2.file_uploader(
    "Concall transcripts or results (optional)", type="pdf", accept_multiple_files=True
)
focus = st.text_input(
    "Debate question (optional)",
    placeholder="Is this stock worth buying at the current valuation for a 2-year hold?",
)
run = st.button("Start debate", type="primary")

if run:
    if not api_key:
        st.error("Add your Anthropic API key in the sidebar to start.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)

    # 1. Market data
    try:
        with st.spinner(f"Fetching market data for {ticker}..."):
            company, sheet, close = get_fact_sheet(ticker)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.subheader(f"Market data: {company}")
    show_market(sheet, close)

    # 2. Document analyst
    digest = ""
    if pdfs:
        try:
            with st.spinner(f"Reading {len(pdfs)} document(s)..."):
                digest = digest_pdfs(client, pdfs, company)
            with st.expander("Document digest", expanded=False):
                st.markdown(digest)
        except anthropic.APIError as e:
            st.warning(f"Couldn't read the PDFs, continuing with market data only. ({e})")

    evidence = f"=== MARKET FACTS ===\n{sheet}"
    if digest:
        evidence += f"\n\n=== DOCUMENT DIGEST ===\n{digest}"

    # 3. Debate
    st.subheader("The debate")
    turns = []
    try:
        for i in range(rounds):
            rname = round_name(i, rounds)
            for side in ("bull", "bear"):
                so_far = transcript_text(turns) or "(No arguments yet. You speak first.)"
                user = f"DEBATE SO FAR:\n{so_far}\n\nGive your {rname.lower()} statement now."
                with st.chat_message(side, avatar=AVATAR[side]):
                    st.markdown(f"**{LABEL[side]}: {rname}**")
                    text = st.write_stream(
                        stream_turn(client, debater_system(side, company, evidence, focus), user)
                    )
                turns.append({"round": rname, "side": side, "text": text})
    except anthropic.APIError as e:
        st.error(f"The API returned an error: {e}")
        st.stop()

    # 4. Judge
    with st.spinner("The judge is deliberating..."):
        msg = client.messages.create(
            model=model,
            max_tokens=1500,
            system=JUDGE_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"EVIDENCE:\n{evidence}\n\nDEBATE TRANSCRIPT:\n{transcript_text(turns)}",
            }],
        )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    verdict = parse_json(raw)
    show_verdict(verdict, raw)

    result = {
        "ticker": ticker, "company": company, "sheet": sheet, "close": close,
        "digest": digest, "turns": turns, "verdict": verdict, "raw": raw,
    }
    st.session_state["last"] = result
    download_button(result)

elif "last" in st.session_state:
    replay(st.session_state["last"])
