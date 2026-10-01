"""
Bull vs Bear: a multi-agent stock debate.

Agents:
  1. Document analyst - reads uploaded PDFs (concalls, results, presentations) once and writes a digest
  2. Bull analyst     - argues the stock is attractive
  3. Bear analyst     - argues the stock is risky
  4. Judge            - a portfolio manager who weighs the debate and returns a JSON verdict

Works with Google Gemini (free tier) or Anthropic Claude.
Run:  ./venv/bin/python -m streamlit run app.py
Educational project only. Not financial advice.
"""

import base64
import json
import math
import os
import re
import time
from datetime import date

import altair as alt
import anthropic
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()  # reads API keys from a .env file in this folder, if there is one
st.set_page_config(page_title="Bull vs Bear", page_icon="🐂", layout="wide")

AVATAR = {"bull": "🐂", "bear": "🐻"}
LABEL = {"bull": "Bull", "bear": "Bear"}
COLOR = {"bull": "green", "bear": "red"}
HEX = {"bull": "#16a34a", "bear": "#dc2626"}

MODELS = {
    "Google Gemini (free tier)": ["gemini-flash-lite-latest", "gemini-flash-latest"],
    "Anthropic Claude": ["claude-sonnet-5-5", "claude-haiku-4-5"],
}
KEY_ENV = {"Google Gemini (free tier)": "GEMINI_API_KEY", "Anthropic Claude": "ANTHROPIC_API_KEY"}
SESSION_LIMIT = 3  # demo debates per visitor (per browser session) on the shared key


def secret(name):
    """Read a value from Streamlit Cloud secrets, or None when running locally without them."""
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def shared_key_for(provider):
    """Return (key, where it came from). 'cloud' keys are shared with visitors, so they get limits."""
    name = KEY_ENV[provider]
    if secret(name):
        return secret(name), "cloud"
    if os.getenv(name):
        return os.getenv(name), "local"
    return "", None


@st.cache_resource
def demo_usage():
    """One counter shared by every visitor of the hosted app. Resets each day (and when the app restarts)."""
    return {"date": None, "count": 0}


def demo_debates_left():
    usage = demo_usage()
    today = date.today().isoformat()
    if usage["date"] != today:
        usage["date"], usage["count"] = today, 0
    daily_limit = int(secret("DEMO_DAILY_LIMIT") or 30)
    return max(daily_limit - usage["count"], 0)


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("Settings")
    provider = st.radio("AI provider", list(MODELS))
    GEMINI = provider.startswith("Google")
    shared_key, key_source = shared_key_for(provider)
    key_name = "Gemini API key" if GEMINI else "Anthropic API key"
    # The shared key is never put into the text box, so visitors can't reveal it with the eye icon
    user_key = st.text_input(
        f"{key_name} (optional)" if shared_key else key_name, type="password",
        help="Get a free Gemini key at aistudio.google.com. Your key is only used for this session and isn't saved.",
    ).strip()
    api_key = user_key or shared_key
    demo_mode = not user_key and key_source == "cloud"

    if demo_mode:
        left = min(demo_debates_left(), SESSION_LIMIT - st.session_state.get("demo_runs", 0))
        st.caption(f"Using the free demo key: {max(left, 0)} debate(s) left for you. Add your own key for unlimited use.")
    elif not user_key and key_source == "local":
        st.caption("Using the key from your .env file.")

    model = st.selectbox(
        "Model", MODELS[provider][:1] if demo_mode else MODELS[provider],
        help="Lite models are faster and less likely to be busy on the free tier.",
    )

    st.subheader("Debate")
    rounds = st.slider("Rounds", 1, 4, 3)
    max_words = st.slider("Max words per turn", 80, 300, 150, step=10)

    with st.expander("Advanced"):
        custom_model = st.text_input("Custom model name", placeholder="Leave empty to use the model above")
        if custom_model.strip():
            model = custom_model.strip()

    st.caption("Educational tool. Not financial advice.")


# ---------------------------------------------------------------- LLM layer (one place for both providers)
def make_client():
    return genai.Client(api_key=api_key) if GEMINI else anthropic.Anthropic(api_key=api_key)


def is_rate_limit(e):
    msg = str(e)
    busy = ["429", "RESOURCE_EXHAUSTED", "rate_limit", "503", "UNAVAILABLE", "overloaded"]
    return any(word in msg for word in busy)


def friendly_error(e):
    if is_rate_limit(e):
        return "The AI provider is busy or you've hit the rate limit. Wait a minute and try again, or pick the Lite model."
    if "API key" in str(e) or "401" in str(e) or "authentication" in str(e).lower():
        return "Your API key wasn't accepted. Check it in the sidebar."
    return f"The AI provider returned an error: {e}"


def generate(client, prompt, system=None, max_tokens=1500, pdfs=None):
    """One complete (non-streamed) response. pdfs = list of (file name, bytes)."""
    for attempt in range(3):
        try:
            if GEMINI:
                contents = []
                for name, data in pdfs or []:
                    contents.append(f"Document file: {name}")
                    contents.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
                contents.append(prompt)
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    # Flash models "think" before answering, and that uses output tokens, so leave room
                    config=types.GenerateContentConfig(
                        system_instruction=system, max_output_tokens=max(max_tokens, 8192)
                    ),
                )
                return resp.text or ""

            content = []
            for name, data in pdfs or []:
                content.append({"type": "text", "text": f"Document file: {name}"})
                content.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf",
                               "data": base64.standard_b64encode(data).decode()},
                })
            content.append({"type": "text", "text": prompt})
            kwargs = {"system": system} if system else {}
            msg = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}], **kwargs,
            )
            return "".join(b.text for b in msg.content if b.type == "text")
        except Exception as e:
            if is_rate_limit(e) and attempt < 2:
                time.sleep(30)
                continue
            raise


def stream(client, system, prompt, max_tokens):
    """Yield text chunks so Streamlit can show each argument as it is written."""
    if GEMINI:
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=max(max_tokens, 8192)
            ),
        ):
            if chunk.text:
                yield chunk.text
    else:
        with client.messages.stream(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as s:
            for text in s.text_stream:
                yield text


# ---------------------------------------------------------------- Number formatting
def clean(v):
    """Turn missing or NaN values into None so they show as n/a."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def indian_commas(n):
    """1234567 -> 12,34,567"""
    s = str(int(round(abs(n))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts) + "," + tail
    return ("-" if n < 0 else "") + s


def fmt_money(v, currency):
    if not isinstance(v, (int, float)):
        return "n/a"
    if currency == "INR":
        return f"₹{indian_commas(v / 1e7)} Cr"
    return f"{currency} {v / 1e9:,.1f}B"


def fmt_price(v, currency):
    if not isinstance(v, (int, float)):
        return "n/a"
    return f"{'₹' if currency == 'INR' else ''}{v:,.2f}"


def fmt_pct(v):
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def fmt_change(v):
    return f"{v * 100:+.1f}%" if isinstance(v, (int, float)) else "n/a"


def fmt_num(v, decimals=1):
    return f"{v:,.{decimals}f}" if isinstance(v, (int, float)) else "n/a"


# ---------------------------------------------------------------- Market data
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(ticker: str):
    """Two years of prices (so the 200-day average covers the whole chart) plus key ratios."""
    t = yf.Ticker(ticker)
    hist = t.history(period="2y")
    if hist.empty:
        raise ValueError(
            f"No price data found for '{ticker}'. "
            "For Indian stocks add .NS (NSE) or .BO (BSE), e.g. INFY.NS."
        )
    try:
        info = t.info or {}
    except Exception:
        info = {}

    df = pd.DataFrame({"Price": hist["Close"]})
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["50-day avg"] = df["Price"].rolling(50).mean()
    df["200-day avg"] = df["Price"].rolling(200).mean()
    year = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]

    def change(days):
        past = df[df.index <= df.index[-1] - pd.Timedelta(days=days)]["Price"]
        return None if past.empty else float(df["Price"].iloc[-1] / past.iloc[-1] - 1)

    facts = {
        "name": info.get("longName") or info.get("shortName") or ticker,
        "ticker": ticker,
        "currency": info.get("currency") or "",
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "price": float(year["Price"].iloc[-1]),
        "chg_1m": change(30),
        "chg_6m": change(182),
        "chg_1y": change(365),
        "low": float(year["Price"].min()),
        "high": float(year["Price"].max()),
        "ma50": clean(float(df["50-day avg"].iloc[-1])),
        "ma200": clean(float(df["200-day avg"].iloc[-1])),
        "market_cap": info.get("marketCap"),
        "pe": info.get("trailingPE"),
        "fpe": info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "rev_growth": info.get("revenueGrowth"),
        "earn_growth": info.get("earningsGrowth"),
        "margin": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"),
        "de": info.get("debtToEquity"),
    }
    facts = {k: clean(v) for k, v in facts.items()}
    return facts, year


def fact_sheet_text(f):
    c = f["currency"]
    return f"""Company: {f['name']} ({f['ticker']})
Sector / industry: {f['sector'] or 'n/a'} / {f['industry'] or 'n/a'}
Last price: {fmt_price(f['price'], c)}
Price change: 1 month {fmt_change(f['chg_1m'])} | 6 months {fmt_change(f['chg_6m'])} | 1 year {fmt_change(f['chg_1y'])}
1-year range: {fmt_price(f['low'], c)} to {fmt_price(f['high'], c)}
50-day average: {fmt_price(f['ma50'], c)} | 200-day average: {fmt_price(f['ma200'], c)}
Market cap: {fmt_money(f['market_cap'], c)}
Trailing P/E: {fmt_num(f['pe'], 2)} | Forward P/E: {fmt_num(f['fpe'], 2)} | Price/Book: {fmt_num(f['pb'], 2)}
Revenue growth (yoy): {fmt_pct(f['rev_growth'])} | Earnings growth (yoy): {fmt_pct(f['earn_growth'])}
Profit margin: {fmt_pct(f['margin'])} | Return on equity: {fmt_pct(f['roe'])}
Debt/Equity (as reported by Yahoo; not meaningful for banks): {fmt_num(f['de'], 2)}"""


# ---------------------------------------------------------------- Agents
def digest_pdfs(client, files, company, context):
    """Document analyst: read all PDFs once and produce a shared digest."""
    prompt = f"""You are a sell-side research analyst covering {company}.
{f'Note from the user about these documents: {context}' if context else ''}

The files may have random names. Identify each document from its content (its type and period,
e.g. "Q1 FY27 investor presentation") and use that short name in every citation, like [Q1 FY27 presentation, p.4].

Start with a short "Documents" list naming each one. Then write a neutral digest for an investment debate with these sections:
1. Financial performance (revenue, profit, margins vs last year and last quarter, with numbers)
2. Management guidance and outlook
3. Positives highlighted by management
4. Risks, weaknesses, or concerns (including anything management avoided or deflected)
5. Notable analyst Q&A moments (skip if there is no transcript)
6. Change over time: if the documents cover more than one period, say what improved, what got worse, and whether earlier guidance was met

Do not add opinions or outside information."""
    return generate(client, prompt, max_tokens=3000, pdfs=[(f.name, f.getvalue()) for f in files])


def debater_system(side, company, evidence, context):
    stance = (
        "argue that this stock is an attractive investment over the next 1-3 years"
        if side == "bull"
        else "argue that this stock is risky or unattractive over the next 1-3 years"
    )
    return f"""You are the {LABEL[side].upper()} analyst in a structured debate about {company}.
Your job: {stance}.
{f'Context or question from the user: {context}' if context else ''}

Rules:
- Use ONLY the evidence below. Never invent numbers, events, or quotes.
- Cite your evidence briefly, e.g. [Facts] or [Q1 FY27 presentation, p.4].
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


def to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------- UI pieces
def show_altair(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # older Streamlit versions
        st.altair_chart(chart, use_container_width=True)


def price_chart(df):
    long = (
        df.reset_index(names="Date")
        .melt("Date", var_name="Series", value_name="Value")
        .dropna()
    )
    lo, hi = long["Value"].min(), long["Value"].max()
    pad = (hi - lo) * 0.05 or 1
    series = ["Price", "50-day avg", "200-day avg"]
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Value:Q", title=None, scale=alt.Scale(domain=[lo - pad, hi + pad])),
            color=alt.Color(
                "Series:N", title=None,
                scale=alt.Scale(domain=series, range=["#0f172a", "#2563eb", "#f59e0b"]),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.condition(alt.datum.Series == "Price", alt.value([1, 0]), alt.value([5, 3])),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"), alt.Tooltip("Value:Q", title="Price", format=",.2f")],
        )
        .properties(height=320)
    )


def show_market(f, df, sheet):
    c = f["currency"]
    st.subheader(f["name"])
    st.caption(f"{f['ticker']} | {f['sector'] or 'n/a'} / {f['industry'] or 'n/a'}")

    row = st.columns(4)
    row[0].metric(
        "Price", fmt_price(f["price"], c),
        delta=f"{fmt_change(f['chg_1m'])} in 1 month" if f["chg_1m"] is not None else None,
    )
    row[1].metric(
        "1-year change", fmt_change(f["chg_1y"]),
        help=f"1-year range: {fmt_price(f['low'], c)} to {fmt_price(f['high'], c)}",
    )
    row[2].metric("Market cap", fmt_money(f["market_cap"], c))
    row[3].metric("P/E (trailing)", fmt_num(f["pe"]), help=f"Forward P/E: {fmt_num(f['fpe'])}")

    row = st.columns(4)
    row[0].metric("Price / book", fmt_num(f["pb"]))
    row[1].metric("Revenue growth", fmt_pct(f["rev_growth"]), help="Year on year, latest quarter")
    row[2].metric("Earnings growth", fmt_pct(f["earn_growth"]), help="Year on year, latest quarter")
    row[3].metric("Return on equity", fmt_pct(f["roe"]))

    show_altair(price_chart(df))
    with st.expander("Fact sheet sent to the agents"):
        st.text(sheet)


def debate_cell(side, content):
    """A bordered card for one argument. content is finished text, or a stream to type out live."""
    with st.container(border=True):
        st.markdown(f":{COLOR[side]}[**{AVATAR[side]} {LABEL[side]}**]")
        if isinstance(content, str):
            st.markdown(content)
            return content
        return st.write_stream(content)


def score_bar(bull, bear):
    total = (bull + bear) or 1
    width = round(bull / total * 100)
    html = (
        '<div style="display:flex;height:14px;border-radius:7px;overflow:hidden;margin:6px 0 4px">'
        f'<div style="width:{width}%;background:{HEX["bull"]}"></div>'
        f'<div style="width:{100 - width}%;background:{HEX["bear"]}"></div></div>'
        '<div style="display:flex;justify-content:space-between;font-size:0.9rem;margin-bottom:8px">'
        f'<span>🐂 Bull {bull:g}/10</span><span>Bear {bear:g}/10 🐻</span></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def show_verdict(verdict, raw):
    with st.container(border=True):
        if not verdict:
            st.warning("The judge didn't return a readable verdict. Its raw answer:")
            st.write(raw)
            return
        winner = verdict.get("winner", "tie")
        title = f"{AVATAR[winner]} {LABEL[winner]} wins the debate" if winner in AVATAR else "🤝 The debate is a tie"
        st.markdown(f"### {title}")
        st.caption(f"Judge's stance: {verdict.get('stance', '?')} | Confidence {verdict.get('confidence', '?')}/10")
        score_bar(to_num(verdict.get("bull_score")), to_num(verdict.get("bear_score")))

        left, right = st.columns(2)
        left.markdown(f":green[**Strongest bull point**]  \n{verdict.get('strongest_bull_point', '')}")
        right.markdown(f":red[**Strongest bear point**]  \n{verdict.get('strongest_bear_point', '')}")
        st.markdown(f"**Key risk to watch:** {verdict.get('key_risk_to_watch', '')}")
        st.markdown(f"**Why:** {verdict.get('reasoning', '')}")
        unsupported = verdict.get("unsupported_claims") or []
        if unsupported:
            with st.expander(f"Claims the judge found unsupported ({len(unsupported)})"):
                for claim in unsupported:
                    st.markdown(f"- {claim}")


def download_button(result):
    md = f"# Bull vs Bear: {result['facts']['name']}\n\n## Market facts\n```\n{result['sheet']}\n```\n\n"
    if result["context"]:
        md += f"**Context:** {result['context']}\n\n"
    if result["digest"]:
        md += f"## Document digest\n{result['digest']}\n\n"
    md += "## Debate\n\n" + "\n\n".join(
        f"**{LABEL[t['side']]} ({t['round']}):** {t['text']}" for t in result["turns"]
    )
    verdict = json.dumps(result["verdict"], indent=2) if result["verdict"] else result["raw"]
    md += f"\n\n## Verdict\n```json\n{verdict}\n```\n"
    st.download_button(
        "Download transcript", md,
        file_name=f"bull_bear_{result['facts']['ticker']}.md", mime="text/markdown",
    )


class Progress:
    """A progress bar with a label, so you can see how far along the debate is."""

    def __init__(self, total):
        self.total, self.done = total, 0
        self.bar = st.progress(0.0, text="Starting")

    def label(self, text):
        self.bar.progress(min(self.done / self.total, 1.0), text=text)

    def advance(self):
        self.done += 1

    def clear(self):
        self.bar.empty()


def replay(result):
    """Redraw a finished debate (Streamlit reruns the script on every click)."""
    show_verdict(result["verdict"], result["raw"])
    show_market(result["facts"], result["df"], result["sheet"])
    if result["digest"]:
        with st.expander("Document digest"):
            st.markdown(result["digest"])
    st.subheader("The debate")
    for rname in dict.fromkeys(t["round"] for t in result["turns"]):
        st.markdown(f"#### {rname}")
        cols = st.columns(2)
        for col, side in zip(cols, ("bull", "bear")):
            for t in result["turns"]:
                if t["round"] == rname and t["side"] == side:
                    with col:
                        debate_cell(side, t["text"])
    download_button(result)


# ---------------------------------------------------------------- Main page
st.title("🐂 Bull vs Bear 🐻")
st.caption("Two AI analysts debate a stock using live market data and your documents. A third AI judges.")

with st.container(border=True):
    col1, col2 = st.columns([1, 2])
    ticker = col1.text_input(
        "Ticker", value="INFY.NS", help="NSE: add .NS, BSE: add .BO, US: plain ticker"
    ).strip().upper()
    pdfs = col2.file_uploader(
        "Concalls, results, or investor presentations (optional)",
        type="pdf", accept_multiple_files=True,
        help="Files are sent to the AI provider for analysis. Use public filings only, not private documents.",
    )
    context = st.text_input(
        "Context or question for the debate (optional)",
        placeholder="Two quarters of investor presentations attached. Is it worth holding for 2 years?",
    ).strip()
    run = st.button("Start debate", type="primary")

if run:
    if not api_key:
        st.error("Add your API key in the sidebar to start.")
        st.stop()
    if demo_mode:
        if st.session_state.get("demo_runs", 0) >= SESSION_LIMIT:
            st.error(f"You've used your {SESSION_LIMIT} free demo debates. Add your own free Gemini key in the sidebar to keep going.")
            st.stop()
        if demo_debates_left() <= 0:
            st.error("Today's free demo debates are used up. Add your own free Gemini key in the sidebar, or try again tomorrow.")
            st.stop()
        st.session_state["demo_runs"] = st.session_state.get("demo_runs", 0) + 1
        demo_usage()["count"] += 1
    client = make_client()
    progress = Progress(total=2 + (1 if pdfs else 0) + rounds * 2)
    verdict_slot = st.container()  # filled in at the end, so the verdict shows at the top

    # 1. Market data
    progress.label("Fetching market data")
    try:
        facts, df = get_market_data(ticker)
    except Exception as e:
        progress.clear()
        st.error(str(e))
        st.stop()
    sheet = fact_sheet_text(facts)
    progress.advance()
    show_market(facts, df, sheet)

    # 2. Document analyst
    digest = ""
    if pdfs:
        progress.label(f"Reading {len(pdfs)} document(s)")
        try:
            digest = digest_pdfs(client, pdfs, facts["name"], context)
            with st.expander("Document digest"):
                st.markdown(digest)
        except Exception as e:
            st.warning(f"Couldn't read the PDFs, so the debate uses market data only. {friendly_error(e)}")
        progress.advance()

    evidence = f"=== MARKET FACTS ===\n{sheet}"
    if digest:
        evidence += f"\n\n=== DOCUMENT DIGEST ===\n{digest}"

    # 3. Debate, one row per round: Bull on the left, Bear on the right
    st.subheader("The debate")
    turns = []
    for i in range(rounds):
        rname = round_name(i, rounds)
        st.markdown(f"#### {rname}")
        cols = st.columns(2)
        for col, side in zip(cols, ("bull", "bear")):
            progress.label(f"Round {i + 1} of {rounds}: {LABEL[side]} is writing")
            so_far = transcript_text(turns) or "(No arguments yet. You speak first.)"
            user = f"DEBATE SO FAR:\n{so_far}\n\nGive your {rname.lower()} statement now."
            system = debater_system(side, facts["name"], evidence, context)
            with col:
                for attempt in range(3):
                    try:
                        text = debate_cell(side, stream(client, system, user, max_words * 2 + 200))
                        break
                    except Exception as e:
                        if is_rate_limit(e) and attempt < 2:
                            st.info("The AI provider is busy. Waiting 30 seconds, then trying again...")
                            time.sleep(30)
                        else:
                            progress.clear()
                            st.error(friendly_error(e))
                            st.stop()
            turns.append({"round": rname, "side": side, "text": text})
            progress.advance()

    # 4. Judge
    progress.label("The judge is deliberating")
    try:
        raw = generate(
            client,
            f"EVIDENCE:\n{evidence}\n\nDEBATE TRANSCRIPT:\n{transcript_text(turns)}",
            system=JUDGE_SYSTEM, max_tokens=1500,
        )
    except Exception as e:
        progress.clear()
        st.error(friendly_error(e))
        st.stop()
    progress.clear()
    verdict = parse_json(raw)
    with verdict_slot:
        show_verdict(verdict, raw)

    result = {
        "facts": facts, "df": df, "sheet": sheet, "context": context,
        "digest": digest, "turns": turns, "verdict": verdict, "raw": raw,
    }
    st.session_state["last"] = result
    download_button(result)

elif "last" in st.session_state:
    replay(st.session_state["last"])
