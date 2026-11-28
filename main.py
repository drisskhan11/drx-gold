import os
import asyncio
import requests
import pandas as pd
import feedparser
from datetime import datetime, timezone
from telegram import Bot
from groq import Groq
from openai import OpenAI

# --- الإعدادات ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# --- إدارة رأس المال (Money Management) ---
ACCOUNT_BALANCE = 100.0 # هنا كتحط رأس المال ديالك بالدولار (مثلا 100)
RISK_PERCENT = 2.0 # كتخاطر بـ 2% من الحساب فـ كل صفقة

def calculate_lot_size(balance, risk_percent, sl_pips):
    risk_amount = balance * (risk_percent / 100)
    if sl_pips == 0:
        return 0.01
    lot_size = risk_amount / (sl_pips * 10)
    lot_size = round(lot_size, 2)
    if lot_size < 0.01:
        lot_size = 0.01
    return lot_size

def get_market_data(symbol="XAU/USD"):
    # إطار 1 دقيقة للإشارات السريعة (Scalping)
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1min&outputsize=100&apikey={TWELVEDATA_API_KEY}"
    response = requests.get(url).json()
    
    if "values" not in response:
        return None
    
    df = pd.DataFrame(response["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    
    # 1. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. EMA
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 4. Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # 5. Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # 6. ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    latest = df.iloc[-1]
    return {
        "price": latest['close'],
        "rsi": round(latest['RSI'], 2),
        "macd": round(latest['MACD'], 2),
        "macd_signal": round(latest['MACD_Signal'], 2),
        "ema_20": round(latest['EMA_20'], 2),
        "ema_50": round(latest['EMA_50'], 2),
        "bb_upper": round(latest['BB_Upper'], 2),
        "bb_lower": round(latest['BB_Lower'], 2),
        "stoch_k": round(latest['Stoch_K'], 2),
        "stoch_d": round(latest['Stoch_D'], 2),
        "atr": round(latest['ATR'], 3),
    }

def get_filtered_news():
    feeds = [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/news_25.rss"
    ]
    keywords = ["gold", "xau", "silver", "metal", "mining", "dollar", "fed", "inflation"]
    news_list = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                text = title.lower()
                if any(kw in text for kw in keywords):
                    news_list.append(f"- {title}")
                    if len(news_list) >= 4: break
            if len(news_list) >= 4: break
        except: continue
    return "\n".join(news_list) if news_list else "لا توجد أخبار مهمة."

def is_trading_session():
    # التداول من 8:00 صباحاً حتى 4:00 بعد الزوال (UTC)
    now = datetime.now(timezone.utc)
    return 8 <= now.hour < 16

def calculate_signal(data):
    price = data['price']
    ema20 = data['ema_20']
    ema50 = data['ema_50']
    rsi = data['rsi']
    macd = data['macd']
    macd_signal = data['macd_signal']
    stoch_k = data['stoch_k']
    stoch_d = data['stoch_d']
    bb_upper = data['bb_upper']
    bb_lower = data['bb_lower']
    atr = data['atr']
    
    score = 0
    direction = "NEUTRAL"
    
    # شروط الشراء
    if price > ema20 and rsi < 70 and macd > macd_signal and stoch_k > stoch_d and price < bb_upper:
        score += 1
    if 40 < rsi < 70: score += 1
    if macd > macd_signal: score += 1
    if stoch_k > stoch_d and stoch_k < 80: score += 1
    if price > ema20: score += 1
    
    if score >= 3:
        direction = "BUY"
    
    # شروط البيع
    score = 0
    if price < ema20 and rsi > 30 and macd < macd_signal and stoch_k < stoch_d and price > bb_lower:
        score += 1
    if 30 < rsi < 60: score += 1
    if macd < macd_signal: score += 1
    if stoch_k < stoch_d and stoch_k > 20: score += 1
    if price < ema20: score += 1
    
    if score >= 3:
        direction = "SELL"
    
    # الأهداف (Scalping)
    if direction == "BUY":
        entry = price
        sl = price - (atr * 1.0)
        tp = price + (atr * 2.0) # Reward 1:2
        sl_pips = (entry - sl) * 100
        lot_size = calculate_lot_size(ACCOUNT_BALANCE, RISK_PERCENT, sl_pips)
        return direction, entry, sl, tp, lot_size
    
    elif direction == "SELL":
        entry = price
        sl = price + (atr * 1.0)
        tp = price - (atr * 2.0)
        sl_pips = (sl - entry) * 100
        lot_size = calculate_lot_size(ACCOUNT_BALANCE, RISK_PERCENT, sl_pips)
        return direction, entry, sl, tp, lot_size
    
    return "NEUTRAL", 0, 0, 0, 0

def analyze_with_ai(model, prompt):
    try:
        if model == "groq":
            res = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            return res.choices[0].message.content
        elif model == "openrouter":
            res = openrouter_client.chat.completions.create(
                model="inclusionai/ling-3.0-flash-fin:free",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ {model}: {str(e)[:50]}..."

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    
    if not is_trading_session():
        print("خارج أوقات السيولة. البوت كيرتاح.")
        return
    
    data = get_market_data()
    news = get_filtered_news()
    
    if not data:
        return

    direction, entry, sl, tp, lot_size = calculate_signal(data)
    
    if direction == "NEUTRAL":
        print(f"لا توجد إشارة. البوت كيراقب.")
        return
    
    prompt = f"""
    أنت مدير صندوق تحوط خبير في المضاربة السريعة. 
    لدينا إشارة {direction} على الذهب.
    - السعر: {data['price']}, الدخول: {entry}, وقف الخسارة: {sl}, الهدف: {tp}
    - RSI: {data['rsi']}, MACD: {data['macd']}, Stoch: {data['stoch_k']}
    - الأخبار: {news}
    هل تؤكد هذه الصفقة السريعة؟ اشرح باختصار.
    """
    
    ai_opinion = analyze_with_ai("openrouter", prompt)

    message = f"""
⚡ **إشارة تداول سريعة (XAU/USD)** ⚡
الاتجاه: **{direction}**
💰 السعر الحالي: {data['price']}
🚪 الدخول: {entry}
🛑 وقف الخسارة (SL): {sl}
🎯 الهدف (TP): {tp}

📊 **حجم الصفقة (Lot Size): {lot_size}**
*(بناءً على حساب {ACCOUNT_BALANCE}$ ومخاطرة {RISK_PERCENT}%)*

📈 RSI: {data['rsi']} | MACD: {data['macd']} | Stoch: {data['stoch_k']}

🤖 **رأي المستشار المالي:**
{ai_opinion}

⚠️ *هذا تحليل تقني وليس نصيحة استثمارية. التزم بإدارة رأس المال.*
    """
    
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')

if __name__ == "__main__":
    asyncio.run(main())
