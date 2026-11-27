import os
import asyncio
import requests
import pandas as pd
import feedparser
from telegram import Bot
from groq import Groq
from openai import OpenAI

# --- إعدادات المفاتيح ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# --- إعدادات Groq ---
groq_client = Groq(api_key=GROQ_API_KEY)

# --- إعدادات OpenRouter ---
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def get_market_data(symbol="XAU/USD"):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1h&outputsize=50&apikey={TWELVEDATA_API_KEY}"
    response = requests.get(url).json()
    
    if "values" not in response:
        return None
    
    df = pd.DataFrame(response["values"])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    
    # حساب RSI يدويا
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # حساب MACD يدويا
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    latest = df.iloc[-1]
    return {
        "price": latest['close'],
        "rsi": round(latest['RSI'], 2),
        "macd": round(latest['MACD'], 2),
        "macd_signal": round(latest['MACD_Signal'], 2)
    }

def get_news():
    feeds = [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/news_25.rss"
    ]
    news_list = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]:
            news_list.append(f"- {entry.title}")
    return "\n".join(news_list)

def analyze_with_ai(model, prompt):
    try:
        if model == "groq":
            res = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
        
        elif model == "openrouter":
            res = openrouter_client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
            
    except Exception as e:
        return f"⚠️ تعذر التحليل من {model}: {str(e)[:50]}..."

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    
    data = get_market_data()
    news = get_news()
    
    if not data:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="⚠️ مشكل فـ جلب البيانات من TwelveData.")
        return

    prompt = f"""
    أنت محلل مالي خبير. لديك البيانات التالية:
    - السعر الحالي: {data['price']}
    - RSI: {data['rsi']}
    - MACD: {data['macd']}
    - إشارة MACD: {data['macd_signal']}
    
    آخر الأخبار:
    {news}
    
    بناءً على هذه البيانات، هل الإشارة هي "شراء" أو "بيع" أو "انتظار"؟ 
    اشرح باختصار في سطرين.
    """

    analysis_openrouter = analyze_with_ai("openrouter", prompt)
    analysis_groq = analyze_with_ai("groq", prompt)

    message = f"""
📊 **تحليل السوق (XAU/USD)**
💰 السعر: {data['price']}
📈 RSI: {data['rsi']}
📉 MACD: {data['macd']}

🤖 **رأي OpenRouter:**
{analysis_openrouter}

🤖 **رأي Groq:**
{analysis_groq}
    """
    
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')

if __name__ == "__main__":
    asyncio.run(main())
