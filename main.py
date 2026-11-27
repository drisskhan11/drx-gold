import os
import asyncio
import requests
import pandas as pd
import pandas_ta_classic as ta
import feedparser
from telegram import Bot
from openai import OpenAI
from groq import Groq

# --- إعدادات المفاتيح ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- إعدادات OpenAI و Groq ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

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
    
    # حساب المؤشرات
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    
    latest = df.iloc[-1]
    return {
        "price": latest['close'],
        "rsi": latest.get('RSI_14', 'N/A'),
        "macd": latest.get('MACD_12_26_9', 'N/A'),
        "macd_signal": latest.get('MACDs_12_26_9', 'N/A')
    }

def get_news():
    # جلب أخبار من مصادر مجانية
    feeds = [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/news_25.rss"
    ]
    news_list = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]: # جلب 3 أخبار من كل مصدر
            news_list.append(f"- {entry.title}")
    return "\n".join(news_list)

def analyze_with_ai(model, prompt):
    if model == "openai":
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    elif model == "groq":
        res = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    
    # 1. جلب البيانات
    data = get_market_data()
    news = get_news()
    
    if not data:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="⚠️ مشكل فـ جلب البيانات من TwelveData.")
        return

    # 2. تحضير السؤال للذكاء الاصطناعي
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

    # 3. التحليل بـ AI
    analysis_openai = analyze_with_ai("openai", prompt)
    analysis_groq = analyze_with_ai("groq", prompt)

    # 4. تجميع الرسالة
    message = f"""
📊 **تحليل السوق (XAU/USD)**
💰 السعر: {data['price']}
📈 RSI: {data['rsi']}
📉 MACD: {data['macd']}

🤖 **رأي OpenAI:**
{analysis_openai}

🤖 **رأي Groq (Llama 3):**
{analysis_groq}
    """
    
    # 5. إرسال الرسالة
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')

if __name__ == "__main__":
    asyncio.run(main())
