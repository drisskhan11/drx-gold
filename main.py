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

groq_client = Groq(api_key=GROQ_API_KEY)
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
    
    # 3. EMA 20 و EMA 50 (متوسطات متحركة)
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 4. Bollinger Bands (20 فترة، 2 انحراف معياري)
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    # 5. Stochastic Oscillator (14, 3, 3)
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
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
    }

def get_dxy():
    # مؤشر الدولار (DXY) من TwelveData
    url = f"https://api.twelvedata.com/time_series?symbol=DXY&interval=1h&outputsize=1&apikey={TWELVEDATA_API_KEY}"
    try:
        response = requests.get(url).json()
        if "values" in response and len(response["values"]) > 0:
            return round(float(response["values"][0]["close"]), 2)
    except:
        pass
    return None

def get_filtered_news():
    feeds = [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/news_25.rss",
        "https://www.fxstreet.com/rss/news",
        "https://www.kitco.com/rss/KitcoNews.xml"
    ]
    # كلمات مفتاحية: الذهب، المعادن، الدولار، الفيدرالي، التضخم
    keywords = ["gold", "xau", "silver", "metal", "mining", "dollar", "fed", "inflation", "bullion", "precious"]
    news_list = []
    
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:  # شوف 10 أخبار من كل مصدر
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                text = (title + " " + summary).lower()
                
                # فلترة: غير الأخبار اللي فيها كلمات مفتاحية
                if any(kw in text for kw in keywords):
                    news_list.append(f"- {title}")
                    if len(news_list) >= 6:  # خد 6 أخبار غير
                        break
            if len(news_list) >= 6:
                break
        except:
            continue
    
    return "\n".join(news_list) if news_list else "لا توجد أخبار مهمة حالياً."

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
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
            
    except Exception as e:
        return f"⚠️ تعذر التحليل من {model}: {str(e)[:50]}..."

async def main():
    bot = Bot(token=TELEGRAM_TOKEN)
    
    data = get_market_data()
    news = get_filtered_news()
    dxy = get_dxy()
    
    if not data:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="⚠️ مشكل فـ جلب البيانات من TwelveData.")
        return

    dxy_text = f"{dxy}" if dxy else "غير متاح"
    
    prompt = f"""
    أنت محلل مالي خبير. لديك البيانات التالية عن الذهب (XAU/USD):
    
    **البيانات التقنية:**
    - السعر الحالي: {data['price']}
    - RSI (14): {data['rsi']}
    - MACD: {data['macd']} (إشارة: {data['macd_signal']})
    - EMA 20: {data['ema_20']} | EMA 50: {data['ema_50']}
    - Bollinger Bands: العلوي {data['bb_upper']} | السفلي {data['bb_lower']}
    - Stochastic K: {data['stoch_k']} | D: {data['stoch_d']}
    
    **مؤشر الدولار (DXY):** {dxy_text}
    
    **آخر الأخبار المهمة (الذهب، المعادن، الدولار):**
    {news}
    
    بناءً على هذه البيانات، هل الإشارة هي "شراء" أو "بيع" أو "انتظار"؟ 
    اشرح باختصار (3-4 أسطر) مع ذكر تأثير مؤشر الدولار والأخبار.
    """

    analysis_openrouter = analyze_with_ai("openrouter", prompt)
    analysis_groq = analyze_with_ai("groq", prompt)

    message = f"""
📊 **تحليل السوق الشامل (XAU/USD)**
💰 السعر: {data['price']}
📈 RSI: {data['rsi']} | Stoch K: {data['stoch_k']}
📉 MACD: {data['macd']} (Signal: {data['macd_signal']})
📊 EMA 20/50: {data['ema_20']} / {data['ema_50']}
🎯 BB: {data['bb_upper']} / {data['bb_lower']}
💵 DXY (الدولار): {dxy_text}

📰 **الأخبار المهمة:**
{news}

🤖 **رأي OpenRouter:**
{analysis_openrouter}

🤖 **رأي Groq:**
{analysis_groq}
    """
    
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='Markdown')

if __name__ == "__main__":
    asyncio.run(main())
