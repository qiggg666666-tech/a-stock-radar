import os
import requests
import feedparser
import akshare as ak
import json

# GitHub Secrets 中需要配置 LLM_API_KEY 和 PUSH_KEY
GEMINI_API_KEY = os.environ.get("LLM_API_KEY")
PUSH_KEY = os.environ.get("PUSH_KEY")
RSS_URL = "https://rsshub.app/twitter/user/sszcw" # 这里替换为你监控的 RSS 源

def run_system():
    # 1. 抓取舆情
    feed = feedparser.parse(RSS_URL)
    content = "\n".join([f"{e.title}: {e.description}" for e in feed.entries[:5]])
    
    # 2. 调用 Gemini 分析
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    prompt = f"分析以下内容，提取其中的A股代码(6位数字)。返回 JSON: {{\"stocks\": [\"600519\"]}}。内容: {content}"
    resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    try:
        text = resp.json()['candidates'][0]['content']['parts'][0]['text']
        start, end = text.find('{'), text.rfind('}') + 1
        stocks = json.loads(text[start:end])['stocks']
        
        # 3. 量化筛选涨幅
        report = "### 📈 舆情精选潜力股\n"
        for s in stocks:
            df = ak.stock_zh_a_spot_em()
            stock = df[df['代码'] == s]
            status = f"涨幅: {stock['涨跌幅'].values[0]}%" if not stock.empty else "未查到"
            report += f"- {s}: {status}\n"
            
        # 4. 推送到微信
        requests.post(f"https://sctapi.ftqq.com/{PUSH_KEY}.send", data={"title": "投研信号", "desp": report})
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    run_system()import os
import requests
import feedparser
import akshare as ak
import json

# GitHub Secrets 中需要配置 LLM_API_KEY 和 PUSH_KEY
GEMINI_API_KEY = os.environ.get("LLM_API_KEY")
PUSH_KEY = os.environ.get("PUSH_KEY")
RSS_URL = "https://rsshub.app/twitter/user/sszcw" # 这里替换为你监控的 RSS 源

def run_system():
    # 1. 抓取舆情
    feed = feedparser.parse(RSS_URL)
    content = "\n".join([f"{e.title}: {e.description}" for e in feed.entries[:5]])
    
    # 2. 调用 Gemini 分析
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    prompt = f"分析以下内容，提取其中的A股代码(6位数字)。返回 JSON: {{\"stocks\": [\"600519\"]}}。内容: {content}"
    resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    try:
        text = resp.json()['candidates'][0]['content']['parts'][0]['text']
        start, end = text.find('{'), text.rfind('}') + 1
        stocks = json.loads(text[start:end])['stocks']
        
        # 3. 量化筛选涨幅
        report = "### 📈 舆情精选潜力股\n"
        for s in stocks:
            df = ak.stock_zh_a_spot_em()
            stock = df[df['代码'] == s]
            status = f"涨幅: {stock['涨跌幅'].values[0]}%" if not stock.empty else "未查到"
            report += f"- {s}: {status}\n"
            
        # 4. 推送到微信
        requests.post(f"https://sctapi.ftqq.com/{PUSH_KEY}.send", data={"title": "投研信号", "desp": report})
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    run_system()import os
import requests
import feedparser
import akshare as ak
import json

# GitHub Secrets 中需要配置 LLM_API_KEY 和 PUSH_KEY
GEMINI_API_KEY = os.environ.get("LLM_API_KEY")
PUSH_KEY = os.environ.get("PUSH_KEY")
RSS_URL = "https://rsshub.app/twitter/user/sszcw" # 这里替换为你监控的 RSS 源

def run_system():
    # 1. 抓取舆情
    feed = feedparser.parse(RSS_URL)
    content = "\n".join([f"{e.title}: {e.description}" for e in feed.entries[:5]])
    
    # 2. 调用 Gemini 分析
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    prompt = f"分析以下内容，提取其中的A股代码(6位数字)。返回 JSON: {{\"stocks\": [\"600519\"]}}。内容: {content}"
    resp = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
    
    try:
        text = resp.json()['candidates'][0]['content']['parts'][0]['text']
        start, end = text.find('{'), text.rfind('}') + 1
        stocks = json.loads(text[start:end])['stocks']
        
        # 3. 量化筛选涨幅
        report = "### 📈 舆情精选潜力股\n"
        for s in stocks:
            df = ak.stock_zh_a_spot_em()
            stock = df[df['代码'] == s]
            status = f"涨幅: {stock['涨跌幅'].values[0]}%" if not stock.empty else "未查到"
            report += f"- {s}: {status}\n"
            
        # 4. 推送到微信
        requests.post(f"https://sctapi.ftqq.com/{PUSH_KEY}.send", data={"title": "投研信号", "desp": report})
    except Exception as e:
        print(f"执行出错: {e}")

if __name__ == "__main__":
    run_system()
