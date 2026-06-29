import os
import requests

def push_to_wechat(title: str, content: str):
    """
    通过 Server 酱 (sctapi.ftqq.com) 推送到微信
    确保你在 GitHub Secrets 中配置了 SENDKEY
    """
    sendkey = os.getenv("SENDKEY")
    if not sendkey:
        print("警告: 未检测到 SENDKEY，无法推送至微信。")
        return

    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    data = {
        "title": title,
        "desp": content
    }
    try:
        res = requests.post(url, data=data)
        if res.status_code == 200:
            print("-> 微信推送成功！")
        else:
            print(f"-> 推送失败，状态码: {res.status_code}")
    except Exception as e:
        print(f"-> 推送异常: {e}")
