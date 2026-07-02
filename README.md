# A-Stock Radar - Triple Cross Strategy

基于 AKShare 的A股选股策略，自动筛选**年月周即将金叉**个股，并推送通知。

## 策略逻辑
- 周线：5MA 即将金叉 20MA（距离 < 0.8%）
- 月线：5MA 即将金叉 20MA（距离 < 1.2%）
- 年线：250日MA 即将金叉 20日MA（距离 < 1.8%）
- 股价 > 5元

## 安装
```bash
pip install -r requirements.txt
