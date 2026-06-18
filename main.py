
from datetime import datetime

print("A股海外舆情雷达启动")
print("当前时间:", datetime.now())

stocks = [
    "宁德时代",
    "中芯国际",
    "比亚迪",
    "寒武纪"
]

print("\n今日监控股票：")

for stock in stocks:
    print("-", stock)
