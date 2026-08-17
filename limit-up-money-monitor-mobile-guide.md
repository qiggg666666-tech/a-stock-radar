# 涨停基因 + 资金流入监控：手机部署说明

## 文件与仓库路径

请把以下3个文件上传到仓库根目录：`monitor.py`、`requirements-monitor.txt`。再把`limit-up-money-monitor.yml`上传到`.github/workflows/limit-up-money-monitor.yml`。如果仓库已经有同名文件，请打开已有文件并选择编辑覆盖，不要创建同名新文件。

| 文件 | GitHub目标路径 | 作用 |
| --- | --- | --- |
| `monitor.py` | `/monitor.py` | 获取近5个交易日涨停基因、今日资金流并联合筛选 |
| `requirements-monitor.txt` | `/requirements-monitor.txt` | Python 3.11依赖 |
| `limit-up-money-monitor.yml` | `/.github/workflows/limit-up-money-monitor.yml` | 交易时段每15分钟运行的独立Workflow |

## Secret配置

在GitHub仓库进入 `Settings → Secrets and variables → Actions → New repository secret`，新增：

| Name | Value |
| --- | --- |
| `SERVERCHAN_SENDKEY` | 你的Server酱SendKey |

密钥不会写入Python文件、CSV、JSON或artifact。不要把SendKey直接粘贴到网页代码档案或Workflow明文中。

## 筛选口径

脚本默认读取近5个交易日涨停池，并与今日主力净流入排名合并。默认阈值为主力净流入不少于3000万元，最多发送Top 20。生成的CSV和JSON写入`output/`，同时通过Server酱发送一条汇总消息。

## 运行时间

GitHub Actions使用UTC。Workflow中的`*/15 1-3,5-7 * * 1-5`对应北京时间工作日09:00–11:00和13:00–15:00每15分钟运行。GitHub Actions的定时任务可能存在延迟；交易日、节假日和数据源临时维护时，运行失败应以artifact和日志为准。

## 手机首次验收

先在Actions页面打开“涨停基因+资金流入监控”，点击 `Run workflow` 手动运行一次。确认Secret已经配置后再点击运行。运行成功后，在页面底部下载`limit-up-money-monitor-运行编号` artifact，检查其中是否有CSV和JSON。若当天没有可用数据，脚本会发送失败提示并以非零状态结束，避免把数据源失败误判为零候选。

## 安全与范围说明

该监控仅输出规则筛选结果，不构成投资建议。它使用公开数据接口，但不保证接口在每个交易日都稳定；如出现涨停池读取失败、资金流字段变化或接口限流，应先查看Actions日志，再调整字段兼容逻辑，不要直接放宽资金流阈值。
