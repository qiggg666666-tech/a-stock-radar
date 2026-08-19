# 吃掉主力研究：每日自动运行手机说明

## 运行范围

这是一条独立的第三Workflow，名称为 **Chi Diao Zhu Li Daily Research**。

它只扫描当日可获得的小流通市值A股共同股票池：流通市值20–200亿元、成交额不少于0.20亿元、排除ST和退市整理标的。它使用价格、均线和相对强弱技术研究条件，不识别任何账户主体或真实“主力”资金意图。

北京时间工作日16:30自动运行；不进入`all`，不重新启用其他停用脚本。每次只由最后的汇总任务发送一条Server酱通知。

## 需要新增或覆盖的文件

仓库已有的`chi_diao_zhu_li_optimized.py`保留不动。请按以下顺序操作：

| 顺序 | 文件 | 仓库位置 | 动作 |
|---|---|---|---|
| 1 | `chi_diao_zhu_li_smallcap_scanner.py` | 仓库根目录 | 覆盖。新版增加共同股票池与腾讯快照备用路径。 |
| 2 | `chi_diao_zhu_li_smallcap_prepare.py` | 仓库根目录 | 新建。仅准备一次当日共同股票池。 |
| 3 | `chi_diao_zhu_li_daily_summary.py` | 仓库根目录 | 新建。统一日期、质量闸门、最终CSV和唯一通知。 |
| 4 | `chi-diao-zhu-li-daily.yml` | `.github/workflows/` | 新建。第三条定时Workflow。 |

每一份文件都必须使用网页的“全选代码”后复制；文件名仅输入表中的英文名，不要复制中文标题或路径说明。

## 首次手动验证

1. 在GitHub仓库顶部点击 **Actions**。
2. 找到 **Chi Diao Zhu Li Daily Research**，点击进入。
3. 点击 **Run workflow**，保持`main`分支，再确认运行。
4. Workflow会先准备共同股票池，再运行A–D四个分片，最后执行唯一汇总。
5. Server酱通知应出现“状态、统一候选信号日、覆盖、数据源错误、最终候选、买点、质量闸门”。

只把`chi-diao-zhu-li-summary` artifact中的`chi_diao_zhu_li_daily_global_latest.csv`视为当日最终研究名单。四个分片CSV用于诊断，不是最终通知名单。

> 本指标是价格、均线和相对强弱技术研究近似版，不代表可观测的主力账户行为，也不构成投资建议。
