# 六类策略单条汇总通知部署说明

## 本版目的

本版本将六类四分片策略统一为“**A/B/C/D分片完成后仅发送一条策略级汇总消息**”。分片任务仍各自生成 CSV、JSON、Markdown 与 checkpoint artifact，但不再直接向手机推送；汇总任务下载对应的四片 artifact 后，按股票代码去重、按策略评分排序，再向 Server酱与可选 Telegram 发送一条摘要。

| 策略 | 子分片 | 汇总任务 | 候选排序字段 |
| --- | --- | --- | --- |
| 底部吸筹快速版 | `bottom-accumulation-fast-a/b/c/d` | `bottom-accumulation-notify-summary` | `score` |
| 形态突破安全版 | `pattern-breakout-fast-a/b/c/d` | `pattern-breakout-notify-summary` | `评分` |
| 小市值趋势扫描 | `smallcap-trend-scan-a/b/c/d` | `smallcap-trend-notify-summary` | `信号评分` |
| VCP快速精简版 | `vcp-screener-a/b/c/d` | `vcp-fast-notify-summary` | `VCP_Score` |
| 牛市确认快速版 | `bull-confirm-screener-a/b/c/d` | `bull-confirm-notify-summary` | `多头得分` |
| 年线涨停v5.2快速版 | `yearline-limitup-screener-a/b/c/d` | `yearline-limitup-v5-notify-summary` | `综合评分` |

> 年线 v7 checkpoint 的独立汇总通知继续保留，不受本次改动影响。

## 覆盖与新增文件

| 本地文件 | 仓库目标路径 | 操作 |
| --- | --- | --- |
| `a-stock-radar-with-all-strategy-summary-notifications.yml` | `.github/workflows/a-stock-radar.yml` | **完整覆盖**主 workflow |
| `strategy_shard_notify_summary.py` | 仓库根目录 | 覆盖为本版通用汇总器 |
| `yearline_limitup_v7_checkpoint.py` | 仓库根目录 | 保留已有文件 |
| `yearline_v7_notify_summary.py` | 仓库根目录 | 保留已有文件 |

本版 workflow 共 **79 个 job**：原有 76 个 job 保留，并新增 VCP、牛市确认、年线 v5.2 三个汇总通知 job。

## GitHub Secrets

在仓库内打开：

```text
Settings → Secrets and variables → Actions → New repository secret
```

| Secret | 用途 | 配置要求 |
| --- | --- | --- |
| `SENDKEY` | Server酱推送 | 使用 Server酱时配置 |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API | 使用 Telegram 时与 Chat ID 同时配置 |
| `TELEGRAM_CHAT_ID` | Telegram 接收对象 | 使用 Telegram 时与 Bot Token 同时配置 |

不要把真实凭据写入 Python、YAML、Markdown、artifact 或 commit。汇总器只在日志中打印成功、失败或跳过状态，不会打印凭据本身。

## VCP、牛市确认与年线 v5.2 的改动细节

| 策略 | 分片直推关闭方式 | 汇总后通知 |
| --- | --- | --- |
| VCP | A分片原有 `SERVERCHAN_KEY` 已清空，B/C/D继续为空 | `vcp-fast-notify-summary` |
| 牛市确认 | A分片原有 `SERVERCHAN_KEY` 已清空，B/C/D继续为空 | `bull-confirm-notify-summary` |
| 年线 v5.2 | A分片移除 `--push`；所有分片 `SENDKEY` 置空 | `yearline-limitup-v5-notify-summary` |

## 首次验收

建议按以下顺序逐个验收，避免首次排查时混淆多类消息：

| 手动目标 | 应观察的汇总任务 | 预期 |
| --- | --- | --- |
| `vcp-screener-a` | `vcp-fast-notify-summary` | VCP仅一条汇总消息 |
| `bull-confirm-screener-a` | `bull-confirm-notify-summary` | 牛市确认仅一条汇总消息 |
| `yearline-limitup-screener-a` | `yearline-limitup-v5-notify-summary` | 年线v5.2仅一条汇总消息 |

首次手动运行时，保持：

```text
strategy_notifications = true
strategy_notify_zero = true
```

即使本次无候选，也会收到“本次无候选”的策略状态消息，便于确认通知链路；只想在有候选时通知时，将 `strategy_notify_zero` 设为 `false`。手动不推送时，将 `strategy_notifications` 设为 `false`。定时运行默认启用二者。

## 日志诊断

| 日志关键词 | 含义 | 后续动作 |
| --- | --- | --- |
| `SUCCESS: Server酱业务返回 code=0` | Server酱已确认接收 | 无需处理 |
| `SUCCESS: Telegram业务返回 ok=true` | Telegram已确认接收 | 无需处理 |
| `FAILED: Server酱业务返回` | HTTP成功但Server酱业务拒绝 | 检查 SendKey、通道状态和服务额度 |
| `SKIPPED: 未配置 SENDKEY` | Server酱未配置 | 新建 `SENDKEY` 或仅使用Telegram |
| `SKIPPED: 未同时配置 TELEGRAM_BOT_TOKEN 与 TELEGRAM_CHAT_ID` | Telegram凭据不完整 | 同时配置两项 Secret |
| `本次无候选` | 分片与汇总任务可正常完成，但未命中候选 | 属于策略结果，不是推送故障 |

## References

[1] [Server酱 Turbo 官网](https://sct.ftqq.com/)

[2] [Telegram Bot API 官方文档](https://core.telegram.org/bots/api)
