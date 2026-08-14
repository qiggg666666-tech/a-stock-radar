# 年线涨停 v7 Checkpoint：接入主 Workflow 说明

## 一次性部署

请将以下两个文件放入仓库。主 workflow 是完整合并版，包含现有任务、小市值研究任务和新增加的 v7 四分片任务。

| 文件 | 仓库路径 | 操作 |
| --- | --- | --- |
| `a-stock-radar-with-yearline-v7-checkpoint-final.yml` | `.github/workflows/a-stock-radar.yml` | **完整覆盖**现有主 workflow。 |
| `yearline_limitup_v7_checkpoint.py` | 仓库根目录 | 新建或覆盖 v7 脚本文件。 |

现有 `yearline_limitup_screener_v5_2_fast.py` 和 `yearline-limitup-screener-a/b/c/d` 不会被删除或替换。新 v7 任务是独立研究层，默认不参与 `all`。

## 自动运行

新增四个任务：

```text
yearline-v7-checkpoint-a  offset=0     limit=1500
yearline-v7-checkpoint-b  offset=1500  limit=1500
yearline-v7-checkpoint-c  offset=3000  limit=1500
yearline-v7-checkpoint-d  offset=4500  limit=1500
```

它们会在**工作日北京时间 16:30**并行启动。每个任务有 340 分钟 Actions 上限，脚本每 25 只股票保存一次阶段性结果。

## Checkpoint 恢复机制

每个分片按“分支 + 北京日期 + 分片标识”使用独立缓存键。任务再次触发时会先恢复同日同分片最近一次 checkpoint，然后脚本会跳过已完成股票并重试上一轮异常股票。

| 情况 | 操作 |
| --- | --- |
| 任务超时或被中断 | 以相同分片目标再次手动运行；不要勾选重置。 |
| 数据源暂时异常 | 直接重跑同一分片，异常代码会自动重试。 |
| 策略参数或日期变更 | 使用新的 `yearline_v7_as_of` 日期，自动形成新 checkpoint 范围。 |
| 必须从头扫描 | 手动运行时将 `yearline_v7_reset` 设为 `true`。 |

> GitHub Actions 的缓存对象不可原地覆盖，因此每次任务会用新的 run key 保存 checkpoint，下一次通过同日同分片前缀恢复最新可用快照。这是预期行为。

## 手动运行

在 **Actions → A-Stock Radar → Run workflow** 中，`target` 可选择：

```text
yearline-v7-checkpoint      # 同时运行 A/B/C/D
yearline-v7-checkpoint-a    # 仅运行 A 分片
yearline-v7-checkpoint-b    # 仅运行 B 分片
yearline-v7-checkpoint-c    # 仅运行 C 分片
yearline-v7-checkpoint-d    # 仅运行 D 分片
```

可以选填 `yearline_v7_as_of`，例如 `2025-12-31`；留空时脚本按北京时间当天运行。运行结束后，在每个 job 的 **Artifacts** 中下载 `yearline-v7-checkpoint-a/b/c/d-日期`，其中包含最终结果和 `checkpoints/` 状态文件。

## 首次验收

先只运行 `yearline-v7-checkpoint-a`。确认 Artifact 中出现：

```text
年线预警_*.csv
年线涨停确认_*.csv
年线v7运行元数据_*.json
checkpoints/yearline_v7_*.json
```

然后再运行 `yearline-v7-checkpoint` 启动四个分片。请将 v7 与既有 v5.2 快速版的候选数量、重叠率、失败数和运行时长分开观察至少 20 个交易日，再决定是否提升为主策略。该系统仅用于研究，不构成投资建议。
