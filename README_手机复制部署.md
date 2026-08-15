# A-Stock Radar：底部吸筹最终修复版（手机复制部署）

## 本包包含什么

这是一套以**共享股票池 + A/B/C/D/E五分片**为基础的最终部署文件集。它解决了以下问题：股票池临时不可用被误判为零候选；四/五片重复请求股票池；分片失败阻断汇总；E片未被通知和仪表盘识别；敏感通知密钥进入分片artifact。

## 必须上传的六个文件

| 序号 | 本包文件名 | GitHub仓库目标路径 | 操作 |
| ---: | --- | --- | --- |
| 1 | `prepare_a_share_universe.py` | 仓库根目录 | 新增/覆盖 |
| 2 | `bottom_accumulation_screener_fast.py` | 仓库根目录 | 覆盖 |
| 3 | `strategy_shard_runner.py` | 仓库根目录 | 覆盖 |
| 4 | `strategy_shard_notify_summary.py` | 仓库根目录 | 覆盖 |
| 5 | `dashboard_snapshot_publisher.py` | 仓库根目录 | 覆盖 |
| 6 | `a-stock-radar.yml` | `.github/workflows/a-stock-radar.yml` | 覆盖 |

> 第6个文件在本压缩包内位于 `.github/workflows/a-stock-radar.yml`；若单独下载后看到其路径，请保持该目录层级。不要把它上传到仓库根目录。

## 手机端安全更新顺序

1. 在Server酱后台生成一个新的SendKey。此前artifact曾包含旧配置值，旧值不应继续使用。
2. 在GitHub仓库手机网页打开 `Settings → Secrets and variables → Actions`，编辑 `SENDKEY` 并保存新值。切勿把SendKey写入任何Python或YAML文件。
3. 在仓库根目录按表中第1–5项逐个新增/覆盖文件；每上传一个文件就提交到 `main` 分支。
4. 最后打开 `.github/workflows/a-stock-radar.yml`，使用本包第6个文件完整覆盖，提交到 `main` 分支。
5. 打开 `Actions → A-Stock Radar → Run workflow`，第一次只选 `bottom-accumulation-fast`，不要选 `all`。

## 预期的任务顺序

```text
bottom-accumulation-universe
  └── 获取一次股票池：AkShare → BaoStock → 最多3天有效缓存
       ├── bottom-accumulation-fast-a（0–1199）
       ├── bottom-accumulation-fast-b（1200–2399）
       ├── bottom-accumulation-fast-c（2400–3599）
       ├── bottom-accumulation-fast-d（3600–4799）
       └── bottom-accumulation-fast-e（4800+）
                 └── bottom-accumulation-notify-summary（只发送一条汇总）
```

## 首次验收标准

| 位置 | 正确结果 |
| --- | --- |
| `bottom-accumulation-universe` | 绿色；artifact内 `a_share_universe_status.json` 的 `state=ready` 或 `degraded_cache` |
| A/B/C/D/E五片 | 实际运行；候选为0也可以，但五片合计必须有 `processed > 0` 才能称为真实零候选 |
| 任意单片超时/失败 | 汇总任务仍执行，通知显示缺片或失败退出码；整次Actions因故障分片而显示失败属正常现象 |
| 汇总通知 | 每次仅1条，不会按五片重复推送 |
| 仪表盘快照 | 应显示分片 `A/B/C/D/E`；如缺片，不能仅以 `ready` 判定完整，应核对分片标签 |

## 已完成的离线验证

六个Python文件已通过编译；共享股票池、脱敏输出、股票池失败的非零退出码、单片超时容错、A-E artifact合并、仪表盘E标签、其他策略仍使用A-D的兼容性、81个job的YAML解析和缩进检查均已通过。

## 回退

部署前请在GitHub的文件页使用下载或复制方式保留现有六个文件。若首次验收异常，用保留的旧文件覆盖回来；不要删除源文件或历史artifact。
