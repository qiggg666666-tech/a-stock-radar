if __name__ == "__main__":
    print("=" * 70)
    print(f"🛒 上升趋势·回调低点买入 | {datetime.now():%Y-%m-%d %H:%M} | 回看{PARAMS['LOOKBACK_DAYS']}天")
    print(f"全扫={'是' if not SCAN_LIMIT else f'限{SCAN_LIMIT}'}; 趋势伞=收盘>MA60+周/月多头; 回调=缩量回踩超卖+止跌")
    print("=" * 70)
    if not is_trading_day():
        print("非交易日, 跳过"); sys.exit(0)
    df = run_scan()
    if df is None or df.empty:
        print("本次无回调买点命中(趋势伞+回调门槛较严, 属正常)"); sys.exit(0)
    # ---- 收尾全部包防护: 扫描已成功, 任何收尾IO/推送异常都不应让job误红 ----
    import traceback
    df, cluster = enrich(df)
    df = df.sort_values(["买点", "分"], ascending=[True, False]).reset_index(drop=True)
    tag = datetime.now().strftime("%Y%m%d")
    csv_ok = False
    try:
        df.to_csv(os.path.join(OUTPUT_DIR, f"dip_buy_{tag}.csv"), index=False, encoding="utf-8-sig")
        csv_ok = True
        with open(os.path.join(OUTPUT_DIR, f"dip_buy_{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump({"date": tag, "params": PARAMS, "cluster": cluster,
                       "n_strong": int(df["买点"].str.contains("🟢").sum()),
                       "n_watch": int((~df["买点"].str.contains("🟢")).sum()),
                       "hits": df.to_dict('records')},
                      f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 已存 output/dip_buy_{tag}.*")
    except Exception as e:
        print(f"\n⚠️ 存盘阶段异常(命中已在内存, 不影响结果): {type(e).__name__}: {e}")
        traceback.print_exc()
        if not csv_ok:
            try:
                df.to_csv(os.path.join(OUTPUT_DIR, f"dip_buy_{tag}.csv"), index=False, encoding="utf-8-sig")
                print(f"⚠️ 退而保存 csv: output/dip_buy_{tag}.csv")
            except Exception as e2:
                print(f"⚠️ csv 也失败: {e2}")
    try:
        disp = df.copy(); disp.insert(2, "板块", disp["行业"])
        print("\n" + disp.head(PUSH_TOP).to_string(index=False))
    except Exception as e:
        print(f"⚠️ 控制台展示异常: {e}")
    if SERVERCHAN_KEY:
        try:
            send_serverchan(f"🛒 回调低点买入 | 🟢{int(df['买点'].str.contains('🟢').sum())} 🟡{int((~df['买点'].str.contains('🟢')).sum())}",
                            build_push(df, cluster))
        except Exception as e:
            print(f"⚠️ 推送异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    sys.exit(0)   # 扫描已成功, 显式成功退出, 不受收尾异常影响
