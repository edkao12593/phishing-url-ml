# 第二資料集

使用 Hannousse & Yahiouche（2021）的 **Web page phishing detection V3**，授權為 CC BY 4.0，資料蒐集於 2020 年 5 月。

- 官方資料與下載：https://data.mendeley.com/datasets/c2gw7fy2j4/3
- Dataset DOI：10.17632/c2gw7fy2j4.3
- Paper DOI：10.1016/j.engappai.2021.104347
- 原始 11,430 筆，包含 `url`、87 個現成特徵與 `status`。
- 僅使用 `url` / `status`，重抽本專案的 43 個特徵。
- 實驗 CSV 的 SHA256：`21093e2902e5441c86a6daf95e86e7c332046e477fdf109a579d7bd81e586d6c`。

下載有 `url` / `status` 欄位的 CSV，放到 `data/external/external.csv`。`legitimate` 對應 0，`phishing` 對應 1。先完成核心 pipeline，再依目的執行：

```shell
# 使用原有模型評估第二來源
python -m src.external_test --input data/external/external.csv
# 加入第二來源的 training partition
python -m src.improve_data --input data/external/external.csv
python -m src.mixed_report
```

外部評估先排除原資料所有 split 的網域，再清理衝突與重複。混合訓練將第二來源以同樣的網域分組規則切成 train/validation/test，只加入 training partition。

原有模型的外部結果見 [frozen/model_comparison.csv](../../results/external/frozen/model_comparison.csv)。混合來源結果見 [mixed/model_comparison.csv](../../results/mixed/model_comparison.csv)。此來源已在混合訓練設計前被分析，因此 mixed holdout 為事後探索；評估方式與限制見 [results/README.md](../../results/README.md)。
