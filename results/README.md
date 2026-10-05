# 實驗結果

本目錄包含分類、URL 擾動、根網址增強、第二來源評估及混合來源訓練的指標與圖表。

正類為 phishing，threshold = 0.5。FP 是正常網址被判為 phishing；FN 是 phishing 被判為正常。

| 實驗 | 結果位置 | 比較內容 |
| --- | --- | --- |
| 同來源分類 | [metrics/model_comparison.csv](metrics/model_comparison.csv) | LR、RF、HistGradientBoosting，以及 RF 增強與 HTTPS 消融 |
| 文字擾動 | [metrics/robustness.csv](metrics/robustness.csv) | 六種變換的 F1、Recall、ROC-AUC、coverage 與 flip rate |
| 根網址增強 | [metrics/root_improvement.csv](metrics/root_improvement.csv) | 加入 `/`、移除 `www` 等 training variants |
| 固定模型的第二來源評估 | [external/frozen/model_comparison.csv](external/frozen/model_comparison.csv) | 原有模型套用至 Hannousse 資料 |
| 混合來源訓練 | [mixed/model_comparison.csv](mixed/model_comparison.csv) | 加入 Hannousse training partition 前後的差異 |

## 觀察

- 原 test 的 RF F1 為 **0.9959**；query padding 後降至 **0.5986**，主要來自誤報。加入 query/path training variants 後，同項 F1 為 **0.9963**，但其他擾動仍可能失敗。
- 固定 RF 在第二來源的 F1 為 **0.6499**，benign FPR 約 **99.40%**。原抽樣的 benign 全為 HTTPS，且 path/query 皆空，跨來源結果顯示同來源高分不能代表實際泛化。
- 在第二來源的 2,116 筆保留組，混合 RF 的 F1 為 **0.8624**，相較同組 RF 的 **0.6496** 改善；FP 從 **1,093** 降至 **258**。原來源 Recall 同時由 **0.9934** 降至 **0.9746**，部分擾動表現也下降。

## 評估方式與限制

核心資料依網域切分，train/validation/test 網域不重疊，增強只使用 training URLs。核心 baseline 依 validation F1 比較，模型設定與 threshold 不依 test 調整。

根網址增強是在發現 Google 誤判後設計；混合訓練是在看過 Hannousse 的整批評估後設計。保留組沒有參與 fit，但已知的資料特性影響了實驗方向。要確認改善能否延伸到其他資料，仍需第三個未用過的來源。

文字擾動沿用原標籤，修改 path/query/subdomain 可能改變資源或網站行為，因此只衡量字串變化下的模型敏感度。Coverage 與 changed-only 指標區分有變化與未變化的輸入。

Feature importance 反映模型使用的特徵，不能直接解釋因果。Bootstrap 以網域重抽樣 1,000 次，報告各項 F1 差異的 95% percentile interval；多項比較沒有共同的顯著性判定。模型機率未校準，外部誤報率仍高，不適合直接作為安全產品。

CSV 快取與重新抽取的浮點特徵仍有精度差異：6,001 筆原 test 中，HGB 有 2 筆分類不同，RF 系列沒有分類差異。核心指標使用 CSV 快取，Demo 即時抽取特徵。

## 重現

Python 3.12；套件版本見 `requirements.txt`。Hannousse CSV 的下載方式見 [data/external/README.md](../data/external/README.md)。在專案根目錄執行：

```shell
python -m pip install -r requirements.txt
python run_pipeline.py
python -m src.improve_root
python -m src.external_test --input data/external/external.csv --models logistic_regression random_forest hist_gradient_boosting random_forest_augmented random_forest_no_https random_forest_root_augmented
python -m src.improve_data --input data/external/external.csv
python -m src.mixed_report
python -m unittest discover -s tests -v
```

Windows 可用 `py -3.12` 取代 `python`；已有 UCI 原始 CSV 時可加 `--skip-download`。

訓練與評估環境：Linux / Python 3.12.14。Windows / Python 3.12.10 的兩模型評估位於 `external/frozen/replication/`。
