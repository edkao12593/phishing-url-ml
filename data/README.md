# 主資料集

使用 Prasad & Chandra（2024）的 **PhiUSIIL Phishing URL (Website)**，來自 UCI Machine Learning Repository，授權為 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

- 資料頁：https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- 官方下載：https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip
- UCI 提供的 DOI：https://doi.org/10.1016/j.cose.2023.103545
- 原始 235,795 筆：134,850 legitimate、100,945 phishing。
- 僅讀取 `URL` 與 `label`，重新抽取 43 個特徵。
- 原標籤 `1 = legitimate, 0 = phishing`；本專案反轉為 `1 = phishing, 0 = benign`。

## 下載與處理

在專案根目錄執行：

```shell
python -m src.download_data
```

也可先下載官方 ZIP，再執行 `python -m src.download_data --archive "C:\Downloads\phiusiil.zip"`。來源檔案的 SHA256 記錄在 `source_metadata.json`，下載後會核對內容。

Canonicalization 統一 scheme、IDNA hostname 與 hostname 尾端的點，保留 path/query/fragment 大小寫、明示 port 和 userinfo。沒有 scheme 的輸入補上 HTTP。刪除無效輸入、標籤衝突與重複 URL，再以 seed 42 分層抽樣 30,000 筆。

以 registrable domain 分組，使用 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`：fold 0 為 test、fold 1 為 validation，其餘為 train。網域解析使用 tldextract 5.3.0 內建 PSL，包含 private suffix；IP 與無已知 suffix 的 URL 使用 hostname 分組。

抽樣、清理筆數與各 split 的資料量見 [data_audit.json](../results/metrics/data_audit.json)。raw/processed records 和模型由執行時產生，不隨 repository 發布。資料授權與 MIT 程式碼授權分開適用。
