# Phishing URL Detection with Machine Learning and Robustness Analysis

以 URL 字串特徵比較 Logistic Regression、Random Forest 與 Gradient Boosting，分析文字擾動、特徵重要性及跨資料來源的表現差異，並測試資料增強與混合來源訓練。
資料來源與授權見 [data/README.md](data/README.md) 和 [data/external/README.md](data/external/README.md)，實驗結果見 [results/README.md](results/README.md)。
提供單筆與批次 Streamlit Demo，推論不連線到輸入網址；模型僅供實驗用途。

在 Python 3.12 環境中，從專案根目錄執行：

```shell
python -m pip install -r requirements.txt
python run_pipeline.py
python -m streamlit run app.py
```

Windows 可用 `py -3.12` 取代 `python`。
