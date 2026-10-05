"""Streamlit interface for single and batch URL predictions."""

import joblib
import pandas as pd
import streamlit as st
from src.features import extract_features, feature_frame
from src.predict import predict_lines, csv_bytes
from src.utils import ROOT

st.set_page_config(page_title="URL ML Research Demo", page_icon="🔎", layout="centered")
st.title("釣魚網址偵測實驗 Demo")
st.caption("URL-only Machine Learning · Offline text analysis")
st.warning("實驗模型，預測不代表網址已驗證安全。")
available = sorted(p.stem for p in (ROOT / "models").glob("*.joblib"))
if not available:
    st.error("尚無模型，請先執行 python run_pipeline.py")
    st.stop()
choice = st.selectbox(
    "模型",
    available,
    index=(
        available.index("random_forest_augmented")
        if "random_forest_augmented" in available
        else 0
    ),
)


@st.cache_resource
def load_model(name):
    return joblib.load(ROOT / "models" / f"{name}.joblib")


with st.form("url_form"):
    url = st.text_input(
        "URL", value="https://example.org/help?lang=en", max_chars=16384
    )
    submitted = st.form_submit_button("分析 URL 字串")
if submitted:
    try:
        feature_values = extract_features(url)
        model_bundle = load_model(choice)
        phishing_probability = float(
            model_bundle["model"].predict_proba(
                feature_frame([url])[model_bundle["features"]]
            )[0, 1]
        )
        st.subheader(
            "Phishing"
            if phishing_probability >= model_bundle["threshold"]
            else "Benign"
        )
        st.metric("模型輸出的 phishing probability", f"{phishing_probability:.1%}")
        st.caption(
            "固定 threshold = 0.5。此分數未做 probability calibration，不能解讀成真實世界的風險機率。"
        )
        cols = st.columns(3)
        for column, feature_name in zip(
            cols, ["url_length", "hostname_length", "path_depth"]
        ):
            column.metric(feature_name, str(feature_values[feature_name]))
        st.dataframe(
            pd.DataFrame(
                {
                    "feature": list(feature_values),
                    "value": list(feature_values.values()),
                }
            ),
            hide_index=True,
            width="stretch",
        )
    except (ValueError, UnicodeError) as error:
        st.error(str(error))

st.divider()
st.subheader("批次 URL 測試")
st.caption(
    "一行一個網址，最多 1,000 筆；略過空白行、保留順序與重複項。格式錯誤不影響其他列。"
)
with st.form("batch_url_form"):
    batch_text = st.text_area(
        "多行 URL",
        value="https://example.com\nhttps://example.com/help?lang=en\njavascript:alert(1)",
        height=180,
        max_chars=2_000_000,
    )
    batch_submitted = st.form_submit_button("批次分析 URL 字串")
if batch_submitted:
    try:
        with st.spinner("離線分析中…"):
            batch_results = predict_lines(batch_text, load_model(choice), choice)
        st.session_state["batch_results"] = batch_results
        st.session_state["batch_model"] = choice
    except (ValueError, UnicodeError) as error:
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_model", None)
        st.error(str(error))
if "batch_results" in st.session_state:
    batch_results = st.session_state["batch_results"]
    batch_model = st.session_state["batch_model"]
    st.caption(
        f"以下是上次批次分析的結果，使用模型：{batch_model}。變更輸入或模型後，請重新按批次分析。"
    )
    if batch_results.empty:
        st.info("請輸入至少一行 URL。")
    else:
        valid_count = int((batch_results.status == "ok").sum())
        flagged_count = int((batch_results.prediction == "Phishing").sum())
        st.write(
            f"共 {len(batch_results)} 筆｜成功 {valid_count} 筆｜"
            f"格式錯誤 {len(batch_results)-valid_count} 筆｜判為 Phishing {flagged_count} 筆"
        )
        st.dataframe(
            batch_results,
            hide_index=True,
            width="stretch",
            column_config={
                "phishing_probability": st.column_config.NumberColumn(
                    "phishing_probability (0–1)", format="%.4f"
                )
            },
        )
        st.download_button(
            "下載批次結果 CSV",
            data=csv_bytes(batch_results),
            file_name=f"batch_predictions_{batch_model}.csv",
            mime="text/csv",
        )
        st.caption("未提供正確標籤，僅列模型預測，不計算評估指標。")
st.divider()
st.caption(
    "Phishing URL Detection with Machine Learning and Robustness Analysis | Research prototype"
)
