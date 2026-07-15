"""
streamlit_app.py — Streamlit front-end dashboard for phishing email classification.
Presents a dark-themed, high-contrast, premium analytical interface.
"""

from __future__ import annotations

import json
import logging
import sys
import os
import requests
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.inference import predict_and_explain
from src.utils import load_json

log = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Startup connection check
if "api_online" not in st.session_state:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=1.0)
        st.session_state.api_online = (r.status_code == 200)
    except Exception:
        st.session_state.api_online = False

def get_prediction(
    email_text: str,
    sender_domain: str | None = None,
    reply_to_domain: str | None = None,
    subject: str = "",
    model_name: str = "lightgbm",
) -> tuple[dict[str, Any], bool]:
    """Retrieves prediction from FastAPI service via HTTP, falling back to local model if unavailable."""
    if st.session_state.get("api_online", False):
        try:
            url = f"{BACKEND_URL}/predict"
            params = {"model": model_name}
            payload = {
                "email_text": email_text,
                "sender_domain": sender_domain if sender_domain else None,
                "reply_to_domain": reply_to_domain if reply_to_domain else None,
                "subject": subject
            }
            # Short timeout to fail fast and trigger local fallback
            response = requests.post(url, json=payload, params=params, timeout=3.0)
            if response.status_code == 200:
                return response.json(), False
            else:
                log.warning("FastAPI returned status %d. Falling back to local inference.", response.status_code)
                st.session_state.api_online = False
        except Exception as e:
            log.warning("HTTP connection to backend API failed: %s. Falling back to local inference.", e)
            st.session_state.api_online = False
            
    res = predict_and_explain(
        email_text=email_text,
        sender_domain=sender_domain,
        reply_to_domain=reply_to_domain,
        subject=subject,
        model_name=model_name
    )
    return res, True

MODELS_DIR = ROOT / "models"

# ─── Custom Premium Styling ──────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme with high-contrast headers (white text) and glassmorphism styling
st.markdown(
    """
    <style>
    /* Force main app background */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Headings styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Custom cards for analytics */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Cost indicator box styling */
    .cost-box {
        background-color: rgba(255, 75, 75, 0.08);
        border-left: 4px solid #ff4b4b;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 15px;
        color: #ffcccc;
    }
    .safe-box {
        background-color: rgba(56, 239, 125, 0.08);
        border-left: 4px solid #2ecc71;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 15px;
        color: #ccffcc;
    }
    
    /* Footer styles */
    .footer {
        text-align: center;
        padding: 30px 0;
        color: #666;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Navigation Sidebar ───────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/shield.png", width=70)
    st.markdown("<h2 style='margin-top: 0px;'>Security Controls</h2>", unsafe_allow_html=True)
    st.markdown("Navigate options below:")
    
    # Visual connection status badge
    if st.session_state.get("api_online", False):
        st.markdown("<span style='color: #2ecc71; font-weight: bold;'>🟢 API Backend Online</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color: #ff4b4b; font-weight: bold;'>⚠️ Local Fallback Active (API Offline)</span>", unsafe_allow_html=True)
    
    st.markdown("")
    
    model_choice = st.selectbox(
        "Select Classification Model",
        options=["lightgbm", "logistic_regression", "random_forest", "naive_bayes"],
        format_func=lambda x: {
            "lightgbm": "LightGBM (Cost-Tuned)",
            "logistic_regression": "Logistic Regression (Linear)",
            "random_forest": "Random Forest (Ensemble)",
            "naive_bayes": "Naive Bayes (Text-Only)"
        }[x],
        index=0,
        help="Choose the model used for email classification in single and batch modes."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Metrics Summary")
    
    # Load and display training metadata in the sidebar
    meta_path = MODELS_DIR / "training_metadata.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        st.caption(f"**Dataset Size:** {meta.get('dataset_size', 'N/A'):,}")
        st.caption(f"**Optimal Threshold:** {meta.get('optimal_threshold', 0.36)}")
        st.caption(f"**Features Extracted:** {meta.get('feature_count', 'N/A'):,}")
        st.caption(f"**Last Retrained:** {meta.get('timestamp', 'N/A')[:10]}")
    else:
        st.caption("Metadata not available.")

# ─── Header Section ───────────────────────────────────────────────────────────

st.markdown("<h1 style='text-align: center;'>🛡️ Phishing Email Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa; margin-bottom: 40px;'>Analyze emails using a combination of NLP semantic vectorizers and structural header/link heuristics.</p>", unsafe_allow_html=True)

# ─── Tab Configuration ────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔍 Single Email Predictor", 
    "📁 Batch Predictor (CSV)", 
    "📈 Model Comparison & ROC"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Single Email Predictor
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    col_input, col_results = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### Input Email Details")
        email_text = st.text_area(
            "Email Text Body (max 50,000 chars)",
            height=220,
            placeholder="Paste raw email body here..."
        )
        
        # Optional metadata inputs representing headers
        with st.expander("Header Metadata (Optional)", expanded=True):
            st.markdown("<small style='color: #888;'>Providing metadata domain checks improves precision, resolving spoofing vectors.</small>", unsafe_allow_html=True)
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                sender_domain = st.text_input("Sender Domain", placeholder="e.g. paypal-security.com")
            with col_m2:
                reply_to_domain = st.text_input("Reply-To Domain", placeholder="e.g. gmail.com")
            subject = st.text_input("Email Subject", placeholder="e.g. Verify your transaction immediately!")

        analyze_clicked = st.button("Analyze Email Details", type="primary", use_container_width=True)

    with col_results:
        st.markdown("### Analysis Report")
        
        if analyze_clicked and email_text.strip():
            # Run inference
            with st.spinner("Analyzing email structure and calculating SHAP values..."):
                try:
                    res, is_fallback = get_prediction(
                        email_text=email_text,
                        sender_domain=sender_domain if sender_domain.strip() else None,
                        reply_to_domain=reply_to_domain if reply_to_domain.strip() else None,
                        subject=subject,
                        model_name=model_choice,
                    )
                    
                    if is_fallback:
                        st.warning("⚠️ Mode: Local Inference Fallback (FastAPI Backend Offline)")
                    
                    prob = res["probability"]
                    is_phish = res["is_phishing_tuned"]
                    threshold = res["optimal_threshold"]
                    
                    # 1. Prediction Banner
                    if is_phish:
                        st.markdown(
                            f"""
                            <div class='metric-card' style='border-top: 4px solid #ff4b4b;'>
                                <h3 style='color: #ff4b4b; margin: 0px;'>⚠️ Suspicious Phishing Signal</h3>
                                <p style='margin: 8px 0 0 0; color: #e0e0e0;'>The email exceeds the cost-tuned safety threshold of <strong>{threshold}</strong>.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class='metric-card' style='border-top: 4px solid #2ecc71;'>
                                <h3 style='color: #2ecc71; margin: 0px;'>✅ Safe Legitimate Signal</h3>
                                <p style='margin: 8px 0 0 0; color: #e0e0e0;'>The email is classified as safe under the decision threshold.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # 2. Probability Score
                    st.markdown("#### Phishing Likelihood")
                    # Visualise progress bar
                    col_bar, col_text = st.columns([4, 1])
                    with col_bar:
                        st.progress(prob)
                    with col_text:
                        st.markdown(f"**{prob * 100:.1f}%**")
                    
                    # Cost preference notice
                    if is_phish:
                        st.markdown(
                            f"""
                            <div class='cost-box'>
                                <strong>Security Risk Tuning Note (5:1 cost preference ratio):</strong><br/>
                                Standard threshold is 0.50, but we tuned the classifier to flag any email above <strong>{threshold}</strong>. 
                                This ensures high-recall protection against credentials harvesting.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div class='safe-box'>
                                <strong>Safety Verification:</strong><br/>
                                Probability remains below the cost-tuned limit. No immediate action required.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # 3. Explanations Plot
                    st.markdown("#### Feature Attributions (SHAP)")
                    if res["top_features"]:
                        feat_df = pd.DataFrame(res["top_features"])
                        # Flip legitimate values to negative to plot left-and-right
                        feat_df["value"] = feat_df.apply(
                            lambda row: row["shap_value"] if row["direction"] == "phishing" else -row["shap_value"],
                            axis=1
                        )
                        
                        fig = px.bar(
                            feat_df.iloc[::-1], # reverse order for top-down bar chart
                            x="value",
                            y="feature",
                            orientation="h",
                            color="direction",
                            color_discrete_map={"phishing": "#ff4b4b", "legitimate": "#2ecc71"},
                            labels={"value": "SHAP Contribution", "feature": "Feature Token"},
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color="#e0e0e0",
                            margin=dict(l=20, r=20, t=10, b=10),
                            height=250,
                            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                            yaxis=dict(showgrid=False),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("SHAP explainability not supported for Naive Bayes baseline.")

                except Exception as ex:
                    st.error(f"Inference pipeline error: {ex}")
        elif analyze_clicked:
            st.warning("Please paste email body text first.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Batch Predictor (CSV)
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown("### Batch Email Processor")
    st.markdown("Upload a CSV file containing columns for email body text and headers to run bulk classification predictions.")
    
    uploaded_file = st.file_uploader("Choose CSV File", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            # Column mapping checks
            text_cols = [c for c in df_batch.columns if "text" in c.lower() or "body" in c.lower()]
            if not text_cols:
                st.error("Uploaded CSV must contain an 'email_text' or 'body' column.")
            else:
                target_col = text_cols[0]
                st.info(f"Targeting text column: **{target_col}**")
                
                # Check optional columns
                sender_cols = [c for c in df_batch.columns if "sender" in c.lower() or "from" in c.lower()]
                reply_cols = [c for c in df_batch.columns if "reply" in c.lower()]
                subj_cols = [c for c in df_batch.columns if "subject" in c.lower()]
                
                s_col = sender_cols[0] if sender_cols else None
                r_col = reply_cols[0] if reply_cols else None
                sub_col = subj_cols[0] if subj_cols else None
                
                run_batch = st.button("Run Batch Inference", type="primary")
                
                if run_batch:
                    predictions = []
                    probabilities = []
                    tuned_labels = []
                    
                    progress_bar = st.progress(0.0)
                    total_rows = len(df_batch)
                    
                    for idx, row in df_batch.iterrows():
                        # Run classification
                        email_t = str(row[target_col])
                        snd = str(row[s_col]) if s_col else None
                        rpl = str(row[r_col]) if r_col else None
                        sbj = str(row[sub_col]) if sub_col else ""
                        
                        res, is_fallback = get_prediction(
                            email_text=email_t,
                            sender_domain=snd,
                            reply_to_domain=rpl,
                            subject=sbj,
                            model_name=model_choice,
                        )
                        
                        predictions.append(res["prediction"])
                        probabilities.append(res["probability"])
                        tuned_labels.append("phishing" if res["is_phishing_tuned"] else "legitimate")
                        
                        # Progress update
                        progress_bar.progress((idx + 1) / total_rows)
                    
                    df_batch["predicted_class"] = predictions
                    df_batch["phishing_probability"] = probabilities
                    df_batch["cost_tuned_label"] = tuned_labels
                    
                    if st.session_state.get("api_online") is False:
                        st.warning("⚠️ Mode: Local Inference Fallback (FastAPI Backend Offline)")
                    st.success("Batch classification complete!")
                    
                    # 1. Dashboard summary cards
                    col_b1, col_b2, col_b3 = st.columns(3)
                    phish_count = (df_batch["cost_tuned_label"] == "phishing").sum()
                    legit_count = len(df_batch) - phish_count
                    
                    with col_b1:
                        st.markdown(
                            f"""
                            <div class='metric-card' style='text-align: center; border-bottom: 4px solid #ff4b4b;'>
                                <h4 style='color: #ff4b4b; margin:0;'>Phishing Flagged</h4>
                                <h2 style='margin:10px 0 0 0;'>{phish_count}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_b2:
                        st.markdown(
                            f"""
                            <div class='metric-card' style='text-align: center; border-bottom: 4px solid #2ecc71;'>
                                <h4 style='color: #2ecc71; margin:0;'>Legitimate Classified</h4>
                                <h2 style='margin:10px 0 0 0;'>{legit_count}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col_b3:
                        st.markdown(
                            f"""
                            <div class='metric-card' style='text-align: center;'>
                                <h4 style='color: #aaa; margin:0;'>Phishing Rate</h4>
                                <h2 style='margin:10px 0 0 0;'>{phish_count / len(df_batch) * 100:.1f}%</h2>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # 2. Charts
                    fig_pie = px.pie(
                        names=["Legitimate", "Phishing"],
                        values=[legit_count, phish_count],
                        color_discrete_sequence=["#2ecc71", "#ff4b4b"],
                        title="Cost-Tuned Batch Distribution"
                    )
                    fig_pie.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e0e0e0",
                        height=280,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # 3. Preview output table
                    st.markdown("#### Classification Output Preview")
                    st.dataframe(
                        df_batch[[target_col, "phishing_probability", "cost_tuned_label"]].head(100),
                        use_container_width=True
                    )
                    
                    # 4. Download file button
                    csv_data = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Annotated CSV Results",
                        data=csv_data,
                        file_name="batch_phishing_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
        except Exception as e_batch:
            st.error(f"Failed to process CSV file: {e_batch}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Model Comparison & ROC
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("### Model Comparison & ROC Analysis")
    st.markdown("Compare the performance of classical ML baselines against our primary tuned LightGBM model on the 15% stratified test split.")
    
    # 1. Metrics table load
    eval_path = MODELS_DIR / "evaluation_results.json"
    if eval_path.exists():
        evals = load_json(eval_path)
        
        # Build pandas dataframe
        rows = []
        for model_name, metrics in evals.items():
            rows.append({
                "Model": model_name.replace("_", " ").title(),
                "Accuracy": f"{metrics['accuracy']*100:.2f}%",
                "Precision": f"{metrics['precision']*100:.2f}%",
                "Recall": f"{metrics['recall']*100:.2f}%",
                "F1-Score": f"{metrics['f1']*100:.2f}%",
                "AUC-ROC": f"{metrics['auc_roc']:.4f}" if metrics["auc_roc"] is not None else "N/A"
            })
        df_evals = pd.DataFrame(rows)
        st.dataframe(df_evals, use_container_width=True)
    else:
        st.info("Evaluation results json file not found.")

    # 2. Dynamic ROC Curves Plotting (No placeholders)
    st.markdown("#### Receiver Operating Characteristic (ROC) Curves")
    st.markdown("<small style='color:#888;'>Curves are plotted dynamically from test-split predictions saved in <code>models/predictions.json</code>.</small>", unsafe_allow_html=True)
    
    pred_path = MODELS_DIR / "predictions.json"
    if pred_path.exists():
        preds = load_json(pred_path)
        y_true = preds.get("y_test")
        
        if y_true:
            fig_roc = go.Figure()
            
            # Map model names to labels
            model_map = {
                "lightgbm": "LightGBM (AUC = {auc:.4f})",
                "random_forest": "Random Forest (AUC = {auc:.4f})",
                "logistic_regression": "Logistic Regression (AUC = {auc:.4f})",
                "naive_bayes": "Naive Bayes (AUC = {auc:.4f})"
            }
            colors_map = {
                "lightgbm": "#39ff14",         # Neon Green
                "random_forest": "#00d2ff",     # Blue
                "logistic_regression": "#ff007f", # Neon Pink
                "naive_bayes": "#ff9f00"        # Orange
            }
            
            for m_key, label_tmpl in model_map.items():
                y_prob = preds.get(m_key)
                if y_prob:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, 
                        y=tpr,
                        mode='lines',
                        name=label_tmpl.format(auc=roc_auc),
                        line=dict(color=colors_map[m_key], width=2)
                    ))
            
            # Add random reference line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Guess (AUC = 0.5000)',
                line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dash')
            ))
            
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1]),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1.05]),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                legend=dict(
                    x=0.55, y=0.15,
                    bgcolor="rgba(14,17,23,0.8)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1
                ),
                height=480,
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.error("No valid test target labels found in predictions.json.")
    else:
        st.warning("predictions.json file not found. Run train.py to generate test predictions.")

# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div class='footer'>🛡️ Phishing Email NLP Classifier Dashboard • Portfolio-Grade Production System</div>", unsafe_allow_html=True)
