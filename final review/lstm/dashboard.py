import streamlit as st
import pandas as pd
import json
from pathlib import Path
import glob

# Set page config to wide mode for better layout
st.set_page_config(layout="wide")

st.title("MARS-Fin: Multi-Modal Architectural Regime-Specific Financial Forecasting")

# --- 1. Create Main Tabs ---
tab1, tab2 = st.tabs(["MARS-Fin Thesis & Summary", "Experiment Deep-Dive"])

# --- TAB 1: PROJECT SUMMARY ---
with tab1:
    st.header("Project Thesis and High-Level Conclusions")
    
    st.markdown("""
    This project proves that a "one-size-fits-all" deep learning model for financial forecasting is not viable. 
    Predictive success is **regime-specific**. The features and problem definitions required for a stable, 
    macro-sensitive asset (`ASIANPAINTS`) are fundamentally different from those required for a volatile, 
    news-driven asset (`ADANIENT`).
    """)

    st.subheader("Volatility Comparison: A Tale of Two Baselines")
    st.markdown("""
    A critical finding was the difference in the "Naive Baseline" (the RMSE of predicting '0' change every day). 
    This baseline is a direct measure of a test set's volatility.
    
    * **`ASIANPAINTS` Naive RMSE: ~291.41**
    * **`ADANIENT` Naive RMSE: ~413.45** (from our final dataset)
    
    This tells us the `ADANIENT` test period (which included the Hindenburg crisis) was **42% more volatile** than the `ASIANPAINTS` test period. Our models weren't just being tested—they were being tested in 'easy mode' 
    and 'hard mode', which explains why finding a signal in the `ADANIENT` data was so much more difficult.
    """)

    st.subheader("Future Scope")
    st.markdown("""
    1.  **Run SHAP on `ADANIENT`:** Now that we have a (weakly) successful model, run a SHAP analysis on it. This will prove *which* features (e.g., `is_fraud_allegation`, `sentiment_weighted_mean_roll_avg_7d`) are the most predictive.
    2.  **Apply "Context" to `ASIANPAINTS`:** Create multi-hot news categories for `ASIANPAINTS` and add them to the successful "Events-Only" model to see if it can improve the 75% accuracy.
    3.  **Solve Technical Analysis:** Find a robust, working `pandas_ta` implementation or build the TA features (RSI, Bollinger Bands) manually to create a true "Tech + Sentiment + Context" model for `ADANIENT`.
    """)

# --- TAB 2: EXPERIMENT DEEP-DIVE ---
with tab2:
    # --- Sidebar for Stock Selection ---
    stock_name = st.sidebar.radio(
        "Select Stock Regime:",
        ('adani', 'asian'),
        format_func=lambda x: "Adani Enterprises (Volatile)" if x == 'adani' else "Asian Paints (Stable)"
    )

    # --- Find All Experiment Files ---
    stock_dir = Path(f"./{stock_name}/")
    json_files = sorted(glob.glob(f"{stock_dir}/*.json"))
    experiment_names = [Path(f).stem for f in json_files]

    # --- Sidebar for Experiment Selection ---
    selected_experiment_name = st.sidebar.selectbox(
        "Select an Experiment:",
        experiment_names,
        index=len(experiment_names) - 1 # Default to the last (most recent) experiment
    )

    # Re-create the full JSON file path from the selected name
    selected_json_file = stock_dir / f"{selected_experiment_name}.json"

    st.header(f"Results for: `{selected_experiment_name}`")

    # --- Context-Aware Analysis Block ---
    st.subheader("Regime Analysis")
    if stock_name == 'asian':
        st.success(
            "📈 **ASIANPAINTS Regime Analysis: Macro-Sensitive**\n\n"
            "**Conclusion:** This asset is measurably influenced by macroeconomic 'shocks'.\n\n"
            "Our breakthrough (Model 6, `lstm_multi_output_delta_GRU_events_only`) achieved **75.13% directional accuracy** and a **17.6% RMSE improvement** over the baseline. "
            "This success came from using the 'Events-Only' feature set, proving that the model learned to read the 'shock' (`_abs_change`) and 'staleness' (`_days_since_change`) of macro data, while ignoring the forward-filled values as noise."
        )
    elif stock_name == 'adani':
        st.warning(
            "📉 **ADANIENT Regime Analysis: Context-Driven**\n\n"
            "**Conclusion:** This asset is **decoupled from macro features** and driven by its own news narrative.\n\n"
            "The 'Events-Only' macro model *failed* for `ADANIENT` (RMSE 127.56 vs. Naive 126.86). Success was *only* achieved (Model 11, `ADANIENT_FINAL_model_Sentiment_Categories`) "
            "after we added your **multi-hot encoded news categories** (e.g., `is_fraud_allegation`). This model beat the high-volatility baseline by **1.5%** and achieved a **54.3%** derived accuracy, proving that *news context* is the only real signal."
        )

    # --- Load and Display Data ---
    try:
        with open(selected_json_file, 'r') as f:
            data = json.load(f)

        # --- 5. Display Key Metrics ---
        st.subheader("Key Performance Metrics")
        
        col1, col2, col3 = st.columns(3)

        # A. Regression Metrics
        reg_metrics = data.get('regression_metrics_unscaled_delta') or data.get('regression_metrics_unscaled')
        
        if reg_metrics:
            # Check for Naive RMSE (Delta models)
            if 'naive_baseline_rmse' in reg_metrics:
                model_rmse = reg_metrics.get('model_rmse')
                naive_rmse = reg_metrics.get('naive_baseline_rmse')
                improvement = reg_metrics.get('improvement_over_naive_rmse_pct', 0)
                
                col1.metric(
                    label="Model RMSE (Delta)",
                    value=f"{model_rmse:.2f}",
                    delta=f"{improvement:.2f}% vs Naive",
                    delta_color="normal" if improvement > 0 else "inverse"
                )
                col2.metric(label="Naive Baseline RMSE", value=f"{naive_rmse:.2f}")
            
            # Fallback for Absolute Price models
            elif 'rmse' in reg_metrics:
                col1.metric(label="Model RMSE (Absolute)", value=f"{reg_metrics.get('rmse'):.2f}")
                col2.metric(label="Model MAE (Absolute)", value=f"{reg_metrics.get('mae'):.2f}")
                col3.info("This is an 'Absolute Price' model. RMSE is not comparable to a Naive Baseline.")

        # B. Classification Metrics
        class_metrics = data.get('derived_classification_metrics') or data.get('classification_metrics_directional')
        
        if class_metrics:
            acc = class_metrics.get('accuracy', 0) * 100
            f1 = class_metrics.get('f1_score_up', 0)
            
            col3.metric(label="Directional Accuracy", value=f"{acc:.2f}%")
            
            st.metric(label="F1-Score (for 'Up')", value=f"{f1:.2f}")
            
            if 'classification_report' in class_metrics:
                st.text("Classification Report:")
                st.code(class_metrics['classification_report'], language='text')

        # --- 6. Display Plots and Images ---
        st.subheader("Visual Results")
        
        # ### <<< IMAGE LOADING FIX >>> ###
        # Get the base name of the experiment, removing the '_results' suffix
        image_base_name = selected_experiment_name.replace('_results', '')
        
        # Find all .png files that start with this base name
        image_files = sorted(glob.glob(f"{stock_dir}/{image_base_name}*.png"))
        
        if not image_files:
            st.warning(f"No PNG images found matching base name: `{image_base_name}`")
        
        # Dynamically create columns for each image
        if image_files:
            image_cols = st.columns(len(image_files))
            for i, img_file in enumerate(image_files):
                img_path = Path(img_file)
                with image_cols[i]:
                    st.write(img_path.name)
                    st.image(str(img_path), use_column_width=True)

        # --- 7. Display Model Configuration ---
        st.subheader("Model & Feature Configuration")
        
        col1, col2 = st.columns(2)
        
        # Display Tuning Params
        if 'tuning_params' in data:
            with col1.expander("Tuning Hyperparameters", expanded=False):
                st.json(data['tuning_params'])
                
        # Display Feature List
        if 'features_used' in data:
            with col2.expander(f"Features Used ({data.get('feature_count', 'N/A')})", expanded=False):
                st.json(data['features_used'])

        # Display the full JSON data for inspection
        with st.expander("Show Full Raw JSON Data"):
            st.json(data)

    except FileNotFoundError:
        st.error(f"Could not find file: {selected_json_file}")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")