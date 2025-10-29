# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional ML imports for forecasting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --- Configuration ---
INPUT_FILE_NAME = "air_quality_dashboard_data.csv"
OUTPUT_FILE_NAME = "monthly_air_quality_data.csv"

# Defined max values for NPI calculation (used for normalization)
MAX_POLLUTANT_VALUES = {
    "PM2.5": 100.0,
    "PM10": 150.0,
    "NO2": 50.0,
    "SO2": 30.0,
    "O3": 70.0,
    "CO": 2.0,
}

# Color scheme for pollutants
POLLUTANT_COLORS = {
    "PM2.5": "#FF6B6B",
    "PM10": "#4ECDC4",
    "NO2": "#45B7D1",
    "SO2": "#96CEB4",
    "O3": "#FFEAA7",
    "CO": "#DDA0DD",
}

# --- Utility / Helper Functions ---


def generate_sample_data(n_rows: int = 1000, start_date: str = "2023-01-01", end_date: str = "2023-12-31"):
    """Generate a realistic-looking sample dataset and save to INPUT_FILE_NAME."""
    date_range = pd.date_range(start=start_date, end=end_date, freq="H")
    dates = np.random.choice(date_range, size=n_rows, replace=True)

    pollutants = list(MAX_POLLUTANT_VALUES.keys())
    pollutant_ranges = {
        "PM2.5": (10, 100),
        "PM10": (20, 150),
        "NO2": (5, 50),
        "SO2": (1, 30),
        "O3": (10, 70),
        "CO": (0.1, 2.0),
    }

    pollutant_data = np.random.choice(pollutants, size=n_rows, p=[0.25, 0.20, 0.18, 0.12, 0.17, 0.08])
    values = [np.random.uniform(*pollutant_ranges[p]) for p in pollutant_data]

    df = pd.DataFrame({"date": dates, "pollutant": pollutant_data, "value_column": values}).sort_values("date")
    df.to_csv(INPUT_FILE_NAME, index=False)
    return df


def calculate_npi(row):
    """Normalized Pollutant Index: pollutant value divided by a pollutant-specific maximum."""
    pollutant = row["pollutant"]
    value = row["value_column"]
    max_val = MAX_POLLUTANT_VALUES.get(pollutant, 1.0)
    return value / max_val


def get_aqi_category(aqi_value: float) -> str:
    """Map AQI value to a category (informative, simplified breaks)."""
    if aqi_value <= 50:
        return "Good"
    if aqi_value <= 100:
        return "Moderate"
    if aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi_value <= 200:
        return "Unhealthy"
    if aqi_value <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def calculate_aqi_from_pollutants_row(row, weights=None):
    """
    Simplified AQI estimator based on weighted pollutant concentrations.
    This is NOT an official EPA AQI computation — it's a combined severity score for visualization.
    """
    if weights is None:
        weights = {"PM2.5": 0.35, "PM10": 0.25, "NO2": 0.15, "SO2": 0.10, "O3": 0.10, "CO": 0.05}
    aqi_val = 0.0
    for p, w in weights.items():
        if p in row.index:
            aqi_val += (row[p] if not np.isnan(row[p]) else 0.0) * w
    return round(aqi_val, 2)


# --- Streamlit UI helpers ---


def apply_custom_css():
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .small-note {
        color: #6c757d;
        font-size: 0.95rem;
    }
    .pollutant-badge {
        display:inline-block;
        padding:6px 10px;
        border-radius:12px;
        color:#fff;
        margin:0 6px 6px 0;
        font-weight:600;
        font-size:0.9rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# --- Data loading / preprocessing (MODIFIED to accept DataFrame) ---


@st.cache_data(ttl=60 * 10)
def preprocess_data(df_input: pd.DataFrame) -> pd.DataFrame:
    """Performs all necessary preprocessing steps on the input DataFrame."""
    df = df_input.copy()
    
    # Validation and cleaning
    required_cols = ['date', 'pollutant', 'value_column']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Uploaded file must contain columns: {', '.join(required_cols)}")
        st.stop()
        
    df.dropna(subset=required_cols, inplace=True)
    
    # Feature engineering
    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day_of_week"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    
    # NPI calculation
    df["npi_score"] = df.apply(calculate_npi, axis=1)
    
    return df


def build_aqi_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot pollutant values into columns per timestamp and compute a combined AQI-like score.
    Returns a dataframe indexed by date with pollutant columns and an 'AQI' column.
    """
    pivot = df.pivot_table(index="date", columns="pollutant", values="value_column", aggfunc="mean")
    # Keep pollutant columns consistent
    for p in MAX_POLLUTANT_VALUES.keys():
        if p not in pivot.columns:
            pivot[p] = np.nan
    pivot = pivot.sort_index()
    pivot_filled = pivot.fillna(method="ffill").fillna(method="bfill").fillna(0)
    pivot_filled["AQI"] = pivot_filled.apply(calculate_aqi_from_pollutants_row, axis=1)
    pivot_filled = pivot_filled.reset_index()
    pivot_filled["month"] = pivot_filled["date"].dt.month
    pivot_filled["month_name"] = pivot_filled["date"].dt.month_name()
    pivot_filled["season"] = pivot_filled["month"].apply(get_season)
    return pivot_filled


# --- Forecasting (simple ML) ---


def train_monthly_forecast_model(monthly_df: pd.DataFrame):
    """
    Train a simple linear regression model to predict monthly average AQI using monthly pollutant means.
    monthly_df should have columns: month, month_name, pollutant columns..., AQI
    """
    # Use pollutant means as features
    feat_cols = [c for c in monthly_df.columns if c in MAX_POLLUTANT_VALUES.keys()]
    if len(monthly_df) < 3 or len(feat_cols) == 0:
        return None, None, None
    X = monthly_df[feat_cols].values
    y = monthly_df["AQI"].values
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    return model, feat_cols, mae


def predict_next_month_aqi(model, last_month_features: pd.Series, feat_cols):
    """Given a trained model and the last month's pollutant means, predict next month AQI."""
    X_next = last_month_features[feat_cols].values.reshape(1, -1)
    return float(model.predict(X_next)[0])


# --- Application ---


def run_air_quality_app():
    st.set_page_config(layout="wide", page_title="Air Quality Dashboard", page_icon="🌍")
    apply_custom_css()

    st.markdown('<h1 class="main-header">🌍 AeroSense: Intelligent Air Quality Analytics Dashboard </h1>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">NPI = Normalized Pollutant Index (0.0 - 1.0). AQI (simplified) shown for quick interpretation.</div>', unsafe_allow_html=True)
    st.write("")

    # --- Sidebar controls (MODIFIED) ---
    st.sidebar.header("Controls")
    
    # NEW: File Uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own CSV data", 
        type="csv", 
        help=f"Must contain columns: date, pollutant, value_column. Falls back to '{INPUT_FILE_NAME}' if empty."
    )
    st.sidebar.markdown("---")
    
    regenerate = st.sidebar.button("🔁 Regenerate sample data (overwrite file)")
    sample_size = st.sidebar.slider("Sample size (rows)", 200, 5000, 1000, step=100)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Date filters")

    # --- Data Loading Logic (MODIFIED) ---
    df_raw = None
    data_source = "Sample Data"

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            data_source = f"Uploaded File: {uploaded_file.name}"
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()
            
    elif regenerate or not os.path.exists(INPUT_FILE_NAME):
        if not os.path.exists(INPUT_FILE_NAME):
             st.sidebar.info("No local data found — generating sample data.")
        
        df_raw = generate_sample_data(n_rows=sample_size)
        st.cache_data.clear() # Clear cache when regenerating
        if regenerate:
            st.sidebar.success("Sample data regenerated.")
        data_source = "Generated Sample Data"
        time.sleep(0.5)
        
    else: # Fallback to local file
        try:
            df_raw = pd.read_csv(INPUT_FILE_NAME)
            data_source = f"Local File: {INPUT_FILE_NAME}"
        except Exception as e:
            st.error(f"Could not load local file '{INPUT_FILE_NAME}'. Error: {e}")
            st.stop()

    st.sidebar.markdown(f"**Data Source:** `{data_source}`")
    
    # --- Preprocessing ---
    if df_raw is None or df_raw.empty:
        st.warning("No data available to process.")
        st.stop()

    # Preprocess the loaded or uploaded DataFrame
    data = preprocess_data(df_raw) 
    
    # Corrected Date input initialization
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()

    # Date range inputs (using dataset limits)
    start_date = st.sidebar.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.sidebar.error("Start date must be <= end date.")
        st.stop()

    # Pollutant filter
    all_pollutants = sorted(data["pollutant"].unique())
    selected_pollutants = st.sidebar.multiselect("Pollutants to include", options=all_pollutants, default=all_pollutants)

    # Aggregation level
    agg_level = st.sidebar.selectbox("Aggregation level", options=["Hourly (raw)", "Daily", "Monthly"], index=2)

    # Filter data
    # FIX: Use the .dt.date accessor on the Timestamp column to compare with the date objects
    filtered = data[
        (data["date"].dt.date >= start_date) &
        (data["date"].dt.date <= end_date) &
        (data["pollutant"].isin(selected_pollutants))
    ].copy()

    if filtered.empty:
        st.warning("No data for the selected filters. Try widening the date range or selecting different pollutants.")
        st.stop()

    # Build AQI timeseries from filtered data
    timeseries = build_aqi_timeseries(filtered)

    # Aggregations for display
    if agg_level == "Daily":
        agg_df = timeseries.copy()
        agg_df["date"] = agg_df["date"].dt.floor("D")
        agg_df = agg_df.groupby("date").mean(numeric_only=True).reset_index()
    elif agg_level == "Monthly":
        agg_df = timeseries.copy()
        agg_df["month"] = agg_df["date"].dt.month
        agg_df = agg_df.groupby("month").mean(numeric_only=True).reset_index()
        # add month_name for monthly view
        agg_df["month_name"] = agg_df["month"].apply(lambda m: pd.Timestamp(2023, int(m), 1).strftime("%B"))
    else:  # hourly/raw
        agg_df = timeseries.copy()

    # --- Top-level KPIs ---
    st.header("Key Metrics")
    avg_npi = filtered["npi_score"].mean()
    avg_aqi = agg_df["AQI"].mean()
    peak_aqi = agg_df["AQI"].max()
    lowest_aqi = agg_df["AQI"].min()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{len(filtered):,}")
    k2.metric("Avg NPI (selected)", f"{avg_npi:.4f}")
    k3.metric("Avg AQI (selected)", f"{avg_aqi:.1f}")
    k4.metric("Peak AQI (selected)", f"{peak_aqi:.1f}")

    # AQI category + gauge
    aqi_cat = get_aqi_category(avg_aqi)
    st.markdown(f"**AQI Category (avg):** `{aqi_cat}`")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_aqi,
        title={"text": "Average AQI (simplified)"},
        gauge={
            "axis": {"range": [0, max(300, peak_aqi * 1.2)]},
            "steps": [
                {"range": [0, 50], "color": "green"},
                {"range": [50, 100], "color": "yellow"},
                {"range": [100, 150], "color": "orange"},
                {"range": [150, 200], "color": "red"},
                {"range": [200, 300], "color": "purple"},
            ],
            "bar": {"color": "darkblue"}
        }
    ))
    st.plotly_chart(gauge, use_container_width=True)

    # --- Tabs for organization ---
    tab_overview, tab_trends, tab_analysis, tab_insights = st.tabs(["Overview", "Trends", "Analysis", "Forecast & Insights"])

    # ----------------- Overview Tab (FIXED LAYOUT) -----------------
    with tab_overview:
        st.subheader("Data Overview")

        # 1. Selected Filters Section
        st.write("**Selected filters**")
        col_filters = st.columns(3)
        with col_filters[0]: st.write(f"- Date range: {start_date} → {end_date}")
        with col_filters[1]: st.write(f"- Pollutants: {', '.join(selected_pollutants)}")
        with col_filters[2]: st.write(f"- Aggregation: {agg_level}")
        st.markdown("---")
        
        # 2. Pollutant Counts and Data Table ALIGNED (The requested fix)
        col_counts, col_data = st.columns([1, 2])
        
        with col_counts:
            st.write("**Pollutant Counts (selected)**")
            counts = filtered["pollutant"].value_counts()
            
            # Display counts using the styled badges
            for p, cnt in counts.items():
                color = POLLUTANT_COLORS.get(p, "#888")
                st.markdown(f'<span class="pollutant-badge" style="background-color:{color}">{p}: {cnt}</span>', unsafe_allow_html=True)
            st.write("") 

        with col_data:
            st.write("**Recent Raw Samples**")
            # Display the DataFrame
            st.dataframe(filtered.sort_values("date", ascending=False).head(200), use_container_width=True)

    # ----------------- Trends Tab -----------------
    with tab_trends:
        st.subheader("Trend Visualizations")

        # Monthly / hourly AQI line
        if agg_level == "Monthly":
            x = agg_df["month_name"]
        else:
            x = agg_df["date"]

        fig_aqi = px.line(agg_df, x=x, y="AQI", title="AQI Trend", labels={"AQI": "AQI", "date": "Date"})
        fig_aqi.update_traces(line=dict(width=3, color="#1f77b4"))
        st.plotly_chart(fig_aqi, use_container_width=True)

        # Pollutant stacked bar (monthly) or per selected agg
        if agg_level == "Monthly":
            pollutant_cols = [c for c in agg_df.columns if c in MAX_POLLUTANT_VALUES.keys()]
            df_stack = agg_df[["month_name"] + pollutant_cols].melt(id_vars="month_name", var_name="pollutant", value_name="value")
            fig_stack = px.bar(df_stack, x="month_name", y="value", color="pollutant", title="Pollutant contribution (by month)", color_discrete_map=POLLUTANT_COLORS)
            fig_stack.update_layout(barmode="stack", xaxis_title="Month", yaxis_title="Concentration")
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.info("Switch to Monthly aggregation to view pollutant stacked contributions.")

        # Hourly pattern if enough hours available
        if "hour" in filtered.columns:
            hourly = filtered.groupby(["hour", "pollutant"])["npi_score"].mean().reset_index()
            fig_hour = px.line(hourly, x="hour", y="npi_score", color="pollutant", title="Hourly NPI Patterns", color_discrete_map=POLLUTANT_COLORS)
            st.plotly_chart(fig_hour, use_container_width=True)

    # ----------------- Analysis Tab -----------------
    with tab_analysis:
        st.subheader("Detailed Analysis")

        # Correlation heatmap between pollutants (based on filtered raw values)
        st.markdown("**Pollutant correlation matrix**")
        pollutant_pivot = filtered.pivot_table(index="date", columns="pollutant", values="value_column")
        corr = pollutant_pivot.corr()
        if corr.isnull().all().all():
            st.info("Not enough pollutant overlap to compute correlations.")
        else:
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

        # Box plots / distribution by pollutant
        st.markdown("**NPI distribution by pollutant**")
        fig_box = px.box(filtered, x="pollutant", y="npi_score", color="pollutant", color_discrete_map=POLLUTANT_COLORS, title="NPI distribution")
        st.plotly_chart(fig_box, use_container_width=True)

        # Monthly heatmap of NPI by pollutant (if monthly range present)
        st.markdown("**Monthly NPI Heatmap**")
        heat = filtered.groupby([filtered["date"].dt.month.rename("month"), "pollutant"])["npi_score"].mean().unstack()
        if heat.empty:
            st.info("Insufficient data to show monthly heatmap.")
        else:
            fig_heat = px.imshow(heat.T, labels=dict(x="Month", y="Pollutant", color="Avg NPI"), x=heat.index, y=heat.columns, text_auto=".2f", aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True)

    # ----------------- Forecast & Insights Tab -----------------
    with tab_insights:
        st.subheader("Forecast & Actionable Insights")

        # Prepare monthly aggregated data for forecasting
        monthly_for_model = timeseries.copy()
        monthly_for_model["month"] = monthly_for_model["date"].dt.month
        
        # Calculate monthly aggregation using numeric_only=True
        monthly_agg = monthly_for_model.groupby("month").mean(numeric_only=True).reset_index()
        
        # ensure pollutant columns exist
        for p in MAX_POLLUTANT_VALUES.keys():
            if p not in monthly_agg.columns:
                monthly_agg[p] = 0.0

        # Train forecast model
        model, feat_cols, mae = train_monthly_forecast_model(monthly_agg)
        if model is None:
            st.info("Not enough data or features to train forecasting model. Need at least 3 monthly records.")
        else:
            st.markdown(f"Model trained on {len(monthly_agg)} months. MAE on training data: **{mae:.2f}**")
            # Predict next month
            last_month = monthly_agg.sort_values("month").iloc[-1]
            predicted_next = predict_next_month_aqi(model, last_month, feat_cols)
            next_month_num = int(last_month["month"]) + 1
            # wrap-around month
            if next_month_num == 13:
                next_month_num = 1
            next_month_name = pd.Timestamp(2023, next_month_num, 1).strftime("%B")
            st.metric("Predicted next month (avg) AQI", f"{predicted_next:.1f}", delta=f"for {next_month_name}")

            # Show trends: actual vs predicted (for training months)
            X_train = monthly_agg[feat_cols].values
            y_train = monthly_agg["AQI"].values
            y_pred_train = model.predict(X_train)
            plot_df = pd.DataFrame({
                "month": monthly_agg["month"],
                "actual": y_train,
                "predicted": y_pred_train
            }).melt(id_vars="month", var_name="type", value_name="AQI")
            plot_df["month_name"] = plot_df["month"].apply(lambda m: pd.Timestamp(2023, int(m), 1).strftime("%b"))
            fig_pred = px.line(plot_df, x="month_name", y="AQI", color="type", title="Actual vs Predicted Monthly AQI (training)")
            st.plotly_chart(fig_pred, use_container_width=True)

            # Simple textual health guidance
            st.subheader("Health Guidance (based on predicted AQI)")
            cat = get_aqi_category(predicted_next)
            if predicted_next <= 50:
                st.success(f"Predicted AQI is {predicted_next:.1f} ({cat}). Air quality expected to be good.")
            elif predicted_next <= 100:
                st.info(f"Predicted AQI is {predicted_next:.1f} ({cat}). Moderate — sensitive groups should be cautious.")
            elif predicted_next <= 150:
                st.warning(f"Predicted AQI is {predicted_next:.1f} ({cat}). Unhealthy for sensitive groups — limit outdoor exposure.")
            elif predicted_next <= 200:
                st.error(f"Predicted AQI is {predicted_next:.1f} ({cat}). Unhealthy — reduce outdoor activity.")
            else:
                st.error(f"Predicted AQI is {predicted_next:.1f} ({cat}). Hazardous — stay indoors and use air purifiers if available.")

    # ----------------- Download / Export -----------------
    st.header("Export / Save Results")
    monthly_export = filtered.copy()
    monthly_export["month"] = monthly_export["date"].dt.month
    monthly_export = monthly_export.groupby(["month", "month_name", "pollutant"])["npi_score"].mean().reset_index().sort_values(["month", "pollutant"])
    st.subheader("Processed monthly NPI preview")
    st.dataframe(monthly_export, use_container_width=True)

    csv = monthly_export.to_csv(index=False)
    st.download_button("⬇️ Download monthly NPI CSV", data=csv, file_name=OUTPUT_FILE_NAME, mime="text/csv")

    # Also allow downloading the full AQI timeseries
    st.subheader("Full AQI timeseries (derived)")
    st.dataframe(agg_df.head(200), use_container_width=True)
    aqi_csv = agg_df.to_csv(index=False)
    st.download_button("⬇️ Download AQI timeseries CSV", data=aqi_csv, file_name="aqi_timeseries.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("Built with ❤️  •  Replace or edit `INPUT_FILE_NAME` to use your own data. Official AQI calculation requires pollutant-specific piecewise breakpoints — this app uses a simplified combined score for demonstration and analysis.")


# --- Entry point ---
if __name__ == "__main__":
    run_air_quality_app()
