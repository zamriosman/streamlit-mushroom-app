import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Try Prophet import (keeps the app beginner-friendly)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# -------------------------------
# Dummy dataset builder
# -------------------------------
@st.cache_data
def make_dummy_mushroom_timeseries(n_days=180, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    # Simple, smoothish signals + noise
    temp = 23 + 2*np.sin(np.linspace(0, 3*np.pi, n_days)) + rng.normal(0, 0.5, n_days)
    hum  = 72 + 3*np.cos(np.linspace(0, 2*np.pi, n_days)) + rng.normal(0, 1.0, n_days)
    co2  = 420 + 10*np.sin(np.linspace(0, 4*np.pi, n_days)) + rng.normal(0, 3.0, n_days)
    df = pd.DataFrame({
        "timestamp": dates,
        "temperature_C": np.round(temp, 2),
        "humidity_%": np.round(hum, 2),
        "CO2_ppm": np.round(co2, 1),
    })
    return df

df = make_dummy_mushroom_timeseries()

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Explore Data", "Forecast (Prophet)"]
)

# -------------------------------
# Page: Overview (General Information)
# -------------------------------
if page == "Overview":
    st.title("🍄 Mushroom Analytics – Simple App")
    st.subheader("General Information")
    st.markdown(
        """
**What this app shows**

- A tiny, beginner-friendly Streamlit example with:
  - a sidebar for navigation,
  - a built-in dummy mushroom environment dataset,
  - quick charts,
  - and a one-page **Prophet** forecasting demo.

**About the dataset**

- **timestamp**: daily records for ~2 months  
- **temperature_C**: simulated temperature readings in °C  
- **humidity_%**: simulated relative humidity (%)  
- **CO2_ppm**: simulated CO₂ concentration (ppm)

**Why forecasting?**

- In mushroom cultivation, anticipating **temperature**, **humidity**, and **CO₂** trends helps:
  - maintain optimal growing conditions,
  - plan ventilation and misting schedules,
  - and trigger alerts before thresholds are breached.

**How to use the Forecast page**

1. Choose a variable (e.g., temperature_C).  
2. Pick how many days to forecast.  
3. Click **Run Forecast** to generate future values using Prophet.
        """
    )
    st.info("Tip: You can switch pages using the sidebar on the left.")

# -------------------------------
# Page: Explore Data
# -------------------------------
elif page == "Explore Data":
    st.title("📊 Explore Data")
    st.write("Below is a sample of the dataset:")
    st.dataframe(df)

    st.markdown("#### Quick Chart")
    x_col = st.selectbox("X-axis", ["timestamp", "temperature_C", "humidity_%", "CO2_ppm"], index=0)
    y_col = st.selectbox("Y-axis", ["temperature_C", "humidity_%", "CO2_ppm"], index=0)

    chart_type = st.radio("Chart type", ["Line", "Bar", "Scatter"], horizontal=True)

    if chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
    elif chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Page: Forecast (Prophet)
# -------------------------------
elif page == "Forecast (Prophet)":
    st.title("🔮 Forecast with Prophet")

    if not PROPHET_AVAILABLE:
        st.error(
            "Prophet is not installed. Install it with:\n\n"
            "`pip install prophet`  (or)  `conda install -c conda-forge prophet`"
        )
    else:
        st.write("Choose a variable and forecast horizon, then click **Run Forecast**.")

        target = st.selectbox(
            "Variable to forecast",
            ["temperature_C", "humidity_%", "CO2_ppm"],
            index=0
        )
        periods = st.slider("Days to forecast", min_value=7, max_value=60, value=21, step=7)

        # Optional: allow changing frequency (keep daily for beginners)
        freq = "D"

        # Button to run forecast
        if st.button("Run Forecast"):
            # Prepare data for Prophet
            prophet_df = df[["timestamp", target]].rename(columns={"timestamp": "ds", target: "y"})

            # Build and fit model
            model = Prophet()
            model.fit(prophet_df)

            # Future DF and predict
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)

            # Merge for nicer plotting with Plotly
            joined = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
                prophet_df, on="ds", how="left"
            )

            st.markdown(f"#### Forecast result for **{target}** (+{periods} days)")
            # Plotly chart: actuals + forecast + interval
            fig = px.line(joined, x="ds", y=["y", "yhat"], labels={"ds": "Date"}, title=f"Prophet Forecast: {target}")
            # Add uncertainty band
            fig.add_traces(px.scatter(joined, x="ds", y="yhat_upper").data)
            fig.add_traces(px.scatter(joined, x="ds", y="yhat_lower").data)
            fig.data[-1].name = "yhat_lower"
            fig.data[-2].name = "yhat_upper"
            st.plotly_chart(fig, use_container_width=True)

            # Show forecast table (last 10 rows)
            st.markdown("#### Forecast table (tail)")
            st.dataframe(forecast.tail(10))

            with st.expander("What am I looking at?"):
                st.write(
                    """
- **y**: the original historical values (actual data).
- **yhat**: Prophet’s predicted value.
- **yhat_lower / yhat_upper**: uncertainty interval around the prediction.
- For a beginner setup, we used Prophet with default settings (no custom seasonality or holidays).
                    """
                )

        with st.expander("Beginner notes on Prophet"):
            st.write(
                """
- Prophet models trend and seasonality automatically.
- Works well on daily data and handles missing days reasonably.
- You can tune seasonality (daily/weekly/yearly), add holidays, or include **extra regressors** (e.g., temperature while forecasting CO₂).
                """
            )

