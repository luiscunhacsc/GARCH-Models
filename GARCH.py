import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
import datetime

#######################################
# 1) Utility & Callback Functions
#######################################

def load_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.dropna(inplace=True)
    # Use 'Adj Close' if available; otherwise, fall back to 'Close'
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['Return'] = 100.0 * df[price_col].pct_change()  # daily return in %
    df.dropna(inplace=True)
    return df

def get_last_trading_day():
    """
    Returns the last weekday (Monday–Friday). 
    Does not handle official market holidays, only weekends.
    """
    today = datetime.date.today()
    # If today is Saturday (weekday=5) or Sunday (weekday=6), roll back
    while today.weekday() > 4:  # Monday=0, Sunday=6
        today -= datetime.timedelta(days=1)
    return today

def fit_garch_model(returns, p=1, q=1):
    """
    Fit a GARCH(p,q) model (default GARCH(1,1)) to the given returns series.
    Returns the fitted model results.
    """
    # arch_model expects a Series of returns (in %), typically
    model = arch_model(returns, vol='Garch', p=p, q=q, dist='normal')
    res = model.fit(update_freq=5, disp='off')
    return res

def forecast_volatility(fitted_model, horizon=5):
    """
    Forecast the next 'horizon' days of volatility (variance) using the fitted model.
    Returns a DataFrame with the forecasted variances and standard deviations.
    """
    forecasts = fitted_model.forecast(horizon=horizon)
    # Extract the forecasted variance from the final row
    var_forecast = forecasts.variance.iloc[-1]  # row: last in-sample, columns: each step
    # Convert to stdev (volatility)
    std_forecast = np.sqrt(var_forecast)
    return var_forecast, std_forecast

def reset_parameters():
    st.session_state["ticker"] = "^GSPC"
    st.session_state["start_date"] = datetime.date(2018, 1, 1)
    st.session_state["end_date"] = datetime.date(2023, 1, 1)
    st.session_state["garch_p"] = 1
    st.session_state["garch_q"] = 1
    st.session_state["forecast_horizon"] = 5

#######################################
# 2) Configure the Streamlit App
#######################################
st.set_page_config(layout="wide")
st.title("📊 GARCH(1,1) Volatility Forecasting Playground")
st.markdown("Explore how to model and forecast time-varying volatility using real financial data with GARCH models.")

#######################################
# 3) Sidebar for Inputs
#######################################
with st.sidebar:
    st.header("⚙️ Parameters")
    st.button("↺ Reset Parameters", on_click=reset_parameters)

    st.markdown("### Data Selection")
    ticker = st.text_input("Ticker (e.g., ^GSPC for S&P 500)", "^GSPC", key="ticker")
    start_date = st.date_input("Start Date", datetime.date(2018, 1, 1), key="start_date")
    end_date = st.date_input(
        "End Date", 
        get_last_trading_day(),  # default to last weekday
        key="end_date"
    )

    st.markdown("### GARCH Model Parameters")
    garch_p = st.slider("GARCH(p,q): p", 1, 3, 1, key="garch_p")
    garch_q = st.slider("GARCH(p,q): q", 1, 3, 1, key="garch_q")

    st.markdown("### Forecast Horizon")
    forecast_horizon = st.slider("Days to Forecast", 1, 20, 5, key="forecast_horizon")

    st.markdown("---")
    st.markdown(
    """
    **Disclaimer**  
    *This tool is for educational purposes only. Accuracy is not guaranteed, and 
    the computed forecasts do not represent actual market values or future performance. 
    The author is Luís Simões da Cunha.*
    """
    )
    st.markdown("""
    <div style="margin-top: 20px;">
        <a href="https://creativecommons.org/licenses/by-nc/4.0/deed.en" target="_blank">
            <img src="https://licensebuttons.net/l/by-nc/4.0/88x31.png" alt="CC BY-NC 4.0">
        </a>
        <br>
        <span style="font-size: 0.8em;">By Luís Simões da Cunha</span>
    </div>
    """, unsafe_allow_html=True)

#######################################
# 4) Create Tabs
#######################################
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎮 Interactive Tool", 
    "📚 Theory Behind GARCH", 
    "📖 Comprehensive Tutorial", 
    "🛠️ Practical Labs",
    "🧠 The Very Basics of Volatility"
])

#######################################
# Tab 1: Interactive Tool
#######################################
with tab1:
    st.subheader("Interactive GARCH Model Fitting & Forecasting")

    # 1) Load data
    df = load_data(ticker, start_date, end_date)
    st.markdown(f"**Loaded {len(df)} trading days** for `{ticker}` from {start_date} to {end_date}.")
    
    # 2) Fit GARCH model
    res = fit_garch_model(df['Return'], p=garch_p, q=garch_q)
    
    # 3) Forecast
    var_fcast, std_fcast = forecast_volatility(res, horizon=forecast_horizon)

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Model Summary")
        # Convert the summary object to a string
        summary_text = res.summary().as_text()
        # Render it in a code block for better readability
        st.markdown(f"```text\n{summary_text}\n```")
    with col2:
        st.markdown("### Forecast Results")
        fcast_df = pd.DataFrame({
            "Day": np.arange(1, forecast_horizon+1),
            "Forecasted Var": var_fcast.values,
            "Forecasted Vol": std_fcast.values
        })
        st.dataframe(fcast_df)

    # 4) Plot historical returns + conditional volatility (in-sample)
    st.markdown("### Historical Returns and In-Sample Volatility")
    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    # Plot daily returns
    ax[0].plot(df.index, df['Return'], color='blue', linewidth=1)
    ax[0].set_title("Daily Returns (%)")
    ax[0].grid(alpha=0.3)

    # In-sample conditional volatility from fitted model
    cond_vol = res.conditional_volatility
    ax[1].plot(cond_vol.index, cond_vol, color='red', linewidth=1)
    ax[1].set_title("In-Sample Conditional Volatility (%)")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

#######################################
# Tab 2: Theory Behind GARCH
#######################################
with tab2:
    st.markdown(r"""
## GARCH Models: A Primer

**What is GARCH?**  
G**eneralized** A**utoregressive** C**onditional** H**eteroskedasticity** models the idea that 
**volatility is time-varying** and exhibits **clustering**: large changes in returns tend to be followed 
by large changes (of either sign), and small changes tend to follow small changes.

**GARCH(1,1) Equation:**  
$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

- $\sigma_t^2$ is the conditional variance at time $t$.  
- $\epsilon_{t-1}^2$ is the squared return shock (residual) from the previous period.  
- $\sigma_{t-1}^2$ is the previous period's conditional variance.  
- $\alpha$ and $\beta$ capture how shocks and past variance affect current variance.

**Why GARCH(1,1)?**  
- It's the most commonly used GARCH specification—simple yet effective at capturing volatility clustering.  
- More advanced variants (EGARCH, GJR-GARCH, etc.) handle asymmetries or leverage effects.
""")

with tab3:
    st.markdown(r"""
## Comprehensive Tutorial on GARCH(1,1)

1. **Gather Time-Series Data**  
   Typically, daily returns $r_t$ of a financial asset (e.g., S&P 500). Convert prices to returns (e.g., percentage changes).

2. **Model Setup**  
   A GARCH(1,1) process for returns $r_t$ might be:

   $$
   r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)
   $$
   $$
   \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
   $$

   - $\omega, \alpha, \beta$ are parameters to be estimated.

3. **Fitting the Model**  
   Use Maximum Likelihood Estimation (MLE) to estimate parameters. The `arch` library in Python automates this.

4. **Forecasting Volatility**  
   Once fitted, we predict future conditional variance $\sigma_{t+h}^2$ over a chosen horizon $h$ (e.g., next 5 days).
   The forecasted volatility is then:
   $$
   \text{Forecasted Volatility} = \sqrt{\text{Forecasted Variance}}
   $$

5. **Interpretation**  
   - Large $\alpha$ means strong reaction to recent shocks.  
   - Large $\beta$ means persistent volatility.  
   - If $\alpha + \beta$ is near 1, volatility shocks die out slowly.

**Use Cases:**  
- Risk management (VaR/ES calculations)  
- Option pricing (stochastic volatility)  
- Portfolio construction (volatility targeting)
""")



#######################################
# Tab 3: Comprehensive Tutorial
#######################################
with tab3:
    st.markdown(r"""
## Comprehensive Tutorial on GARCH(1,1)

1. **Gather Time-Series Data**  
   Typically, daily returns $r_t$ of a financial asset (e.g., S&P 500). Convert prices to returns (e.g., percentage changes).

2. **Model Setup**  
   A GARCH(1,1) process for returns $r_t$ might be:

   $$
   r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)
   $$
   $$
   \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
   $$

   - $\omega, \alpha, \beta$ are parameters to be estimated.

3. **Fitting the Model**  
   Use Maximum Likelihood Estimation (MLE) to estimate parameters. The `arch` library in Python automates this.

4. **Forecasting Volatility**  
   Once fitted, we predict future conditional variance $\sigma_{t+h}^2$ over a chosen horizon $h$ (e.g., next 5 days).
   The forecasted volatility is then:
   $$
   \text{Forecasted Volatility} = \sqrt{\text{Forecasted Variance}}
   $$

5. **Interpretation**  
   - Large $\alpha$ means strong reaction to recent shocks.  
   - Large $\beta$ means persistent volatility.  
   - If $\alpha + \beta$ is near 1, volatility shocks die out slowly.

**Use Cases:**  
- Risk management (VaR/ES calculations)  
- Option pricing (stochastic volatility)  
- Portfolio construction (volatility targeting)
""")
    
#######################################
# Tab 4: Practical Labs
#######################################
with tab4:
    st.header("🔬 Practical GARCH Labs")
    st.markdown("""
    **Lab 1: S&P 500 Volatility Clustering**  
    - Fit a GARCH(1,1) to the S&P 500 daily returns (default).  
    - Observe how well the model captures volatility spikes.

    **Lab 2: Changing p, q**  
    - Increase `p` or `q` in the sidebar.  
    - Compare AIC/BIC or the in-sample fit to see if a higher-order GARCH is beneficial.

    **Lab 3: Different Tickers**  
    - Replace `^GSPC` with another index or stock (e.g., `AAPL`).  
    - Observe how volatility clustering differs across assets.
    """)

#######################################
# Tab 5: The Very Basics of Volatility
#######################################
with tab5:
    st.header("🧠 The Very Basics of Volatility")
    st.markdown("""
    **Volatility** refers to the degree of variation in a trading price over time—often measured by standard deviation or variance of returns.
    
    - High volatility = large, unpredictable price swings.
    - Low volatility = relatively stable price movements.

    **Why does it matter?**  
    - Risk management: Investors care about potential drawdowns.
    - Pricing derivatives: Volatility is a key input in option pricing.
    - Economic signals: Surging volatility often indicates market stress.

    **Volatility Clustering**  
    - Empirical phenomenon where large changes tend to cluster together.
    - GARCH models are designed to capture and forecast this behavior in financial time series.
    """)

