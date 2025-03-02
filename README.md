# GARCH Volatility Forecasting Playground: Modeling Time-Varying Volatility

## 1. What Is This?

This interactive application demonstrates how to model and forecast time-varying volatility using **GARCH models**. In particular, it fits a GARCH(1,1) (or more generally, GARCH(p,q)) model to real-world financial data (e.g., S&P 500 returns) sourced from Yahoo Finance and forecasts future volatility.

- **What It Is:**  
  GARCH models capture the phenomenon of **volatility clustering** by modeling the conditional variance of returns as a function of past squared returns and past variances. The most common specification, GARCH(1,1), has the form:  
  \[
    \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
  \]
  where \(\sigma_t^2\) is the conditional variance, \(\epsilon_{t-1}\) is the previous period’s return shock, and \(\omega\), \(\alpha\), and \(\beta\) are parameters to be estimated.

- **Why Teach It:**  
  Understanding GARCH models is critical for grasping how volatility behaves in financial markets. They are widely used in risk management, option pricing, and portfolio construction because they capture the dynamic, time-varying nature of risk.

- **Example:**  
  Fit a GARCH(1,1) model to daily S&P 500 returns, then forecast next-week volatility. The playground provides interactive tools to change the data period, model parameters (p and q), and forecast horizon.

**Note:** This tool is for educational purposes only. Accuracy is not guaranteed, and the computed forecasts do not represent actual market values or future performance. The author is Luís Simões da Cunha.

## 2. Setting Up a Local Development Environment

### 2.1 Prerequisites

1. **A computer** (Windows, macOS, or Linux).
2. **Python 3.9 or higher** (Python 3.12 preferred, but anything 3.9+ should work).  
   - If you do not have Python installed, visit [python.org/downloads](https://www.python.org/downloads/) to install the latest version.
3. **Visual Studio Code (VS Code)**  
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)
4. **Git** (optional, but recommended for cloning the repository).  
   - Install from [git-scm.com/downloads](https://git-scm.com/downloads)

### 2.2 Downloading the Project

#### Option 1: Cloning via Git (Recommended)

1. Open **Terminal** (macOS/Linux) or **Command Prompt** / **PowerShell** (Windows).
2. Navigate to the folder where you want to download the project:
   ```bash
   cd Documents
   ```
3. Run the following command:
   ```bash
   git clone https://github.com/yourusername/garch_vol_playground.git
   ```
4. Enter the project folder:
   ```bash
   cd garch_vol_playground
   ```

#### Option 2: Download as ZIP

1. Visit [https://github.com/yourusername/garch_vol_playground](https://github.com/yourusername/garch_vol_playground)
2. Click **Code > Download ZIP**.
3. Extract the ZIP file into a local folder.

### 2.3 Creating a Virtual Environment

It is recommended to use a virtual environment (`venv`) to manage dependencies:

1. Open **VS Code** and navigate to the project folder.
2. Open the integrated terminal (`Ctrl + ~` in VS Code or via `Terminal > New Terminal`).
3. Run the following commands to create and activate a virtual environment:
   ```bash
   python -m venv venv
   ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

### 2.4 Installing Dependencies

After activating the virtual environment, install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A sample **requirements.txt** might include:
```txt
streamlit
numpy
pandas
matplotlib
yfinance
arch
```

This installs libraries such as:
- **Streamlit** (for the interactive UI)
- **NumPy** and **Pandas** (for numerical and data manipulations)
- **Matplotlib** (for plotting)
- **yfinance** (to fetch data from Yahoo Finance)
- **arch** (for GARCH modeling)

## 3. Running the Application

To launch the GARCH Volatility Forecasting Playground, execute:

```bash
streamlit run garch_vol_playground.py
```

This should open a new tab in your web browser with the interactive tool. If it does not open automatically, check the terminal for a URL (e.g., `http://localhost:8501`) and open it manually.

### 3.1 Troubleshooting

- **ModuleNotFoundError:** Ensure the virtual environment is activated (`venv\Scripts\activate` on Windows or `source venv/bin/activate` on macOS/Linux).
- **Python not recognized:** Make sure Python is installed and added to your system's PATH.
- **Browser does not open automatically:** Manually enter the `http://localhost:8501` URL in your browser.

## 4. Editing the Code

If you want to make modifications:
1. Open `garch_vol_playground.py` in **VS Code**.
2. Modify the code as needed.
3. Restart the Streamlit app after changes (press `Ctrl + C` in the terminal to stop, then rerun `streamlit run garch_vol_playground.py`).

## 5. Additional Resources

- **Streamlit Documentation:** [docs.streamlit.io](https://docs.streamlit.io)
- **GARCH Model Overview:** [Investopedia: GARCH](https://www.investopedia.com/terms/g/garch.asp)
- **ARCH Package Documentation:** [arch.readthedocs.io](https://arch.readthedocs.io/en/latest/)

## 6. Support

For issues or suggestions, open an **Issue** on GitHub:  
[https://github.com/yourusername/garch_vol_playground/issues](https://github.com/yourusername/garch_vol_playground/issues)

---

*Happy exploring GARCH models and the dynamic world of volatility forecasting!*
