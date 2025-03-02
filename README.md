# GARCH Volatility Forecasting Playground: Modeling Time-Varying Volatility

## 1. What Is This?

This interactive application demonstrates how to model and forecast time-varying volatility using **GARCH models**. In this playground, a GARCH(1,1) model (or more generally, a GARCH(p,q) model) is fitted to real-world financial data (e.g., daily S&P 500 returns fetched from Yahoo Finance) and is used to forecast future volatility over a specified horizon.

- **What It Is:**  
  GARCH models capture **volatility clustering** by modeling the conditional variance of returns as a function of past squared return shocks and past variances. The most common specification, GARCH(1,1), is given by:

  $$
  \sigma_t^2 = \omega + \alpha\, \epsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2
  $$

  where:  
  - \(\sigma_t^2\) is the conditional variance at time \(t\),  
  - \(\epsilon_{t-1}\) is the return shock from the previous period,  
  - \(\omega\), \(\alpha\), and \(\beta\) are parameters to be estimated.

- **Why Teach It:**  
  Understanding GARCH models is critical for grasping how volatility evolves over time. They are widely used in risk management, option pricing, and portfolio construction because they effectively capture the dynamic, time-varying nature of market risk.

- **Example:**  
  The playground allows you to fit a GARCH(1,1) model to daily S&P 500 returns and forecast the next week's volatility. Interactive tools let you adjust the data period, the model parameters (p and q), and the forecast horizon.

**Important Legal Notice:**  
*This tool is for educational purposes only. No accuracy guarantees are provided, and the computed forecasts do not represent actual market values or future performance. The author, Luís Simões da Cunha, disclaims any liability for losses or damages resulting from the use of this tool.*

---

## 2. Setting Up a Local Development Environment

### 2.1 Prerequisites

1. **A Computer:** Works on Windows, macOS, or Linux.
2. **Python 3.9 or Higher:** Python 3.12 is preferred, but any version 3.9+ should work.  
   - [Download Python](https://www.python.org/downloads/)
3. **Visual Studio Code (VS Code):**  
   - [Download VS Code](https://code.visualstudio.com/)
4. **Git:** (Optional but recommended for cloning the repository.)  
   - [Download Git](https://git-scm.com/downloads)

### 2.2 Downloading the Project

#### Option 1: Cloning via Git (Recommended)

1. Open **Terminal** (macOS/Linux) or **Command Prompt/PowerShell** (Windows).
2. Navigate to your desired folder:
   ```bash
   cd Documents
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/garch_vol_playground.git
   ```
4. Change to the project directory:
   ```bash
   cd garch_vol_playground
   ```

#### Option 2: Download as ZIP

1. Visit [https://github.com/yourusername/garch_vol_playground](https://github.com/yourusername/garch_vol_playground)
2. Click **Code > Download ZIP**.
3. Extract the ZIP file to a local folder.

### 2.3 Creating a Virtual Environment

Using a virtual environment is recommended to manage dependencies:

1. Open **VS Code** and navigate to the project folder.
2. Open the integrated terminal (e.g., `Ctrl + ~` or via **Terminal > New Terminal**).
3. Run the following commands:
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

After activating your virtual environment, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A sample **requirements.txt** for this project might include:

```txt
streamlit
numpy
pandas
matplotlib
yfinance
arch
```

This installs:
- **Streamlit** for the interactive UI,
- **NumPy** and **Pandas** for data manipulation,
- **Matplotlib** for plotting,
- **yfinance** to fetch financial data,
- **arch** for GARCH modeling.

---

## 3. Running the Application

To launch the GARCH Volatility Forecasting Playground, run:

```bash
streamlit run garch_vol_playground.py
```

This should automatically open a new browser tab with the interactive tool. If not, check the terminal for a URL (e.g., `http://localhost:8501`) and open it manually.

### 3.1 Troubleshooting

- **ModuleNotFoundError:** Ensure your virtual environment is activated (`venv\Scripts\activate` on Windows or `source venv/bin/activate` on macOS/Linux).
- **Python Not Recognized:** Verify Python is installed and properly added to your PATH.
- **Browser Does Not Open Automatically:** Manually enter the provided URL from the terminal.

---

## 4. Editing the Code

To modify the application:
1. Open `garch_vol_playground.py` in **VS Code**.
2. Make your desired changes.
3. Restart the Streamlit app by pressing `Ctrl + C` in the terminal, then run:
   ```bash
   streamlit run garch_vol_playground.py
   ```

---

## 5. Additional Resources

- **Streamlit Documentation:** [docs.streamlit.io](https://docs.streamlit.io)
- **GARCH Model Overview:** [Investopedia: GARCH](https://www.investopedia.com/terms/g/garch.asp)
- **ARCH Package Documentation:** [arch.readthedocs.io](https://arch.readthedocs.io/en/latest/)

---

## 6. Support

For issues or suggestions, please open an **Issue** on GitHub:

[https://github.com/yourusername/garch_vol_playground/issues](https://github.com/yourusername/garch_vol_playground/issues)

---

*Happy exploring GARCH models and the dynamic world of volatility forecasting!*

---

### Legal Disclaimer

This tool is provided **"as is"** for educational purposes only. No warranty, expressed or implied, is given regarding the accuracy or reliability of the forecasts and analyses produced by this tool. The author, Luís Simões da Cunha, shall not be liable for any errors, omissions, or damages arising from its use.
