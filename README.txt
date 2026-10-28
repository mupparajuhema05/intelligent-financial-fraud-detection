Intelligent Financial Fraud Detection Using Python

Files:
- app.py: Streamlit webpage and machine-learning workflow
- requirements.txt: required Python packages

Run locally:
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run: pip install -r requirements.txt
4. Run: streamlit run app.py
5. Open the local URL printed in the terminal.

CSV format:
- One row per transaction.
- One column must contain the target labels: 0 = legitimate, 1 = fraud.
- Other columns are used as model features.
- The sample data is synthetic and is only for demonstration.

Privacy and limitations:
- Do not upload real customer financial data to an unapproved environment.
- This is an educational prototype, not a production fraud detection system.
- Predictions should be reviewed and must not be treated as proof of fraud.
