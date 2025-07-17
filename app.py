# File: bsdss_app.py (Final, Corrected, and Robust Version)

# ==============================================================================
# Business Sustainability Decision Support System (BSDSS)
# ==============================================================================

# --- 1. Imports ---
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# --- 2. Caching and Model Loading ---

@st.cache_resource
def load_sentiment_model():
    """
    Loads the FinBERT model and tokenizer for sentiment analysis.
    Uses caching to prevent reloading on every run.
    """
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

# --- 3. Data Ingestion & Feature Engineering Module ---

@st.cache_data(ttl=3600)
def get_financial_data(ticker_symbol):
    """
    Fetches financial statements with more robust error checking.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        if not ticker.info or 'shortName' not in ticker.info:
            st.error(f"Could not find a valid company for ticker '{ticker_symbol}'. The symbol may be incorrect or fully delisted.")
            return None, None
        
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        
        if income_stmt.empty or balance_sheet.empty:
            company_name = ticker.info.get('shortName', ticker_symbol)
            st.warning(f"Successfully found '{company_name}', but it has no financial statement data available on Yahoo Finance.")
            return None, None
            
        return income_stmt, balance_sheet
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data for {ticker_symbol}: {e}")
        return None, None

def find_financial_statement_item(statement, possible_keys):
    """
    Searches a financial statement for an item using a list of possible names.
    """
    for key in possible_keys:
        if key in statement.index:
            return statement.loc[key]
    return None

def calculate_financial_ratios(income_stmt, balance_sheet):
    """
    Calculates key financial ratios. Includes robust logic for negative equity.
    """
    ratios = {}
    try:
        latest_year = income_stmt.columns[0]

        # Liquidity: Current Ratio
        current_assets_series = find_financial_statement_item(balance_sheet, ['Current Assets', 'Total Current Assets'])
        current_liabilities_series = find_financial_statement_item(balance_sheet, ['Current Liabilities', 'Total Current Liabilities'])
        if current_assets_series is not None and current_liabilities_series is not None:
            ratios['current_ratio'] = current_assets_series[latest_year] / current_liabilities_series[latest_year] if current_liabilities_series[latest_year] else np.nan
        else:
            ratios['current_ratio'] = np.nan

        # Leverage: Debt-to-Equity Ratio (with robust negative equity handling)
        total_liabilities_series = find_financial_statement_item(balance_sheet, ['Total Liab', 'Total Liabilities', 'Total Liabilities Net Minority Interest'])
        total_equity_series = find_financial_statement_item(balance_sheet, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest'])
        if total_liabilities_series is not None and total_equity_series is not None:
            total_liabilities = total_liabilities_series[latest_year]
            total_equity = total_equity_series[latest_year]
            if total_equity > 0:
                ratios['debt_to_equity'] = total_liabilities / total_equity
            else:
                ratios['debt_to_equity'] = 99.0 # Assign a large penalty for negative equity
        else:
            ratios['debt_to_equity'] = np.nan

        # Profitability: Profit Margin
        net_income_series = find_financial_statement_item(income_stmt, ['Net Income', 'Net Income From Continuing Ops'])
        total_revenue_series = find_financial_statement_item(income_stmt, ['Total Revenue', 'Total Operating Revenue'])
        if net_income_series is not None and total_revenue_series is not None:
            ratios['profit_margin'] = net_income_series[latest_year] / total_revenue_series[latest_year] if total_revenue_series[latest_year] else np.nan
        else:
            ratios['profit_margin'] = np.nan
            
    except Exception as e:
        st.warning(f"Could not calculate all ratios due to a data issue: {e}")
    return ratios

@st.cache_data(ttl=3600, show_spinner=False)
def get_news_sentiment(ticker_symbol, _sentiment_pipeline):
    """
    Fetches news and calculates sentiment. Handles items missing a 'title'.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news
        if not news: return 0.0
        
        headlines = [item['title'] for item in news if 'title' in item]
        if not headlines: return 0.0

        sentiments = _sentiment_pipeline(headlines[:10])
        score = sum(s['score'] if s['label'] == 'positive' else -s['score'] for s in sentiments if s['label'] != 'neutral')
        return score / len(sentiments) if sentiments else 0.0
    except Exception as e:
        st.warning(f"Could not process news sentiment for {ticker_symbol}: {e}")
        return 0.0

# --- 4. Modeling Module ---

def create_synthetic_training_data(size=200):
    np.random.seed(42)
    data = {'current_ratio': np.random.uniform(0.5, 3.5, size), 'debt_to_equity': np.random.uniform(0.1, 5.0, size), 'profit_margin': np.random.uniform(-0.2, 0.5, size), 'sentiment_score': np.random.uniform(-0.5, 0.5, size)}
    df = pd.DataFrame(data)
    sustainability_score = (2*df['current_ratio']-1*df['debt_to_equity']+3*df['profit_margin']+1.5*df['sentiment_score'])
    probability = 1 / (1 + np.exp(-sustainability_score))
    df['is_sustainable'] = (probability > 0.5).astype(int)
    return df.drop('is_sustainable', axis=1), df['is_sustainable']

def train_model(X_train, y_train, model_type='logistic'):
    if model_type == 'logistic': model = LogisticRegression(random_state=42)
    else: model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def predict_sustainability(model, features):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return prediction, probability

# --- 5. Recommendation Module ---

def generate_recommendations(features, prediction):
    recs = []
    if prediction == 0: recs.append("🚨 **High Risk Alert:** Model predicts a high risk of business collapse. Strategic intervention is critical.")
    else: recs.append("✅ **Positive Outlook:** Model predicts business sustainability. Focus on maintaining strengths.")
    
    if 'current_ratio' in features and not pd.isna(features['current_ratio']) and features['current_ratio'] < 1.0: recs.append("🔴 **Liquidity Concern:** Current Ratio is critically low. Action: Urgently improve cash flow and manage short-term liabilities.")
    if 'debt_to_equity' in features and not pd.isna(features['debt_to_equity']) and features['debt_to_equity'] > 2.0: recs.append("🔴 **Leverage Risk:** Debt-to-Equity is high. Action: Focus on deleveraging or raising equity.")
    if 'debt_to_equity' in features and not pd.isna(features['debt_to_equity']) and features['debt_to_equity'] > 10.0: recs.append("🔥 **Extreme Leverage / Negative Equity:** Debt levels are unsustainable. This is a severe sign of financial distress.")
    if 'profit_margin' in features and not pd.isna(features['profit_margin']) and features['profit_margin'] < 0.0: recs.append("🔴 **Profitability Crisis:** Company is unprofitable. Action: Conduct a full review of the business model and cost structure.")
    if 'sentiment_score' in features and features['sentiment_score'] < -0.1: recs.append("🔴 **Negative Sentiment:** Public market perception is negative. Action: Engage in proactive public relations.")
    
    if len(recs) == 1: recs.append("All key indicators appear healthy. Continue monitoring.")
    return recs

# --- 6. Streamlit Dashboard ---

def main():
    st.set_page_config(page_title="Business Sustainability DSS", layout="wide")
    st.title("📈 Business Sustainability Decision Support System (BSDSS)")
    
    sentiment_analyzer = load_sentiment_model()

    st.sidebar.header("Controls")
    ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, PTON, BBBYQ)", "TSLA").upper()
    model_choice = st.sidebar.selectbox("Choose Prediction Model", ['Logistic Regression', 'XGBoost'])
    
    if st.sidebar.button("Analyze"):
        if not ticker_input:
            st.warning("Please enter a stock ticker.")
        else:
            with st.spinner(f"Analyzing {ticker_input}..."):
                income_stmt, balance_sheet = get_financial_data(ticker_input)
                if income_stmt is None: return

                financial_ratios = calculate_financial_ratios(income_stmt, balance_sheet)
                sentiment_score = get_news_sentiment(ticker_input, sentiment_analyzer)
                
                features = {'current_ratio': financial_ratios.get('current_ratio', 0), 'debt_to_equity': financial_ratios.get('debt_to_equity', 0), 'profit_margin': financial_ratios.get('profit_margin', 0), 'sentiment_score': sentiment_score}
                for k, v in features.items():
                    if pd.isna(v): features[k] = 0
                
                features_df = pd.DataFrame([features])
                X, y = create_synthetic_training_data()
                model = train_model(X, y, 'logistic' if model_choice == 'Logistic Regression' else 'xgboost')
                prediction, probability = predict_sustainability(model, features_df)
                recommendations = generate_recommendations(features, prediction)
                
            st.header(f"Analysis for {ticker_input}")
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 1: st.success("**Prediction: Likely to Sustain**")
                else: st.error("**Prediction: Potential Collapse Risk**")
            with col2:
                st.metric("Sustainability Probability", f"{probability:.2%}")

            st.subheader("Key Sustainability Indicators")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Ratio", f"{features['current_ratio']:.2f}")
            c2.metric("Debt-to-Equity", f"{features['debt_to_equity']:.2f}")
            c3.metric("Profit Margin", f"{features['profit_margin']:.2%}")
            c4.metric("News Sentiment", f"{features['sentiment_score']:.2f}")
            
            st.subheader("Actionable Recommendations")
            for rec in recommendations: st.markdown(f"- {rec}")

if __name__ == "__main__":
    main()
