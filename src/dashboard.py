import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Pre-Delinquency Dashboard", layout="wide")

# Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .high-risk {
        color: #d32f2f;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Pre-Delinquency Intervention Engine")
st.markdown("### Proactive Risk Monitoring Dashboard")


# Tabs for different views
tab1, tab2 = st.tabs(["Live Monitor", "System Logs"])

with tab1:
    # Metric Placeholders
    col1, col2, col3, col4 = st.columns(4)

    def fetch_high_risk_customers():
        try:
            response = requests.get(f"{API_URL}/customers/high-risk")
            if response.status_code == 200:
                return response.json()
            return []
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to Risk API. Is it running?")
            return []

    def fetch_score(customer_id):
        try:
            # Pass dummy data payload assuming API will use mocked features if feature key exists,
            # or simplified flow. The current mock in API ignores payload for features if key exists.
            payload = {"customer_id": customer_id}
            response = requests.post(f"{API_URL}/score", json=payload)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            return None

    # Load Data
    customers = fetch_high_risk_customers()

    if customers:
        df = pd.DataFrame(customers)
        
        # Top Level Metrics
        total_risk_volume = df['riskScore'].sum() if 'riskScore' in df.columns else 0
        avg_risk = df['riskScore'].mean() if 'riskScore' in df.columns else 0
        
        with col1:
            st.metric("High Risk Customers", len(customers))
        with col2:
            st.metric("Avg Risk Score", f"{avg_risk:.1f}")
        with col3:
            st.metric("Intervention Queue", len(customers)) # Placeholder
        with col4:
            st.metric("SLA Breaches", 0) # Placeholder

        # Main Layout
        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.subheader("⚠️ At-Risk Accounts")
            
            # Interactive Table
            selected_customer_id = None
            
            # Display as a clean simplified table
            if not df.empty:
                display_df = df[['id', 'name', 'riskScore', 'stressFactor']]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Simple selector
                selected_id_input = st.selectbox(
                    "Select Customer for Deep Dive:",
                    options=df['id'].tolist(),
                    format_func=lambda x: f"{x} - {df[df['id']==x]['name'].values[0]}"
                )
                selected_customer_id = selected_id_input

        with right_col:
            if selected_customer_id:
                customer_data = df[df['id'] == selected_customer_id].iloc[0]
                
                st.subheader(f"Risk Profile: {customer_data['name']}")
                
                # Gauge Chart for Risk Score
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = customer_data['riskScore'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Delinquency Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "salmon"}],
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Risk Factors Explainer
                st.markdown("#### 🔍 Primary Risk Drivers")
                # In a real scenario, we'd call /score to get fresh SHAP values. 
                # For this demo, we can use the 'reasons' from the list or call score.
                
                score_details = fetch_score(selected_customer_id)
                if score_details and 'reasons' in score_details:
                    reasons = score_details['reasons']
                    # Display SHAP reasons
                    if isinstance(reasons, list) and len(reasons) > 0 and isinstance(reasons[0], dict):
                        for r in reasons:
                            score_val = r.get('score', 0)
                            icon = "🔻" if score_val < 0 else "🔺"
                            st.write(f"**{r.get('feature')}**: {score_val:.4f} {icon}")
                    else:
                         st.write("No detailed SHAP reasons available.")

                elif 'reasons' in customer_data:
                     # Fallback to static list
                     for r in customer_data['reasons']:
                         st.write(f"• {r}")

                # Trend Analysis (Mock Data)
                if 'trend' in customer_data:
                    st.markdown("#### 📅 Stress Signal Trend")
                    trend_df = pd.DataFrame(customer_data['trend'])
                    fig_trend = px.line(trend_df, x='month', y='stress', 
                                        title='Financial Stress Index (Last 6 Months)',
                                        markers=True)
                    st.plotly_chart(fig_trend, use_container_width=True)

                # Action Buttons
                st.markdown("#### 🛡️ Recommended Intervention")
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("📞 Trigger Callback", key="call"):
                        # Call API /notify
                        try:
                            requests.post(f"{API_URL}/notify", json={"customer_id": selected_customer_id, "action": "call"})
                            st.success("Callback scheduled!")
                        except Exception as e:
                            st.error(f"Failed to trigger callback: {e}")
                with action_col2:
                    if st.button("📧 Send Payment Plan", key="email"):
                        try:
                            requests.post(f"{API_URL}/notify", json={"customer_id": selected_customer_id, "action": "email"})
                            st.success("Email sent!")
                        except Exception as e:
                            st.error(f"Failed to send email: {e}")

    else:
        st.info("No high risk customers found. System is healthy.")

with tab2:
    st.header("System Logs (Simulation)")
    
    col_logs1, col_logs2 = st.columns(2)
    
    with col_logs1:
        st.subheader("SNS Notifications")
        try:
            if os.path.exists("logs/sns_events.jsonl"):
                with open("logs/sns_events.jsonl", "r") as f:
                    sns_logs = [json.loads(line) for line in f]
                st.json(sns_logs[::-1]) # Show newest first
            else:
                st.info("No SNS logs found yet.")
        except Exception as e:
            st.error(f"Error reading SNS logs: {e}")

    with col_logs2:
        st.subheader("Kafka Events")
        try:
            if os.path.exists("logs/kafka_events.jsonl"):
                with open("logs/kafka_events.jsonl", "r") as f:
                    kafka_logs = [json.loads(line) for line in f]
                st.json(kafka_logs[::-1])
            else:
                st.info("No Kafka logs found yet.")
        except Exception as e:
            st.error(f"Error reading Kafka logs: {e}")
