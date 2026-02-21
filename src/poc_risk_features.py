import pandas as pd
import numpy as np

def load_and_create_features():
    """
    Demonstrates how to create a Behavioral Risk Feature (Days Past Due)
    using the provided dataset.
    """
    print("Loading datasets for Proof of Concept...")
    
    # 1. Load Repayment History (Behavioral Data)
    # We use nrows=10000 for speed in this demo
    try:
        installments = pd.read_csv('d:/SpaceBar/installments_payments.csv', nrows=10000)
        print(f"Successfully loaded {len(installments)} rows from installments_payments.csv")
    except FileNotFoundError:
        print("Error: installments_payments.csv not found.")
        return

    # 2. Feature Engineering: Calculate Days Past Due (DPD)
    # DPD = Days Entry Payment - Days Instalment
    # Positive value means late payment. Negative means early payment.
    installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    
    # 3. Aggregate by Customer (SK_ID_CURR)
    # We want to know the "Average Days Late" for each customer
    behavioral_features = installments.groupby('SK_ID_CURR').agg({
        'DPD': ['mean', 'max', 'count']
    })
    
    # Flatten columns
    behavioral_features.columns = ['AVG_DPD', 'MAX_DPD', 'COUNT_INSTALLMENTS']
    behavioral_features.reset_index(inplace=True)
    
    print("\n--- Behavioral Features Created ---")
    print(behavioral_features.head())
    
    # 4. Identification: Who is "Pre-Delinquent"?
    # Rule: If Avg DPD > 5 days, flag as "At Risk"
    behavioral_features['IS_AT_RISK'] = behavioral_features['AVG_DPD'] > 5
    
    at_risk_count = behavioral_features['IS_AT_RISK'].sum()
    print(f"\nidentified {at_risk_count} customers showing early signs of stress (Avg DPD > 5).")
    
    print("\nProof of Concept Complete: We can detect early stress signals from existing data.")

if __name__ == "__main__":
    load_and_create_features()
