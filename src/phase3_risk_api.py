from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import time
from typing import Any, Dict, List
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException

# ... (imports) ...

HIGH_RISK_CUSTOMERS = [
    {
        "id": 100045,
        "name": "Alicia W.",
        "riskScore": 82,
        "stressFactor": "salary_delay",
        "reasons": ["Salary drift", "Late payment trend", "Rising DPD"],
        "trend": [
            {"month": "Sep", "stress": 38},
            {"month": "Oct", "stress": 42},
            {"month": "Nov", "stress": 50},
            {"month": "Dec", "stress": 58},
            {"month": "Jan", "stress": 64},
            {"month": "Feb", "stress": 72},
        ],
    },
    {
        "id": 100112,
        "name": "Jordan K.",
        "riskScore": 76,
        "stressFactor": "utilization_spike",
        "reasons": ["Utilization spike", "High balance variance", "Cash advance uptick"],
        "trend": [
            {"month": "Sep", "stress": 30},
            {"month": "Oct", "stress": 36},
            {"month": "Nov", "stress": 44},
            {"month": "Dec", "stress": 53},
            {"month": "Jan", "stress": 57},
            {"month": "Feb", "stress": 61},
        ],
    },
    {
        "id": 100231,
        "name": "Priya S.",
        "riskScore": 69,
        "stressFactor": "payment_irregularity",
        "reasons": ["Payment inconsistency", "Recent DPD", "Short-term volatility"],
        "trend": [
            {"month": "Sep", "stress": 22},
            {"month": "Oct", "stress": 24},
            {"month": "Nov", "stress": 31},
            {"month": "Dec", "stress": 40},
            {"month": "Jan", "stress": 46},
            {"month": "Feb", "stress": 52},
        ],
    },
]


class FileBasedCache:
    def __init__(self, filename="data/store.json"):
        self.filename = filename
        self.data = {}
        self._load()
        self._ensure_defaults()

    def _load(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = {}
        else:
             self.data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        self._save()

    def _ensure_defaults(self):
        # Pre-populate if empty or keys missing
        hr_key = os.getenv("HIGH_RISK_CUSTOMERS_KEY", "high_risk_customers")
        if hr_key not in self.data:
            self.data[hr_key] = json.dumps(HIGH_RISK_CUSTOMERS)
            
            # Pre-populate some dummy feature data for scoring demo
            for customer in HIGH_RISK_CUSTOMERS:
                cid = customer['id']
                feat_key = f"features:{cid}"
                if feat_key not in self.data:
                    self.data[feat_key] = json.dumps({
                        "AMT_INCOME_TOTAL": 150000,
                        "AMT_CREDIT": 500000,
                        "avg_days_past_due": 12.5,
                        "late_payment_trend": 0.05
                    })
            self._save()

class FileBasedSNS:
    def publish(self, TopicArn, Message, Subject=None):
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "sns_events.jsonl"
        
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "TopicArn": TopicArn,
            "Message": Message,
            "Subject": Subject
        }
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            
        return {"MessageId": "simulated-message-id"}


# ... (imports) ...

def log_audit_event(event_type: str, payload: Dict[str, Any]) -> None:
    audit_dir = Path("d:/SpaceBar/logs")
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "audit_trail.jsonl"
    record = {
        "event_type": event_type,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "payload": payload,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_model(model_path: Path) -> xgb.Booster:
    # Use Booster directly to match training phase
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def load_feature_list(path: Path) -> List[str]:
    features = pd.read_csv(path)["feature"].astype(str).tolist()
    if not features:
        raise ValueError("Feature list is empty")
    return features


def build_feature_frame(feature_list: List[str], payload: Dict[str, Any]) -> pd.DataFrame:
    row = {name: float(payload.get(name, 0)) for name in feature_list}
    return pd.DataFrame([row], columns=feature_list)


def get_top3_shap(explainer: shap.TreeExplainer, X: pd.DataFrame) -> List[Dict[str, float]]:
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Handle single or multi-row X. Here X is single row.
    # shap_values might be (rows, cols)
    if len(shap_values.shape) == 1:
        vals = shap_values
    else:
        vals = shap_values[0]

    feature_names = X.columns.to_list()
    top_idx = np.argsort(np.abs(vals))[-3:][::-1]
    return [
        {"feature": feature_names[i], "score": float(vals[i])}
        for i in top_idx
    ]


from contextlib import asynccontextmanager

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML components on startup
    ml_models["model"] = load_model(MODEL_PATH)
    ml_models["feature_list"] = load_feature_list(FEATURE_PATH)
    ml_models["explainer"] = shap.TreeExplainer(ml_models["model"])
    ml_models["redis"] = FileBasedCache()
    ml_models["sns"] = FileBasedSNS() 


    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(title="Pre-Delinquency Risk Scoring API", lifespan=lifespan)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/xgb_model.json"))
FEATURE_PATH = Path(os.getenv("FEATURE_PATH", "artifacts/feature_list.csv"))
# REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0") # Not used
FEATURE_LATENCY_MS = float(os.getenv("FEATURE_LATENCY_THRESHOLD_MS", "100"))
HIGH_RISK_CUSTOMERS_KEY = os.getenv("HIGH_RISK_CUSTOMERS_KEY", "high_risk_customers")

# Global loading removed. Features loaded in lifespan.

HIGH_RISK_CUSTOMERS = [
    {
        "id": 100045,
        "name": "Alicia W.",
        "riskScore": 82,
        "stressFactor": "salary_delay",
        "reasons": ["Salary drift", "Late payment trend", "Rising DPD"],
        "trend": [
            {"month": "Sep", "stress": 38},
            {"month": "Oct", "stress": 42},
            {"month": "Nov", "stress": 50},
            {"month": "Dec", "stress": 58},
            {"month": "Jan", "stress": 64},
            {"month": "Feb", "stress": 72},
        ],
    },
    {
        "id": 100112,
        "name": "Jordan K.",
        "riskScore": 76,
        "stressFactor": "utilization_spike",
        "reasons": ["Utilization spike", "High balance variance", "Cash advance uptick"],
        "trend": [
            {"month": "Sep", "stress": 30},
            {"month": "Oct", "stress": 36},
            {"month": "Nov", "stress": 44},
            {"month": "Dec", "stress": 53},
            {"month": "Jan", "stress": 57},
            {"month": "Feb", "stress": 61},
        ],
    },
    {
        "id": 100231,
        "name": "Priya S.",
        "riskScore": 69,
        "stressFactor": "payment_irregularity",
        "reasons": ["Payment inconsistency", "Recent DPD", "Short-term volatility"],
        "trend": [
            {"month": "Sep", "stress": 22},
            {"month": "Oct", "stress": 24},
            {"month": "Nov", "stress": 31},
            {"month": "Dec", "stress": 40},
            {"month": "Jan", "stress": 46},
            {"month": "Feb", "stress": 52},
        ],
    },
]


@app.get("/customers/high-risk")
def list_high_risk_customers() -> List[Dict[str, Any]]:
    print("DEBUG: Request received for /customers/high-risk")
    redis_client = ml_models["redis"]
    try:
        raw = redis_client.get(HIGH_RISK_CUSTOMERS_KEY)
        print(f"DEBUG: Redis Key {HIGH_RISK_CUSTOMERS_KEY} returned: {raw}")
    except Exception as e:
        print(f"DEBUG: Redis get failed: {e}")
        raw = None

    if raw is not None:
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                print(f"DEBUG: Returning {len(data)} customers from Redis")
                return data
        except json.JSONDecodeError:
            print("DEBUG: JSON decode error")
            pass
    
    print(f"DEBUG: Returning fallback list of {len(HIGH_RISK_CUSTOMERS)} customers")
    return HIGH_RISK_CUSTOMERS


@app.post("/score")
def score(payload: Dict[str, Any]) -> Dict[str, Any]:
    redis_client = ml_models["redis"]
    model = ml_models["model"]
    feature_list = ml_models["feature_list"]
    explainer = ml_models["explainer"]

    customer_id = payload.get("customer_id")
    if customer_id is None:
        raise HTTPException(status_code=400, detail="customer_id is required")

    feature_key = f"features:{customer_id}"
    start = time.perf_counter()
    raw = redis_client.get(feature_key)
    feature_latency_ms = (time.perf_counter() - start) * 1000

    if feature_latency_ms > FEATURE_LATENCY_MS:
        log_audit_event(
            "feature_latency_breach",
            {
                "customer_id": customer_id,
                "latency_ms": round(feature_latency_ms, 2),
                "threshold_ms": FEATURE_LATENCY_MS,
            },
        )
        raise HTTPException(
            status_code=503,
            detail="Feature retrieval exceeded latency SLA",
        )

    if raw is None:
        raise HTTPException(status_code=404, detail="features not found")

    features_payload = json.loads(raw)
    X = build_feature_frame(feature_list, features_payload)

    # Convert to DMatrix for Booster prediction
    dtest = xgb.DMatrix(X)
    # Binary classification with logistic objective returns probability of class 1 directly
    prob = model.predict(dtest)
    # If prob is a numpy array, extract float
    if isinstance(prob, np.ndarray):
        prob = float(prob[0])
    
    risk_score = round(prob * 100, 2)
    reasons = get_top3_shap(explainer, X)

    response = {
        "customer_id": customer_id,
        "risk_score": round(risk_score, 2),
        "reasons": reasons,
        "feature_latency_ms": round(feature_latency_ms, 2),
        "latency_sla_ok": feature_latency_ms <= FEATURE_LATENCY_MS,
    }

    log_audit_event("risk_score", response)
    return response


@app.post("/notify")
@app.post("/api/notify")
def notify(payload: Dict[str, Any]) -> Dict[str, Any]:
    topic_arn = os.getenv("SNS_TOPIC_ARN")
    publish_result = "skipped"

    if topic_arn:
        try:
            # Try to use Boto3 if available and configured
            import boto3
            client = boto3.client("sns")
            client.publish(
                TopicArn=topic_arn,
                Message=json.dumps(payload),
                Subject="Risk Intervention",
            )
            publish_result = "sent_aws"
        except Exception:
            # Fallback to FileBasedSNS
            try:
                sns_client = FileBasedSNS()
                sns_client.publish(
                    TopicArn=topic_arn,
                    Message=json.dumps(payload),
                    Subject="Risk Intervention (Simulated)",
                )
                publish_result = "sent_simulated"
            except Exception as e:
                print(f"SNS simulation failed: {e}")
                publish_result = "failed"

    log_audit_event("intervention_notify", {"payload": payload, "sns": publish_result})
    return {"status": publish_result}
