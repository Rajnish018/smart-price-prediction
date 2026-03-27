from fastapi import FastAPI
from pydantic import BaseModel
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

app = FastAPI(title="Smart Pricing ML API")

# =========================
# LOAD MODEL + ENCODERS
# =========================
model = joblib.load("multi_model.pkl")
commodity_encoder = joblib.load("commodity_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

# =========================
# CEDA CONFIG
# =========================
BASE_URL = "https://agmarknet.ceda.ashoka.edu.in"
API_KEY = "d633d5593c43418605b1d16b5141397bd9e44618bf7b2947e6b704c37fb5a830"

headers = {
    "Content-Type": "application/json",
    "Origin": BASE_URL,
    "Referer": f"{BASE_URL}/",
    "x-api-key": API_KEY,
    "User-Agent": "Mozilla/5.0"
}

# =========================
# REQUEST MODELS
# =========================
class PredictInput(BaseModel):
    commodity: str
    state: str
    arrival: float
    prev_price: float


class LivePredictInput(BaseModel):
    commodity: str
    state: str


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# MANUAL PREDICTION
# =========================
@app.post("/predict")
def predict(data: PredictInput):
    try:
        commodity_encoded = commodity_encoder.transform([data.commodity])[0]
        state_encoded = state_encoder.transform([data.state])[0]

        demand_index = 1 / (data.arrival + 1)

        features = pd.DataFrame([{
            "commodity_encoded": commodity_encoded,
            "state_encoded": state_encoded,
            "arrival": data.arrival,
            "demand_index": demand_index,
            "prev_price": data.prev_price
        }])

        price = model.predict(features)[0]

        return {
            "commodity": data.commodity,
            "state": data.state,
            "recommended_price": round(price, 2)
        }

    except Exception as e:
        return {"error": str(e)}


# =========================
# FETCH CEDA DATA
# =========================
def fetch_ceda_data(commodity_name, state_name):
    # encode
    commodity_id_map = get_commodity_map()
    state_id_map = get_state_map()

    commodity_id = commodity_id_map.get(commodity_name)
    state_id = state_id_map.get(state_name)

    if not commodity_id or not state_id:
        raise Exception("Invalid commodity/state")

    date_ranges = [30, 90, 180, 365]

    for days in date_ranges:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)

        payload = {
            "commodity_id": commodity_id,
            "state_id": state_id,
            "district_id": 0,
            "calculation_type": "d",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d")
        }

        price_res = requests.post(f"{BASE_URL}/api/prices", json=payload, headers=headers)
        qty_res = requests.post(f"{BASE_URL}/api/quantities", json=payload, headers=headers)

        price_data = price_res.json().get("data", [])

        if price_data:
            break
    else:
        return None

    qty_data = qty_res.json().get("data", [])

    df_price = pd.DataFrame(price_data)
    df_qty = pd.DataFrame(qty_data)

    df = df_price.copy()

    df = df.rename(columns={
        "t": "date",
        "cmdty": "commodity",
        "p_modal": "price"
    })

    df["arrival"] = df_qty["qty"] if "qty" in df_qty else 0

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["arrival"] = pd.to_numeric(df["arrival"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna()
    df = df.sort_values(by="date")

    df["prev_price"] = df["price"].shift(1)
    df["demand_index"] = 1 / (df["arrival"] + 1)

    df = df.dropna()

    return df


# =========================
# GET MAPS
# =========================
def get_commodity_map():
    res = requests.get(f"{BASE_URL}/api/commodities", headers=headers)
    data = res.json().get("data", [])
    return {item["commodity_disp_name"]: item["commodity_id"] for item in data}


def get_state_map():
    res = requests.get(f"{BASE_URL}/api/states", headers=headers)
    data = res.json().get("data", [])
    return {item["census_state_name"]: item["census_state_id"] for item in data}


# =========================
# LIVE PREDICTION
# =========================
@app.post("/predict-live")
def predict_live(data: LivePredictInput):
    try:
        df = fetch_ceda_data(data.commodity, data.state)

        if df is None or len(df) == 0:
            return {"error": "No data available"}

        latest = df.iloc[-1]

        commodity_encoded = commodity_encoder.transform([data.commodity])[0]
        state_encoded = state_encoder.transform([data.state])[0]

        features = pd.DataFrame([{
            "commodity_encoded": commodity_encoded,
            "state_encoded": state_encoded,
            "arrival": latest["arrival"],
            "demand_index": latest["demand_index"],
            "prev_price": latest["prev_price"]
        }])

        predicted_price = model.predict(features)[0]

        return {
            "commodity": data.commodity,
            "state": data.state,
            "current_price": round(latest["price"], 2),
            "predicted_price": round(predicted_price, 2),
            "arrival": round(latest["arrival"], 2)
        }

    except Exception as e:
        return {"error": str(e)}