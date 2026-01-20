from flask import Blueprint, jsonify, request
from pathlib import Path
import pandas as pd

from src.predict import predict

api = Blueprint("api", __name__)

LOG_PATH = Path("monitoring/logs/requests.parquet")


@api.get("/health")
def health():
    return jsonify({"status": "ok"})


@api.post("/predict")
def predict_route():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Body deve ser JSON"}), 400

    try:
        # --- executa predição ---
        result = predict(payload)

        # --- logging para drift monitoring (não pode quebrar a API) ---
        try:
            feats = payload.get("features")
            rows = feats if isinstance(feats, list) else [feats]
            df_log = pd.DataFrame(rows)

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

            if LOG_PATH.exists():
                df_old = pd.read_parquet(LOG_PATH)
                df_new = pd.concat([df_old, df_log], ignore_index=True)
            else:
                df_new = df_log

            df_new.to_parquet(LOG_PATH, index=False)
        except Exception:
            # não derruba a API por causa de logging
            pass

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400
