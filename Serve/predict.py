import json
import time

MAX_CHARS = 5000   # protect endpoint
MAX_BATCH = 32     # protect endpoint

def input_fn(serialized_input_data, content_type):
    # Accept JSON only (backend-friendly)
    if content_type == "application/json":
        body = serialized_input_data.decode("utf-8")
        payload = json.loads(body)
        return payload
    raise Exception(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Expected request JSON:
      {"text": "review..."} OR {"texts": ["a", "b", ...]}
    Returns JSON-like dict:
      {"predictions":[{"label":0/1,"prob":0.xx}], "latency_ms": N}
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Validate & normalize -----
    if isinstance(input_data, dict) and "text" in input_data:
        texts = [input_data["text"]]
    elif isinstance(input_data, dict) and "texts" in input_data:
        texts = input_data["texts"]
    else:
        raise ValueError("Invalid JSON. Use {'text': ...} or {'texts': [...]}")

    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("texts must be a non-empty list")

    if len(texts) > MAX_BATCH:
        raise ValueError(f"Batch too large. Max {MAX_BATCH}")

    clean = []
    for s in texts:
        if not isinstance(s, str):
            raise ValueError("Each text must be a string")
        s = s.strip()
        if len(s) == 0:
            raise ValueError("Empty text not allowed")
        if len(s) > MAX_CHARS:
            s = s[:MAX_CHARS]
        clean.append(s)

    # ----- Tokenize as a batch (faster) -----
    enc = model.tokenizer(
        clean,
        truncation=True,
        padding=True,                 # dynamic padding = faster than max_length always
        max_length=model.max_len,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # ----- Inference -----
    with torch.inference_mode():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()

    preds = [{"label": int(p >= 0.5), "prob": float(p)} for p in probs]
    latency_ms = int((time.time() - t0) * 1000)

    return {"predictions": preds, "latency_ms": latency_ms}

def output_fn(prediction_output, accept):
    # Always return JSON string
    return json.dumps(prediction_output)
