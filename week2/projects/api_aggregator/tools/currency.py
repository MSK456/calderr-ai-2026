import re
import httpx

SUPPORTED = ["USD", "EUR", "GBP", "PKR", "AED", "SAR", "JPY", "CAD", "AUD", "INR", "TRY"]

def fetch_rates(base: str = "USD") -> dict:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"https://open.er-api.com/v6/latest/{base.upper()}")
            resp.raise_for_status()
            data = resp.json()
        if data.get("result") != "success":
            return {"error": "API returned failure"}
        return {
            "base": base.upper(),
            "rates": {k: v for k, v in data["rates"].items() if k in SUPPORTED},
            "last_updated": data.get("time_last_update_utc", "Unknown")
        }
    except Exception as e:
        return {"error": str(e)}

def convert_currency(amount: float, from_cur: str, to_cur: str) -> dict:
    data = fetch_rates(from_cur)
    if "error" in data:
        return data
    rate = data["rates"].get(to_cur.upper())
    if not rate:
        return {"error": f"{to_cur} not in supported currencies: {SUPPORTED}"}
    return {
        "from": f"{amount} {from_cur.upper()}",
        "to": f"{round(amount * rate, 2)} {to_cur.upper()}",
        "rate": rate,
        "inverted_rate": round(1 / rate, 6)
    }