import os
import re
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("EPIAS_USERNAME")
password = os.getenv("EPIAS_PASSWORD")

login_url = "https://giris.epias.com.tr/cas/v1/tickets"
ptf_url = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/dam/data/mcp"
REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 3
ROLLING_REFRESH_HOURS = 48
FORWARD_LOOK_DAYS = 7
CSV_PATH = "data/ptf_dataset.csv"


def post_with_retries(url, **kwargs):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.post(url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"İstek hatası ({attempt}/{MAX_RETRIES}): {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 10)

    raise last_error

# 1) TGT AL
tgt_response = post_with_retries(
    login_url,
    headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain"
    },
    data={
        "username": username,
        "password": password
    }
)

print("TGT durum kodu:", tgt_response.status_code)

tgt_text = tgt_response.text.strip()

if tgt_text.startswith("TGT-"):
    tgt = tgt_text
else:
    tgt_match = re.search(r"/cas/v1/tickets/([^\" ]+)", tgt_text)

    if not tgt_match:
        print("TGT alınamadı")
        print(tgt_text[:1000])
        exit()

    tgt = tgt_match.group(1)

print("TGT alındı:", tgt[:20] + "...")

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "TGT": tgt
}


def get_start_date_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return datetime(2020, 1, 1)

    old_df = pd.read_csv(csv_path)

    if old_df.empty or "date" not in old_df.columns:
        return datetime(2020, 1, 1)

    parsed = pd.to_datetime(old_df["date"], errors="coerce")
    last_date = parsed.max()

    if pd.isna(last_date):
        return datetime(2020, 1, 1)

    print("Son kayıt:", last_date)

    return last_date.to_pydatetime().replace(tzinfo=None) - timedelta(
        hours=ROLLING_REFRESH_HOURS
    )


# 2) TARİH ARALIKLARINI PARÇA PARÇA ÇEK
old_df = pd.DataFrame()

if os.path.exists(CSV_PATH):
    old_df = pd.read_csv(CSV_PATH)
    print("Eski PTF satır sayısı:", len(old_df))

start_date = get_start_date_from_csv(CSV_PATH)
start_date = start_date.replace(minute=0, second=0, microsecond=0)
end_date = datetime.now() + timedelta(days=FORWARD_LOOK_DAYS)

all_items = []

current_start = start_date

while current_start < end_date:
    current_end = current_start + timedelta(days=365)

    if current_end > end_date:
        current_end = end_date

    payload = {
        "startDate": current_start.strftime("%Y-%m-%dT00:00:00+03:00"),
        "endDate": current_end.strftime("%Y-%m-%dT00:00:00+03:00")
    }

    print("Çekiliyor:", payload["startDate"], "→", payload["endDate"])

    response = post_with_retries(
        ptf_url,
        json=payload,
        headers=headers
    )

    print("Durum:", response.status_code)

    if response.status_code != 200:
        print(response.text[:1500])
        break

    data = response.json()
    items = data.get("items", [])

    all_items.extend(items)

    print("Gelen satır:", len(items), "| Toplam:", len(all_items))

    current_start = current_end + timedelta(days=1)

    time.sleep(1)

print("Yeni gelen satır:", len(all_items))

# 3) CSV'YE KAYDET
new_df = pd.DataFrame(all_items)
df = pd.concat([old_df, new_df], ignore_index=True)

if "date" in df.columns and "hour" in df.columns:
    df = df.drop_duplicates(subset=["date", "hour"], keep="last")
    df = df.sort_values(by=["date", "hour"])

if "datetime" in df.columns:
    df = df.drop(columns=["datetime"])

df.to_csv(CSV_PATH, index=False)

print("Bitti.")
print("Toplam satır:", len(df))
print("CSV kaydedildi:", CSV_PATH)
