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

# 1) TGT AL
tgt_response = requests.post(
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

# 2) TARİH ARALIKLARINI PARÇA PARÇA ÇEK
start_date = datetime(2020, 1, 1)
end_date = datetime.now()

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

    response = requests.post(
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

# 3) CSV'YE KAYDET
df = pd.DataFrame(all_items)

df = df.drop_duplicates(subset=["date", "hour"])
df = df.sort_values(by=["date", "hour"])

csv_path = "data/ptf_dataset.csv"
df.to_csv(csv_path, index=False)

print("Bitti.")
print("Toplam satır:", len(df))
print("CSV kaydedildi:", csv_path)