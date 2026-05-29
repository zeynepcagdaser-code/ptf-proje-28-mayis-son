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

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"

FDPP_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/dpp"
FDPP_FIRST_VERSION_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/generation/data/dpp-first-version"

CSV_PATH = "data/kgup_combined.csv"


def get_tgt():
    response = requests.post(
        LOGIN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/plain"
        },
        data={
            "username": username,
            "password": password
        }
    )

    print("TGT durum kodu:", response.status_code)

    tgt_text = response.text.strip()

    if tgt_text.startswith("TGT-"):
        return tgt_text

    tgt_match = re.search(r"/cas/v1/tickets/([^\" ]+)", tgt_text)

    if not tgt_match:
        print("TGT alınamadı")
        print(tgt_text[:1000])
        exit()

    return tgt_match.group(1)


def fetch_epias_data(url, start_date, end_date, source_type, tgt, max_days):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    all_items = []
    current_start = start_date

    while current_start <= end_date:
        current_end = current_start + timedelta(days=max_days - 1)

        if current_end > end_date:
            current_end = end_date

        payload = {
            "startDate": current_start.strftime("%Y-%m-%dT00:00:00+03:00"),
            "endDate": current_end.strftime("%Y-%m-%dT00:00:00+03:00"),
            "region": "TR1"
        }

        print(f"{source_type} çekiliyor:", payload["startDate"], "→", payload["endDate"])

        response = requests.post(url, json=payload, headers=headers)

        print("Durum:", response.status_code)

        if response.status_code != 200:
            print(response.text[:1500])
            break

        data = response.json()
        items = data.get("items", [])

        for item in items:
            item["source_type"] = source_type

        all_items.extend(items)

        print("Gelen satır:", len(items), "| Toplam:", len(all_items))

        current_start = current_end + timedelta(days=1)
        time.sleep(1)

    return all_items


def get_last_date_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return None

    old_df = pd.read_csv(csv_path)

    if old_df.empty or "date" not in old_df.columns:
        return None

    old_df["date_dt"] = pd.to_datetime(old_df["date"], errors="coerce")
    last_date = old_df["date_dt"].max()

    if pd.isna(last_date):
        return None

    return last_date.to_pydatetime().replace(tzinfo=None)


tgt = get_tgt()
print("TGT alındı:", tgt[:20] + "...")

today = datetime.now().replace(tzinfo=None)

old_df = pd.DataFrame()

if os.path.exists(CSV_PATH):
    old_df = pd.read_csv(CSV_PATH)
    print("Eski KGÜP satır sayısı:", len(old_df))

last_date = get_last_date_from_csv(CSV_PATH)

if last_date is None:
    print("Mevcut CSV yok veya boş. İlk kurulum yapılıyor.")

    fdpp_items = fetch_epias_data(
        url=FDPP_URL,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2022, 12, 31),
        source_type="fdpp",
        tgt=tgt,
        max_days=90
    )

    fdpp_first_items = fetch_epias_data(
        url=FDPP_FIRST_VERSION_URL,
        start_date=datetime(2023, 1, 1),
        end_date=today,
        source_type="fdpp_first_version",
        tgt=tgt,
        max_days=365
    )

    new_df = pd.DataFrame(fdpp_items + fdpp_first_items)

else:
    print("Son kayıt tarihi:", last_date)

    start_update = last_date.replace(hour=0, minute=0, second=0, microsecond=0)

    print("Güncelleme başlangıcı:", start_update)

    new_items = fetch_epias_data(
        url=FDPP_FIRST_VERSION_URL,
        start_date=start_update,
        end_date=today,
        source_type="fdpp_first_version",
        tgt=tgt,
        max_days=365
    )

    new_df = pd.DataFrame(new_items)

print("Yeni gelen satır:", len(new_df))

final_df = pd.concat([old_df, new_df], ignore_index=True)

if "date" in final_df.columns and "time" in final_df.columns:
    final_df = final_df.drop_duplicates(subset=["date", "time"], keep="last")
    final_df = final_df.sort_values(by=["date", "time"])

if "date_dt" in final_df.columns:
    final_df = final_df.drop(columns=["date_dt"])

final_df.to_csv(CSV_PATH, index=False)

print("Bitti.")
print("Final satır:", len(final_df))
print("CSV kaydedildi:", CSV_PATH)