import os
import re
import time
import requests
import pandas as pd

from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("EPIAS_USERNAME")
PASSWORD = os.getenv("EPIAS_PASSWORD")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"

REALTIME_URL = (
    "https://seffaflik.epias.com.tr/"
    "electricity-service/v1/generation/data/realtime-generation"
)

CSV_PATH = "data/realtime_generation.csv"
MAX_DAYS_PER_REQUEST = 90


def get_tgt():

    for attempt in range(3):

        try:

            response = requests.post(
                LOGIN_URL,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/plain"
                },
                data={
                    "username": USERNAME,
                    "password": PASSWORD
                },
                timeout=(20, 60)
            )

            print("TGT durum kodu:", response.status_code)

            tgt_text = response.text.strip()

            if tgt_text.startswith("TGT-"):
                return tgt_text

            tgt_match = re.search(
                r"/cas/v1/tickets/([^\" ]+)",
                tgt_text
            )

            if tgt_match:
                return tgt_match.group(1)

        except Exception as e:

            print(f"TGT deneme {attempt+1}/3 hata:", e)

            time.sleep(5)

    raise Exception("TGT alınamadı")


def fetch_epias_data(start_date, end_date, tgt):

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    all_items = []

    current_start = start_date

    while current_start <= end_date:

        current_end = current_start + timedelta(
            days=MAX_DAYS_PER_REQUEST - 1
        )

        if current_end > end_date:
            current_end = end_date

        payload = {
            "startDate": current_start.strftime(
                "%Y-%m-%dT00:00:00+03:00"
            ),
            "endDate": current_end.strftime(
                "%Y-%m-%dT00:00:00+03:00"
            ),
            "region": "TR1"
        }

        print(
            "Gerçek zamanlı üretim çekiliyor:",
            payload["startDate"],
            "→",
            payload["endDate"]
        )

        success = False

        for attempt in range(3):

            try:

                response = requests.post(
                    REALTIME_URL,
                    json=payload,
                    headers=headers,
                    timeout=(20, 180)
                )

                print("Durum:", response.status_code)

                if response.status_code != 200:

                    print(response.text[:1000])

                    time.sleep(5)

                    continue

                data = response.json()

                items = data.get("items", [])

                all_items.extend(items)

                print(
                    "Gelen satır:",
                    len(items),
                    "| Toplam:",
                    len(all_items)
                )

                success = True

                break

            except Exception as e:

                print(
                    f"İstek deneme {attempt+1}/3 hata:",
                    e
                )

                time.sleep(5)

        if not success:

            raise Exception(
                f"Veri çekilemedi: {payload}"
            )

        current_start = current_end + timedelta(days=1)

        time.sleep(1)

    return all_items


tgt = get_tgt()

print("TGT alındı:", tgt[:20] + "...")

# CSV varsa güncelle
if os.path.exists(CSV_PATH):

    old_df = pd.read_csv(CSV_PATH)

    print("Eski satır:", len(old_df))

    old_df["datetime"] = pd.to_datetime(
        old_df["date"],
        errors="coerce"
    )

    if old_df["datetime"].isna().all() and "hour" in old_df.columns:

        old_df["datetime"] = pd.to_datetime(
            old_df["date"].astype(str) + " " + old_df["hour"].astype(str),
            errors="coerce"
        )

    last_date = old_df["datetime"].max()

    if pd.isna(last_date):

        raise Exception("CSV içinden son kayıt tarihi okunamadı")

    print("Son kayıt:", last_date)

    start_date = last_date.to_pydatetime().replace(tzinfo=None)

    start_date = start_date + timedelta(hours=1)

    start_date = start_date.replace(
        minute=0,
        second=0,
        microsecond=0
    )

else:

    old_df = pd.DataFrame()

    start_date = datetime(2020, 1, 1)

# düne kadar çek
end_date = datetime.now() - timedelta(days=1)

print("Başlangıç:", start_date)
print("Bitiş:", end_date)

new_items = fetch_epias_data(
    start_date=start_date,
    end_date=end_date,
    tgt=tgt
)

print("Yeni gelen satır:", len(new_items))

new_df = pd.DataFrame(new_items)

final_df = pd.concat(
    [old_df, new_df],
    ignore_index=True
)

if "date" in final_df.columns and "hour" in final_df.columns:

    final_df = final_df.drop_duplicates(
        subset=["date", "hour"]
    )

    final_df = final_df.sort_values(
        by=["date", "hour"]
    )

if "datetime" in final_df.columns:

    final_df = final_df.drop(columns=["datetime"])

final_df.to_csv(CSV_PATH, index=False)

print("Bitti.")
print("Final satır:", len(final_df))
print("CSV kaydedildi:", CSV_PATH)
