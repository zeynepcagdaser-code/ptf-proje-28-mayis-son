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
REAL_CONSUMPTION_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"

CSV_PATH = "data/real_consumption.csv"
REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 3
MAX_DAYS_PER_REQUEST = 90
ROLLING_REFRESH_HOURS = 48
FORWARD_LOOK_DAYS = 7


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


def get_tgt():
    response = post_with_retries(
        LOGIN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "text/plain"
        },
        data={
            "username": USERNAME,
            "password": PASSWORD
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


def parse_datetime_columns(df):
    if df.empty or "date" not in df.columns:
        return pd.Series(dtype="datetime64[ns]")

    parsed = pd.to_datetime(df["date"], errors="coerce")

    time_column = None

    if "time" in df.columns:
        time_column = "time"
    elif "hour" in df.columns:
        time_column = "hour"

    if parsed.isna().all() and time_column:
        parsed = pd.to_datetime(
            df["date"].astype(str) + " " + df[time_column].astype(str),
            errors="coerce"
        )

    return parsed


def get_start_date_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return datetime(2020, 1, 1)

    old_df = pd.read_csv(csv_path)

    if old_df.empty:
        return datetime(2020, 1, 1)

    parsed = parse_datetime_columns(old_df)
    last_date = parsed.max()

    if pd.isna(last_date):
        print("CSV içinden son kayıt tarihi okunamadı. Baştan çekilecek.")
        return datetime(2020, 1, 1)

    print("Son kayıt:", last_date)

    return last_date.to_pydatetime().replace(tzinfo=None) - timedelta(
        hours=ROLLING_REFRESH_HOURS
    )


def fetch_real_consumption_data(start_date, end_date, tgt):
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "TGT": tgt
    }

    all_items = []
    current_start = start_date
    safe_end_date = end_date

    while current_start <= safe_end_date:
        current_end = current_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1)

        if current_end > safe_end_date:
            current_end = safe_end_date

        payload = {
            "startDate": current_start.strftime("%Y-%m-%dT00:00:00+03:00"),
            "endDate": current_end.strftime("%Y-%m-%dT00:00:00+03:00")
        }

        print("Gerçekleşen tüketim çekiliyor:", payload["startDate"], "→", payload["endDate"])

        response = post_with_retries(
            REAL_CONSUMPTION_URL,
            json=payload,
            headers=headers
        )

        print("Durum:", response.status_code)

        if response.status_code != 200:
            print(response.text[:1500])

            if "geçmiş zaman olmalıdır" in response.text and current_end > datetime.now():
                safe_end_date = datetime.now().replace(
                    minute=0,
                    second=0,
                    microsecond=0
                )
                print("Gerçekleşen tüketim endpoint'i ileri endDate kabul etmedi. Güvenli bitişe dönülüyor:", safe_end_date)

                if current_start <= safe_end_date:
                    continue

            break

        data = response.json()
        items = data.get("items", [])

        all_items.extend(items)

        print("Gelen satır:", len(items), "| Toplam:", len(all_items))

        current_start = current_end + timedelta(days=1)
        time.sleep(1)

    return all_items


tgt = get_tgt()
print("TGT alındı:", tgt[:20] + "...")

old_df = pd.DataFrame()

if os.path.exists(CSV_PATH):
    old_df = pd.read_csv(CSV_PATH)
    print("Eski gerçekleşen tüketim satır sayısı:", len(old_df))

start_date = get_start_date_from_csv(CSV_PATH)
start_date = start_date.replace(minute=0, second=0, microsecond=0)

end_date = datetime.now() + timedelta(days=FORWARD_LOOK_DAYS)
end_date = end_date.replace(minute=0, second=0, microsecond=0)

print("Başlangıç:", start_date)
print("Bitiş:", end_date)

if start_date > end_date:
    print("Yeni çekilecek gerçekleşen tüketim verisi yok.")
    new_df = pd.DataFrame()
else:
    new_items = fetch_real_consumption_data(start_date, end_date, tgt)
    print("Yeni gelen satır:", len(new_items))
    new_df = pd.DataFrame(new_items)

final_df = pd.concat([old_df, new_df], ignore_index=True)

if "date" in final_df.columns and "time" in final_df.columns:
    final_df = final_df.drop_duplicates(subset=["date", "time"], keep="last")
    final_df = final_df.sort_values(by=["date", "time"])
elif "date" in final_df.columns and "hour" in final_df.columns:
    final_df = final_df.drop_duplicates(subset=["date", "hour"], keep="last")
    final_df = final_df.sort_values(by=["date", "hour"])

if "datetime" in final_df.columns:
    final_df = final_df.drop(columns=["datetime"])

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
final_df.to_csv(CSV_PATH, index=False)

print("Bitti.")
print("Final satır:", len(final_df))
print("CSV kaydedildi:", CSV_PATH)
