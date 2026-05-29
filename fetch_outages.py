import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("EPIAS_USERNAME")
PASSWORD = os.getenv("EPIAS_PASSWORD")

LOGIN_URL = "https://giris.epias.com.tr/cas/v1/tickets"
OUTAGES_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/data/market-message-system"
REGION_LIST_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/data/umm-region-list"
MESSAGE_TYPE_LIST_URL = "https://seffaflik.epias.com.tr/electricity-service/v1/markets/data/umm-message-type-list"

CSV_PATH = "data/outages.csv"
REQUEST_TIMEOUT = (10, 120)
MAX_RETRIES = 3
MAX_DAYS_PER_REQUEST = 90
ROLLING_REFRESH_HOURS = 48
FORWARD_LOOK_DAYS = 7

COLUMN_ORDER = [
    "regionId",
    "regionShortName",
    "messageTypeId",
    "messageTypeName",
    "id",
    "orgName",
    "powerPlantName",
    "uevcbId",
    "uevcbName",
    "caseStartDate",
    "caseEndDate",
    "operatorPower",
    "capacityAtCaseTime",
    "reason",
    "faultDetailCount",
    "detailStartHour",
    "detailEndHour",
    "minRemainingCapacity",
    "maxPreFaultPower",
    "maxFaultCausedPowerLoss",
    "totalFaultCausedPowerLoss",
    "totalFaultCausedEnergyLoss",
]


def request_with_retries(method, url, **kwargs):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"İstek hatası ({attempt}/{MAX_RETRIES}): {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 10)

    raise last_error


def post_with_retries(url, **kwargs):
    return request_with_retries("POST", url, **kwargs)


def get_with_retries(url, **kwargs):
    return request_with_retries("GET", url, **kwargs)


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
        raise SystemExit(1)

    return tgt_match.group(1)


def parse_datetime_columns(df):
    if df.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")

    for column in ["caseStartDate", "caseEndDate", "date", "time"]:
        if column in df.columns:
            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)

            if not parsed.isna().all():
                return parsed

    return pd.Series(dtype="datetime64[ns, UTC]")


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


def get_region(headers):
    response = get_with_retries(REGION_LIST_URL, headers=headers)
    print("Bölge listesi durum:", response.status_code)

    if response.status_code != 200:
        print(response.text[:1500])
        raise SystemExit(1)

    items = response.json().get("items", [])

    if not items:
        raise SystemExit("Bölge listesi boş döndü.")

    region = items[0]
    print("Kullanılan bölge:", region)
    return region


def get_message_types(headers):
    response = get_with_retries(MESSAGE_TYPE_LIST_URL, headers=headers)
    print("Mesaj tipi listesi durum:", response.status_code)

    if response.status_code != 200:
        print(response.text[:1500])
        raise SystemExit(1)

    items = response.json().get("items", [])

    if not items:
        raise SystemExit("Mesaj tipi listesi boş döndü.")

    print("Mesaj tipleri:", items)
    return items


def build_outage_rows(items, region, message_type):
    rows = []

    for item in items:
        fault_details = item.get("faultDetails") or []
        detail_hours = [detail.get("hour") for detail in fault_details if detail.get("hour")]
        remaining_capacities = [
            detail.get("remainingCapacity")
            for detail in fault_details
            if detail.get("remainingCapacity") is not None
        ]
        pre_fault_powers = [
            detail.get("preFaultPower")
            for detail in fault_details
            if detail.get("preFaultPower") is not None
        ]
        power_losses = [
            detail.get("faultCausedPowerLoss")
            for detail in fault_details
            if detail.get("faultCausedPowerLoss") is not None
        ]
        energy_losses = [
            detail.get("faultCausedEnergyLoss")
            for detail in fault_details
            if detail.get("faultCausedEnergyLoss") is not None
        ]

        row = {
            "regionId": region.get("regionId"),
            "regionShortName": region.get("regionShortName"),
            "messageTypeId": message_type.get("id"),
            "messageTypeName": message_type.get("typeName"),
            "id": item.get("id"),
            "orgName": item.get("orgName"),
            "powerPlantName": item.get("powerPlantName"),
            "uevcbId": item.get("uevcbId"),
            "uevcbName": item.get("uevcbName"),
            "caseStartDate": item.get("caseStartDate"),
            "caseEndDate": item.get("caseEndDate"),
            "operatorPower": item.get("operatorPower"),
            "capacityAtCaseTime": item.get("capacityAtCaseTime"),
            "reason": item.get("reason"),
            "faultDetailCount": len(fault_details),
            "detailStartHour": min(detail_hours) if detail_hours else None,
            "detailEndHour": max(detail_hours) if detail_hours else None,
            "minRemainingCapacity": min(remaining_capacities) if remaining_capacities else None,
            "maxPreFaultPower": max(pre_fault_powers) if pre_fault_powers else None,
            "maxFaultCausedPowerLoss": max(power_losses) if power_losses else None,
            "totalFaultCausedPowerLoss": sum(power_losses) if power_losses else None,
            "totalFaultCausedEnergyLoss": sum(energy_losses) if energy_losses else None,
        }
        rows.append(row)

    return rows


def fetch_outage_rows(start_date, end_date, headers, region, message_types):
    all_rows = []
    current_start = start_date
    safe_end_date = end_date

    while current_start <= safe_end_date:
        current_end = current_start + timedelta(days=MAX_DAYS_PER_REQUEST - 1)

        if current_end > safe_end_date:
            current_end = safe_end_date

        for message_type in message_types:
            payload = {
                "startDate": current_start.strftime("%Y-%m-%dT00:00:00+03:00"),
                "endDate": current_end.strftime("%Y-%m-%dT00:00:00+03:00"),
                "regionId": region["regionId"],
                "mesajTipId": message_type["id"],
            }

            print(
                "Kesinti verisi çekiliyor:",
                payload["startDate"],
                "→",
                payload["endDate"],
                "|",
                message_type["typeName"]
            )

            response = post_with_retries(OUTAGES_URL, json=payload, headers=headers)
            print("Durum:", response.status_code)

            if response.status_code != 200:
                print(response.text[:1500])

                if "geçmiş zaman olmalıdır" in response.text and current_end > datetime.now():
                    safe_end_date = datetime.now().replace(
                        minute=0,
                        second=0,
                        microsecond=0
                    )
                    print("Kesinti endpoint'i ileri endDate kabul etmedi. Güvenli bitişe dönülüyor:", safe_end_date)

                    if current_start <= safe_end_date:
                        break

                return all_rows

            items = response.json().get("items", [])
            rows = build_outage_rows(items, region, message_type)
            all_rows.extend(rows)

            print(
                "Ana kayıt:",
                len(items),
                "| CSV satırı:",
                len(rows),
                "| Toplam:",
                len(all_rows)
            )

            time.sleep(1)

        current_start = current_end + timedelta(days=1)

    return all_rows


tgt = get_tgt()
print("TGT alındı:", tgt[:20] + "...")

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "TGT": tgt
}

region = get_region(headers)
message_types = get_message_types(headers)

old_df = pd.DataFrame(columns=COLUMN_ORDER)

if os.path.exists(CSV_PATH):
    old_df = pd.read_csv(CSV_PATH)
    print("Eski kesinti satır sayısı:", len(old_df))

start_date = get_start_date_from_csv(CSV_PATH)
start_date = start_date.replace(minute=0, second=0, microsecond=0)

end_date = datetime.now() + timedelta(days=FORWARD_LOOK_DAYS)
end_date = end_date.replace(minute=0, second=0, microsecond=0)

print("Başlangıç:", start_date)
print("Bitiş:", end_date)

if start_date > end_date:
    print("Yeni çekilecek kesinti verisi yok.")
    new_df = pd.DataFrame(columns=COLUMN_ORDER)
else:
    new_rows = fetch_outage_rows(start_date, end_date, headers, region, message_types)
    print("Yeni gelen satır:", len(new_rows))
    new_df = pd.DataFrame(new_rows)

    for column in COLUMN_ORDER:
        if column not in new_df.columns:
            new_df[column] = pd.NA

    new_df = new_df[COLUMN_ORDER]

final_df = pd.concat([old_df, new_df], ignore_index=True)

for column in COLUMN_ORDER:
    if column not in final_df.columns:
        final_df[column] = pd.NA

if "id" in final_df.columns and "messageTypeId" in final_df.columns:
    final_df = final_df.drop_duplicates(subset=["messageTypeId", "id"], keep="last")
    final_df = final_df.sort_values(by=["caseStartDate", "messageTypeId", "id"], na_position="last")
elif "id" in final_df.columns:
    final_df = final_df.drop_duplicates(subset=["id"], keep="last")
    final_df = final_df.sort_values(by=["caseStartDate", "id"], na_position="last")

if "datetime" in final_df.columns:
    final_df = final_df.drop(columns=["datetime"])

final_df = final_df[COLUMN_ORDER]

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
final_df.to_csv(CSV_PATH, index=False)

print("Bitti.")
print("Final satır:", len(final_df))
print("CSV kaydedildi:", CSV_PATH)
