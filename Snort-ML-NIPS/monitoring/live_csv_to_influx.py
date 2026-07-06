import time
import os
import csv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

CSV_FILES = [
    ("/home/nadyahra/ml_ids_log_aemlp.csv", "ml_ids", "aemlp"),
    ("/home/nadyahra/ml_ips_log_aemlp.csv", "ml_ips", "aemlp"),

    ("/home/nadyahra/ml_ids_log_mlp.csv", "ml_ids", "mlp"),
    ("/home/nadyahra/ml_ips_log_mlp.csv", "ml_ips", "mlp"),

    ("/home/nadyahra/ml_ids_log_lgbm.csv", "ml_ids", "lgbm"),
    ("/home/nadyahra/ml_ips_log_lgbm.csv", "ml_ips", "lgbm"),

    ("/home/nadyahra/ml_ids_log_aemlp_new.csv", "ml_ids", "aemlp_new"),
    ("/home/nadyahra/ml_ips_log_aemlp_new.csv", "ml_ips", "aemlp_new"),

    ("/home/nadyahra/ml_ids_log_mlp_new.csv", "ml_ids", "mlp_new"),
    ("/home/nadyahra/ml_ips_log_mlp_new.csv", "ml_ips", "mlp_new"),

    ("/home/nadyahra/ml_ids_log_lgbm_new.csv", "ml_ids", "lgbm_new"),
    ("/home/nadyahra/ml_ips_log_lgbm_new.csv", "ml_ips", "lgbm_new"),
]

INFLUX_URL = "http://localhost:8086"
TOKEN = "ISI_TOKEN_INFLUXDB_DI_SINI"
ORG = "mlnips"
BUCKET = "snort_ml"

client = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

positions = {}
last_sizes = {}

for path, measurement, model in CSV_FILES:
    positions[path] = 0
    last_sizes[path] = 0


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except:
        return None


def safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except:
        return None


def read_new_rows(path):
    rows = []

    try:
        if not os.path.exists(path):
            positions[path] = 0
            last_sizes[path] = 0
            return rows

        current_size = os.path.getsize(path)

        # Kalau file dihapus lalu dibuat ulang, atau dikosongkan,
        # posisi baca harus reset ke awal.
        if current_size < positions[path]:
            positions[path] = 0

        # Kalau file kosong, reset posisi.
        if current_size == 0:
            positions[path] = 0
            last_sizes[path] = 0
            return rows

        with open(path, "r", newline="") as f:
            header_line = f.readline()

            if not header_line:
                positions[path] = 0
                return rows

            header = next(csv.reader([header_line.strip()]))

            if positions[path] == 0:
                # skip header
                positions[path] = f.tell()
            else:
                f.seek(positions[path])

            reader = csv.reader(f)

            for values in reader:
                if len(values) != len(header):
                    continue

                row = dict(zip(header, values))
                rows.append(row)

            positions[path] = f.tell()
            last_sizes[path] = current_size

    except FileNotFoundError:
        positions[path] = 0
        last_sizes[path] = 0

    except Exception as e:
        print("Read error:", path, e)

    return rows


def add_tag_if_exists(point, row, key):
    if key in row and row[key] != "":
        point = point.tag(key, str(row[key]))
    return point


def add_float_field_if_exists(point, row, key):
    if key in row and row[key] != "":
        value = safe_float(row[key])
        if value is not None:
            point = point.field(key, value)
    return point


def add_int_field_if_exists(point, row, key):
    if key in row and row[key] != "":
        value = safe_int(row[key])
        if value is not None:
            point = point.field(key, value)
    return point


print("CSV to InfluxDB started...")
print("Bucket:", BUCKET)
print("Org:", ORG)

while True:
    for path, measurement, model in CSV_FILES:
        new_rows = read_new_rows(path)

        for row in new_rows:
            try:
                point = Point(measurement)
                point = point.tag("model", model)

                # Tags
                point = add_tag_if_exists(point, row, "client_ip")
                point = add_tag_if_exists(point, row, "src_ip")
                point = add_tag_if_exists(point, row, "dst_ip")
                point = add_tag_if_exists(point, row, "stage")
                point = add_tag_if_exists(point, row, "action")
                point = add_tag_if_exists(point, row, "decision")

                # Float fields
                point = add_float_field_if_exists(point, row, "score")
                point = add_float_field_if_exists(point, row, "max_score")
                point = add_float_field_if_exists(point, row, "threshold")
                point = add_float_field_if_exists(point, row, "block_threshold")
                point = add_float_field_if_exists(point, row, "suspicious_threshold")
                point = add_float_field_if_exists(point, row, "duration_us")

                # Integer fields
                point = add_int_field_if_exists(point, row, "label")
                point = add_int_field_if_exists(point, row, "blocked")
                point = add_int_field_if_exists(point, row, "block_decision")
                point = add_int_field_if_exists(point, row, "ipset_applied")
                point = add_int_field_if_exists(point, row, "pkt_count")
                point = add_int_field_if_exists(point, row, "high_score_hit")
                point = add_int_field_if_exists(point, row, "suspicious_hit")

                write_api.write(bucket=BUCKET, org=ORG, record=point)

                print("Written:", measurement, model, row)

            except Exception as e:
                print("Write error:", path, e)
                print("Row:", row)

    time.sleep(1)
