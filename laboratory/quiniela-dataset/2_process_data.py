import json
import csv
import os

origin_dir_name = 'json_raw_data'
dir_name = 'datasets'
os.makedirs(dir_name, exist_ok=True)


def extract_data(start_year, end_year):
    print(f"Processing {start_year}/{end_year}")
    with open(f"{origin_dir_name}/{start_year}.json", "r") as infile:
        data = json.load(infile)

    with open(f"{dir_name}/dataset.csv", "a", newline="") as outfile:
        writer = csv.writer(outfile)

        for d in data:
            for match in d["partidos"]:
                local = match["local"]
                visitante = match["visitante"]
                signo = match["signo"].strip()
                fecha_completa = match["fecha_completa"]

                writer.writerow([local, visitante, signo, fecha_completa])

    if start_year < end_year:
        extract_data(start_year + 1, end_year)


extract_data(2009, 2022)
