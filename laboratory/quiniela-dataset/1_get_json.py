import requests
import json
import os

# Set the base URL
url = "https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAQU&celebrados=true"
dir_name = 'json_raw_data'
os.makedirs(dir_name, exist_ok=True)


def get_json(year, max_year):
    start_date = f"{year}0101"
    end_date = f"{year}1231"

    params = {
        "fechaInicioInclusiva": start_date,
        "fechaFinInclusiva": end_date
    }

    print(f"Getting year {year}/{max_year}")
    response = requests.get(url, params=params)

    data = response.json()

    with open(f"{dir_name}/{year}.json", "w") as outfile:
        json.dump(data, outfile, indent=2)

    if year < max_year:
        get_json(year + 1, max_year)


get_json(2009, 2022)
