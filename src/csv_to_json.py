import argparse
import csv
import json
import os

def csv_to_json(csv_file, json_file):
    """
    Converts a CSV file with 'instruction', 'input', and 'output' columns into a JSON format.
    """
    data = []
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('instruction') and row.get('output'):
                data.append({
                    "instruction": row.get('instruction', "").strip(),
                    "input": row.get('input', "").strip(),
                    "output": row.get('output', "").strip()
                })

    with open(json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file to JSON format.")
    parser.add_argument(
        '--csv', required=True, help="Path to the input CSV file."
    )
    parser.add_argument(
        '--json', required=True, help="Path to the output JSON file."
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' does not exist.")
    else:
        csv_to_json(args.csv, args.json)
