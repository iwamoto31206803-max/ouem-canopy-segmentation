"""Read-only, bounded HF dataset schema probe. No bulk download."""
import argparse
import json
from urllib.request import Request, urlopen

API = "https://datasets-server.huggingface.co"

def get_json(url):
    with urlopen(Request(url, headers={"User-Agent":"ouem-canopy-schema-probe/0.1"}), timeout=30) as response:
        return json.load(response)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="restor/tcd")
    args = p.parse_args()
    if args.dataset != "restor/tcd":
        p.error("P1-1 permits only restor/tcd")
    from urllib.parse import urlencode
    splits = get_json(API+"/splits?"+urlencode({"dataset":args.dataset}))
    print(json.dumps({"splits":splits.get("splits", [])}, indent=2))
    rows = get_json(API+"/rows?"+urlencode({"dataset":args.dataset,"config":"default","split":"train","offset":0,"length":1}))
    for item in rows.get("rows", []):
        row = item["row"]
        print(json.dumps({"fields":list(row),"image_id":row.get("image_id"),"oam_id":row.get("oam_id"),"validation_fold":row.get("validation_fold"),"license":row.get("license"),"annotation_type":type(row.get("annotation")).__name__,"segments_type":type(row.get("segments")).__name__},indent=2))
    print("Schema only: pixel decoding and source integrity NOT VERIFIED")

if __name__ == "__main__":
    main()
