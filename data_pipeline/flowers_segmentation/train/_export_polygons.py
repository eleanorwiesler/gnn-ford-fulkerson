#!/usr/bin/env python3
import json
import csv
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Export image filename + segmentation polygon to CSV"
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="path to COCO JSON file (with 'images' and 'annotations')"
    )
    p.add_argument(
        "-o", "--output", required=True,
        help="path to output CSV file"
    )
    return p.parse_args()

def main():
    args = parse_args()
    # 1) load JSON
    with open(args.input, "r") as f:
        coco = json.load(f)

    # 2) build image_id -> file_name lookup
    id2name = {img["id"]: img["file_name"] 
               for img in coco.get("images", [])}

    # 3) open CSV and write header
    with open(args.output, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["file_name", "segmentation"])

        # 4) for each annotation, grab its polygon
        for ann in coco.get("annotations", []):
            segs = ann.get("segmentation", [])
            if not segs:
                continue
            # take the first polygon (COCO may have multiple)
            poly = segs[0]
            fname = id2name.get(ann["image_id"], "")
            # write file_name and the JSON‐dumped list of coords
            writer.writerow([fname, json.dumps(poly)])

    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
