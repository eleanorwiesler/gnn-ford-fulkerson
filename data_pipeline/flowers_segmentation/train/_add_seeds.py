#!/usr/bin/env python3
import pandas as pd
import json
import random
import argparse

def point_in_poly(x, y, poly):
    """
    Ray-casting point-in-polygon.
    `poly` is a list of (x,y) tuples.
    """
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def gen_seeds(poly_pts, n_inside=6, n_outside=6, img_w=640, img_h=640):
    # bounding box for faster inside sampling
    xs = [p[0] for p in poly_pts]
    ys = [p[1] for p in poly_pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    fg, bg = [], []
    # inside seeds
    while len(fg) < n_inside:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if point_in_poly(x, y, poly_pts):
            fg.append([int(x), int(y)])
    # outside seeds
    while len(bg) < n_outside:
        x = random.uniform(0, img_w)
        y = random.uniform(0, img_h)
        if not point_in_poly(x, y, poly_pts):
            bg.append([int(x), int(y)])
    return fg, bg

def main():
    p = argparse.ArgumentParser(
        description="Filter duplicate filenames and add fg/bg seeds"
    )
    p.add_argument("--input",  default="_polygons.csv",
                   help="CSV with file_name & segmentation")
    p.add_argument("--output", default="_polygons_with_seeds.csv",
                   help="where to write the augmented CSV")
    p.add_argument("--n_inside",  type=int, default=6)
    p.add_argument("--n_outside", type=int, default=6)
    args = p.parse_args()

    # 1) load
    df = pd.read_csv(args.input, converters={
        "segmentation": lambda s: json.loads(s)
    })

    # 2) drop all rows for any filename that appears more than once
    df = df.groupby('file_name').filter(lambda x: len(x) == 1).copy()

    # 3) for each row, build polygon pts and generate seeds
    fg_seeds, bg_seeds = [], []
    for seg in df["segmentation"]:
        # seg is like [x0,y0,x1,y1,...]
        pts = list(zip(seg[0::2], seg[1::2]))
        fg, bg = gen_seeds(pts,
                           n_inside=args.n_inside,
                           n_outside=args.n_outside)
        fg_seeds.append(fg)
        bg_seeds.append(bg)

    df["fg_seeds"] = fg_seeds
    df["bg_seeds"] = bg_seeds

    # 4) dump out; lists will be JSON‐encoded
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows with seeds → {args.output}")

if __name__ == "__main__":
    main()
