#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ast
import os
import sys

def overlay_and_save(csv_path, images_dir, image_filename, output_path):
    # 1) load the CSV and parse the seed lists
    df = pd.read_csv(
        csv_path,
        converters={
            'fg_seeds': ast.literal_eval,
            'bg_seeds': ast.literal_eval
        }
    )

    # 2) pick the row for this image
    row = df[df['file_name'] == image_filename]
    if row.empty:
        raise FileNotFoundError(f"No entry for '{image_filename}' in {csv_path}")
    row = row.iloc[0]
    fg_seeds = row['fg_seeds']
    bg_seeds = row['bg_seeds']

    # 3) load the image
    img_path = os.path.join(images_dir, image_filename)
    img = mpimg.imread(img_path)

    # 4) plot
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    if fg_seeds:
        x_fg, y_fg = zip(*fg_seeds)
        plt.scatter(x_fg, y_fg, s=50, facecolors='none', edgecolors='r', label='Foreground')
    if bg_seeds:
        x_bg, y_bg = zip(*bg_seeds)
        plt.scatter(x_bg, y_bg, s=50, marker='x', c='b', label='Background')

    plt.axis('off')
    plt.legend(loc='upper right')

    # 5) save to disk
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Overlay saved to {output_path}")
    # (optional) show on screen:
    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python overlay_seeds.py <csv_path> <images_dir> <image_filename> <output_png>")
        sys.exit(1)

    _, csv_path, images_dir, image_filename, output_png = sys.argv
    overlay_and_save(csv_path, images_dir, image_filename, output_png)
