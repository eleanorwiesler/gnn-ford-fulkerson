from __future__ import division
import cv2
import numpy as np
import os
import sys
import pandas as pd
import ast
import time
import argparse
from math import exp, pow
from collections import defaultdict
import networkx as nx
from augmentingPath import augmentingPath

graphCutAlgo = {"ap": augmentingPath}
SIGMA = 60
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
CUTCOLOR = (0, 0, 255)
SOURCE, SINK = -2, -1
SF = 5
SCALE = 100
default_size = 30
radius = 10
thickness = -1
ALL_GROUPS = ["flowers"]

def load_seeds_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    seed_dict = {}
    for _, row in df.iterrows():
        filename = row['file_name']
        fg = ast.literal_eval(row['fg_seeds'])
        bg = ast.literal_eval(row['bg_seeds'])
        seed_dict[filename] = {'fg': fg, 'bg': bg}
    return seed_dict

def boundaryPenalty(ip, iq):
    return int(SCALE * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2))))

def ScaleSeeds(seeds, r, c, radius):
    r0, c0 = seeds.shape
    scaled_seeds = np.zeros((r, c))
    for i in range(r0):
        for j in range(c0):
            if seeds[i][j] in [OBJCODE, BKGCODE]:
                x = i * r // r0
                y = j * c // c0
                cv2.circle(scaled_seeds, (y, x), radius, int(seeds[i][j]), thickness)
    return scaled_seeds

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r:
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c:
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K

def makeTLinks(graph, seeds, K):
    r, c = seeds.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                graph[SOURCE][x] = K
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K
    return

def buildGraph(image, loaded_seeds):
    V = image.size + 2
    graph = {i: defaultdict(int) for i in range(V)}
    makeNLinks(graph, image)

    seeds = np.zeros(image.shape, dtype="uint8")
    for (x, y) in loaded_seeds["fg"]:
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            seeds[y][x] = OBJCODE
    for (x, y) in loaded_seeds["bg"]:
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            seeds[y][x] = BKGCODE

    seededImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    scale_ratio = seededImage.shape[0] // seeds.shape[0]
    seed_radius = seededImage.shape[0] // default_size
    for i in range(seeds.shape[0]):
        for j in range(seeds.shape[1]):
            if seeds[i][j] in [OBJCODE, BKGCODE]:
                color = {OBJCODE: OBJCOLOR, BKGCODE: BKGCOLOR}[seeds[i][j]]
                cv2.circle(seededImage, (j * scale_ratio, i * scale_ratio), seed_radius, color, thickness)

    seeds = ScaleSeeds(seeds, image.shape[0], image.shape[1], image.shape[0] // default_size)
    makeTLinks(graph, seeds, SCALE * V ** 2)
    return graph, seededImage

def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // c, c[0] % c)
            colorPixel(c[1] // c, c[1] % c)
    return image

def imageSegmentation(imagename, folder, group, size=(30, 30), algo="ap", loadseed=None):
    imagefile = os.path.join(folder, imagename)
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)

    cutdir = os.path.join(folder, group + "_cuts", str(size[0]))
    os.makedirs(cutdir, exist_ok=True)

    seeded_image_dir = os.path.join(folder, group + "_seeded", str(size[0]))
    os.makedirs(seeded_image_dir, exist_ok=True)

    V = size[0] * size[1] + 2
    global SOURCE, SINK
    SOURCE = V - 2
    SINK = V - 1

    graph, seededImage = buildGraph(image, loadseed)

    if seededImage is not None:
        cv2.imwrite(os.path.join(seeded_image_dir, imagename.replace(".jpg", "_seeded.jpg")), seededImage)

    start = time.time()
    flows, cuts, path_count, avg_len = graphCutAlgo[algo](graph, V, SOURCE, SINK)
    end = time.time()

    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    save_path = os.path.join(cutdir, imagename.replace(".jpg", "_cuts.jpg"))
    cv2.imwrite(save_path, image)

    print(f"[✓] Processed {imagename}: {save_path}")
    return flows, cuts, path_count, avg_len, graph, end - start, image

def convert_to_networkx(graph_dict, image, source, sink):
    G = nx.DiGraph()
    r, c = image.shape

    # Add all pixel nodes with attributes
    for i in range(r):
        for j in range(c):
            idx = i * c + j
            G.add_node(idx, x=j, y=i, intensity=int(image[i, j]))

    # Add source and sink nodes
    G.add_node(source, type='source')
    G.add_node(sink, type='sink')

    # Add edges with attributes
    for u in graph_dict:
        for v, cap in graph_dict[u].items():
            if cap > 0:
                edge_type = "tlink" if (u == source or v == sink) else "nlink"
                G.add_edge(u, v, capacity=cap, type=edge_type)
    return G

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with seed data")
    parser.add_argument("--group", type=str, default="flowers")
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--save_graphs", action="store_true", help="Save NetworkX graphs as .gpickle")
    args = parser.parse_args()

    seed_dict = load_seeds_from_csv(args.csv)

    for file_name, seeds in seed_dict.items():
        image_path = os.path.join(args.folder, file_name)
        if os.path.exists(image_path):
            flows, cuts, path_count, avg_len, graph, duration, image = imageSegmentation(
                imagename=file_name,
                folder=args.folder,
                group=args.group,
                size=(args.size, args.size),
                algo="ap",
                loadseed=seeds
            )
            if args.save_graphs:
                gray_image = cv2.imread(os.path.join(args.folder, file_name), cv2.IMREAD_GRAYSCALE)
                gray_image = cv2.resize(gray_image, (args.size, args.size))
                nx_graph = convert_to_networkx(graph, gray_image, SOURCE, SINK)
                gname = file_name.replace(".jpg", "_graph.gpickle")
                out_path = os.path.join(args.folder, args.group + "_graphs", gname)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                nx.write_gpickle(nx_graph, out_path)
                print(f"[✓] Saved NetworkX graph to: {out_path}")
        else:
            print(f"[✗] Missing file: {image_path}")

'''
to run:
python original_faster_ff_code/imagesegmentation.py \
  --folder data_pipeline/flowers_segmentation/train \
  --csv data_pipeline/flowers_segmentation/train/_polygons.csv \
  --group flowers \
  --size 30 \
  --save_graphs
'''