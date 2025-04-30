import json

# open json file
with open('/Users/tracebaxley/Desktop/gnn-ford-fulkerson-1/data_pipeline/flowers_segmentation/train/_annotations.coco.json', 'r') as f:
    coco_data = json.load(f)

# track images with >1 polygon in any annotation
invalid_image_ids = set()

for ann in coco_data['annotations']:
    segmentation = ann['segmentation']
    # Check if segmentation is a list of multiple polygons
    if isinstance(segmentation, list) and len(segmentation) > 1:
        # Check if any polygon is itself a list (COCO format)
        if any(isinstance(poly, list) for poly in segmentation):
            invalid_image_ids.add(ann['image_id'])

# Filter out invalid images and their annotations
filtered_images = [img for img in coco_data['images'] if img['id'] not in invalid_image_ids]
filtered_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] not in invalid_image_ids]

# Update the COCO data
coco_data['images'] = filtered_images
coco_data['annotations'] = filtered_annotations

# Save the filtered annotations to a new file
with open('filtered_annotations.json', 'w') as f:
    json.dump(coco_data, f, indent=2)

print(f"Filtered out {len(invalid_image_ids)} invalid images.")