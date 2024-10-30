import pandas as pd
import torch
import numpy as np
from voxdet.bbox_func.bbox_iou import calculate_iou
from qct_data.merge_ctscan_annots import get_matches, cluster_annotations
from loguru import logger
from tqdm.auto import tqdm
from qct_utils.schema.dim import BBox3D ,BBoxC3D

def find_enclosing_bbox(bboxes):
    """
    Finds the smallest bounding box that encloses all other bounding boxes.
    
    Parameters:
    bboxes (list of tuples): List of tuples where each tuple represents a bounding box in the form (x1, y1, x2, y2).

    Returns:
    tuple: The enclosing bounding box in the form (x_min, y_min, x_max, y_max).
    """
    # Initialize min and max coordinates with extreme values
    x_min = float('inf')
    y_min = float('inf')
    z_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')
    z_max = float('-inf')

    # Loop through each bounding box
    for (z1, y1, x1, z2,y2,x2) in bboxes:
        # Update the min and max coordinates
        x_min = min(x_min, x1, x2)
        y_min = min(y_min, y1, y2)
        z_min = min(z_min,z1,z2)
        x_max = max(x_max, x1, x2)
        y_max = max(y_max, y1, y2)
        z_max = max(z_max,z1,z2)

    # Return the enclosing bounding box
    return [z_min, y_min, x_min, z_max, y_max, x_max]

df = pd.read_csv("incorrect_gt.csv")

for i in tqdm(df.index) :
    row = df.loc[i]
    dataset = row["dataset"]
    sid = row["scan_name"]
    bx = torch.load( f"/cache/fast_data_nas8/qct/shubham/det_gt_annot/{dataset}/{sid}.pt",
                        map_location="cpu")

    boxes = bx["bbox"]
    gt_boxes = []
    for box in boxes :
        bboxc3d = BBoxC3D.from_xcyczcwhd(np.array(box)[[1,0,2,4,3,5]])
        out = bboxc3d.to_xyz()
        final_gt_box = out[[2,1,0,5,4,3]]
        gt_boxes.append(final_gt_box.tolist())


    gt_boxes = np.array(gt_boxes)


    iou_matrix = calculate_iou(gt_boxes , gt_boxes)
    np.fill_diagonal(iou_matrix, 0)
    matches = get_matches(iou_matrix, iou_thr= 0.1)
    mappings = cluster_annotations(matches, len(gt_boxes)) #map annotations

    final_gt_boxes = []
    for m in mappings : 
        box = [gt_boxes[i] for i in m]
        box = find_enclosing_bbox(box)
        final_gt_boxes.append(box)

    bx['gt_boxes'] = np.array(final_gt_boxes)

    new_meta = []
    for m in mappings :
        if len(m) > 1 : 
            max_dict = max((bx["meta"][idx] for idx in m), key=lambda d: d['long_axis_diameter'])
            new_meta.append(max_dict)
        else :
            new_meta.append(bx["meta"][m[0]])

    bx['meta'] = new_meta
    torch.save(bx , f"/cache/fast_data_nas8/qct/shubham/det_gt_annot/{dataset}/{sid}.pt")


