from monai.metrics.utils import get_mask_edges, get_surface_distance

# faster way, using monai
def compute_hausdorff_monai(pred, gt, max_dist):
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist