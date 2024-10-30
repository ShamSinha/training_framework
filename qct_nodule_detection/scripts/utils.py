import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from voxdet.metrics.sub_level_analysis import compute_metrics_from_cache, get_all_file_locs


def generate_recall_curves(root, recalls, num_nodules, subgroup_name=None):
    if subgroup_name == "volume":
        x = ["xsmall", "small", "medium", "large", "xlarge"]
    elif subgroup_name == "texture":
        x = ["solid", "part-solid", "non-solid"]
    else:
        raise ValueError('Only volume and texture subgroups supported for now')

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(f'nodule {subgroup_name}')
    ax1.set_ylabel('recall', color=color)
    ax1.plot(x, recalls, color=color)  # Use plot method for line plot
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('# nodules', color=color)
    ax2.plot(x, num_nodules, color=color)  # Use plot method for line plot
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_ylim(min(recalls), max(recalls))
    ax2.set_ylim(min(num_nodules), max(num_nodules))

    fig.tight_layout()
    save_loc = f"{root}/{subgroup_name}_recall_curve.png"
    plt.savefig(save_loc)
    img_name = save_loc.split("/")[-1]
    return img_name

def get_source_df(root, meters, summary_df):
    source_analysis_dict = {}
    mapper = {
        "internal": ["qic"],
        "external": ["wcg", "qxr_fda", "qe"],
        "public": ["dsb_test", "nlst_test", "lidc_test_1reader", "lidc_test_3reader"],
        "multi_reader": ["lidc_test_3reader"],
        "single_reader": ["dsb_test", "lidc_test_1reader", "nlst_test", "qic", "wcg", "qe"],
    }

    for key in mapper:
        df = get_concise_metrics(root, dirs=[mapper[key]], meters=meters, all_dataset_name=key)
        num_scans = summary_df.loc[mapper[key]]['num_scans'].sum()
        df['fpr'] = (df['fp']/num_scans).round(2)
        df.drop(['tp', 'fp', 'fn'], axis=1, inplace=True)
        source_analysis_dict[key] = df.to_dict(orient='records')[0]

    df = pd.DataFrame.from_dict(source_analysis_dict, orient='index')
    return df

def get_concise_metrics(root, dirs, meters, callbacks=None, all_dataset_name='all'):
    df_dict = {}
    for dataset_name in tqdm(dirs):
        file_locs = get_all_file_locs(root, dataset_name)
        all_flag = False
        if isinstance(dataset_name, list):
            dataset_name = "_".join(dataset_name)
            all_flag = True
        show_cols = ["FROC", "recall", "precision", "avg_tp_iou", "tp", "fp", "fn"]
        if callbacks:
            show_cols = ["recall", "tp", "fp", "fn", "avg_tp_iou"]
        df = compute_metrics_from_cache(meters, file_locs, callbacks=callbacks)
        df = df[[c for c in df.columns if c in show_cols]]
        ds_metrics = df.to_dict(orient='records')[0]
        dataset_name = all_dataset_name if all_flag else dataset_name
        df_dict[dataset_name] = ds_metrics

    df = pd.DataFrame.from_dict(df_dict, orient='index')
    return df
