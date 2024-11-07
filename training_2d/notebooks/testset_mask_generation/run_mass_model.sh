#!/bin/bash

# Total number of samples in your test_df
TOTAL_SAMPLES=$(wc -l < /fast_data_e2e_1/cxr/qxr_ln_data/LN_test/combined_test_csv_updated_internal_test_13-08-24.csv)

# Divide the dataset into 4 equal parts
PART_SIZE=$((TOTAL_SAMPLES / 4))

# Define start and end indices for each process
START_IDX_1=0
END_IDX_1=$((PART_SIZE - 1))

START_IDX_2=$PART_SIZE
END_IDX_2=$((2 * PART_SIZE - 1))

START_IDX_3=$((2 * PART_SIZE))
END_IDX_3=$((3 * PART_SIZE - 1))

START_IDX_4=$((3 * PART_SIZE))
END_IDX_4=$((TOTAL_SAMPLES - 1))

# Run 4 processes in parallel
# python /home/users/shreshtha.singh/qxr_training/notebooks/testset_mask_generation/generate_mass_segmentation.py $START_IDX_1 $END_IDX_1 1 &
# python /home/users/shreshtha.singh/qxr_training/notebooks/testset_mask_generation/generate_mass_segmentation.py $START_IDX_2 $END_IDX_2 2 &
# python /home/users/shreshtha.singh/qxr_training/notebooks/testset_mask_generation/generate_mass_segmentation.py $START_IDX_3 $END_IDX_3 3 &
python /home/users/shreshtha.singh/qxr_training/notebooks/testset_mask_generation/generate_mass_segmentation.py $START_IDX_4 $END_IDX_4 4 &
