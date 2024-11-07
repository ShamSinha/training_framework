#!/bin/bash

directory="/cache/fast_data_nas8/qct/shubham/fpr_cache/test"

train_directory="/cache/fast_data_nas8/qct/shubham/fpr_cache_48_48_16/train"
val_directory="/cache/fast_data_nas8/qct/shubham/fpr_cache_48_48_16/val"

dst= "/cache/fast_data_nas8/qct/shubham/fpr_cache_48_48_16/test"

# Get list of filenames
files=$(ls "$directory")

# Iterate over the list of filenames and do something with each file
for file in $files; do
    train_filepath="${train_directory}/${file}"
    val_filepath="${val_directory}/${file}"

    # Check if it is a file
    if [ -f "${train_filepath}" ]; then
        echo "${train_filepath} yes"
        # mv -v "$train_filepath" "$dst/"
    # else
    #     echo "${val_filepath}"
        # mv -v "$val_filepath" "$dst/"
    fi
  echo "$file"
done


