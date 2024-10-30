#!/bin/bash

# Define source and destination directories
src="/cache/fast_data_nas8/qct/shubham/fpr_cache/train"
dst="/cache/fast_data_nas8/qct/shubham/fpr_cache/test"

# Navigate to source directory
cd "$src"

# Calculate total files and the number of files to move
total_files=$(ls -1 | wc -l)
percent_to_move=10  # Change this to your desired percentage
files_to_move=$((total_files * percent_to_move / 100))

# Move the files
ls -1 | head -n $files_to_move | xargs -I {} mv {} "$dst/"

echo "Moved $files_to_move files from $src to $dst."