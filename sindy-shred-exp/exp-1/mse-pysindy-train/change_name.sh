#!/bin/bash

echo "Starting filename correction for mse_pysidny_train_*.npy files..."

# Find all files with the typo pattern
for file in mse_pysidny_train_*.npy; do
    # Check if the file exists (in case there are no matches)
    if [ -f "$file" ]; then
        # Extract the index number using pattern matching
        index=$(echo "$file" | sed 's/mse_pysidny_train_\(.*\)\.npy/\1/')
        
        # Create new filename with correct spelling
        new_file="mse_pysindy_train_${index}.npy"
        
        # Rename the file
        mv "$file" "$new_file"
        echo "Renamed: $file → $new_file"
    fi
done

echo "Filename correction complete!"
