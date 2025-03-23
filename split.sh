#!/bin/bash

# Set the folder name (without .tar.gz)
FOLDER_NAME="subsampled_multicoil_train_10"

# Create a tar.gz archive from the folder
tar czf "${FOLDER_NAME}.tar.gz" "$FOLDER_NAME" && echo "DONE"

# Show the size of the compressed file in human-readable format
ls -lh "${FOLDER_NAME}.tar.gz"
