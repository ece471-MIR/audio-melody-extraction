#!/bin/bash

# downloads MIR-1K and MIREX05 training datasets

set -e

DATA_DIR="./datasets"
MIR1K_URL="http://mirlab.org/dataset/public/MIR-1K.zip"
MIREX05_URL="https://labrosa.ee.columbia.edu/projects/melody/mirex05TrainFiles.zip"

echo "=================================="
echo "AH1 dataset downloader"
echo "=================================="

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -d "MIR-1K" ] && [ -d "MIR-1K/wavfile" ]; then 
    echo "MIR-1K already exists, skipping..."
else
    echo "downloading MIR-1K..."
    curl -L -o MIR-1K.zip "$MIR1K_URL"
    echo "Extracting MIR-1K..."
    unzip -q MIR-1K.zip
    rm MIR-1K.zip
    echo "MIR-1K done!"
fi 

if [ -d "mirex05TrainFiles" ]; then
    echo "MIREX05 already exists, skipping..."
else
    echo "downloading MIREX05..."
    curl -L -k -o mirex05TrainFiles.zip "$MIREX05_URL"
    echo "Extracting MIREX05..."
    unzip -q mirex05TrainFiles.zip
    rm mirex05TrainFiles.zip
    echo "MIREX05 done!"
fi 

cd ..

echo ""
echo ""
echo "=================================="
echo "Download complete!"
echo "=================================="
echo "MIR-1K:   $DATA_DIR/MIR-1K"
echo "MIREX05:  $DATA_DIR/mirex05TrainFiles"
echo ""
echo "To run preprocessing:"
echo "  uv run preprocessing.py"
