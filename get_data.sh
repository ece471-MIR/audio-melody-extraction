#!/bin/bash

# downloads MIR-1K and MIREX05 training datasets

set -e

DATA_DIR="./datasets"
MIR1K_URL="http://mirlab.org/dataset/public/MIR-1K.zip"
MIREX05_URL="https://labrosa.ee.columbia.edu/projects/melody/mirex05TrainFiles.zip"
ADC2004_URL="http://labrosa.ee.columbia.edu/projects/melody/adc2004_full_set.zip"

echo "=============================================="
echo "MIR-1K, MIREX05 and ADC2004 Dataset Downloader"
echo "=============================================="

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -d "MIR-1K" ] && [ -d "MIR-1K/Wavfile" ]; then 
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

if [ -d "adc2004_full_set" ]; then
    echo "ADC2004 already exists, skipping..."
else
    echo "downloading ADC2004..."
    curl -L -k -o adc2004_full_set.zip "$ADC2004_URL"
    echo "Extracting ADC2004..."
    unzip -q adc2004_full_set.zip
    rm adc2004_full_set.zip
    echo "ADC2004 done!"
fi 

cd ..

echo ""
echo ""
echo "=================================="
echo "Download complete!"
echo "=================================="
echo "MIR-1K:   $DATA_DIR/MIR-1K"
echo "MIREX05:  $DATA_DIR/mirex05TrainFiles"
echo "ADC2004:  $DATA_DIR/adc2004_full_set"
echo ""
echo "To run preprocessing:"
echo "  uv run preprocessing.py"
