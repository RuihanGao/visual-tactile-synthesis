echo "Download TouchClothing dataset..."
URL="https://drive.google.com/uc?export=download&id=1VlgYpDSxQP70sYpFERHuzKnTNIH4Gf4s"
ZIP_FILE=TouchClothing_dataset.zip
gdown $URL -O $ZIP_FILE
unzip -q $ZIP_FILE
rm $ZIP_FILE