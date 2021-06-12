echo "downloading large folders from drive as zip..."

gdown --id 1O-qra6DER4ZgCtoGDU0odpxU7ExyX8m4

echo "unzipping..."

unzip -q data.zip

echo "removing zip file..."

rm data.zip

echo "done"