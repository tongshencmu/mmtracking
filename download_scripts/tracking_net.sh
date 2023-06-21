ACCESS_TOKEN=$1

mkdir trackingnet
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1-bwBQP949zSDAESdTWSWvUOjPgESSF9U?alt=media -o trackingnet/TRAIN_0.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1Q-3RQyKEN4qe402-iAbFPTgEga1E054v?alt=media -o trackingnet/TRAIN_1.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1vdEkKNbNa-oTOUOFdui__6_6IdvQp0C3?alt=media -o trackingnet/TRAIN_2.zip
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1Nmep9J3nowTRpl4QYt1uT1RPehovZzBF?alt=media -o trackingnet/TRAIN_3.zip

curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1YwO9o7370zG9gQUaSQzOSkeoyGNz7nrE?alt=media -o trackingnet/TRAIN_3.zip