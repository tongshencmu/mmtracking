ACCESS_TOKEN=$1

mkdir lasot
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1KZh6TztW_DPBOWPCT433kjyJDIjpgEYA?alt=media -o lasot/LaSOTBenchmark.zip

mkdir lasot_ext
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1jPKQnKoGzJ5BppgXL6TZMEoHRaP3cr_a?alt=media -o lasot_ext/LaSOT_extension.zip

mkdir got10k
curl -H "Authorization: Bearer $ACCESS_TOKEN" https://www.googleapis.com/drive/v3/files/1C_SwWF9421WDIXBgOkXAjLiyEDc1hcPh?alt=media -o got10k/got10k.zip