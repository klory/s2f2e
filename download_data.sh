#!/bin/bash

DATA_DIR="./data/128"
if [ -d "$DATA_DIR" ]; then
  echo "data already downloaded"
else
	mkdir "./data"
	wget -O ./data/128.tar.gz https://drive.google.com/uc\?export\=download\&id\=1im4RFbVL5t0CdK08fu5XmJsyPvSoL1gx
	tar -zxvf ./data/128.tar.gz -C ./data/
fi

