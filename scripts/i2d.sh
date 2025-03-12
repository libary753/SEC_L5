gpu=0

# parsing
CUDA_VISIBLE_DEVICES=$gpu python parsing.py --out_dir output/parsing_i2d --skip_1st_page True

# retrieval
CUDA_VISIBLE_DEVICES=$gpu python retrieval_i2d.py