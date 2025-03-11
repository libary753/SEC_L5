gpu=0

# indexing
CUDA_VISIBLE_DEVICES=$gpu python indexing.py --out_dir output/indexing_i2d --skip_1st_page True

# retrieval
CUDA_VISIBLE_DEVICES=$gpu python retrieval.py