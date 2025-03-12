gpu=1

# indexing
CUDA_VISIBLE_DEVICES=$gpu python parsing.py --out_dir output/indexing_t2d --skip_1st_page False

# retrieval
python retrieval_t2d.py