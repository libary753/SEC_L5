# L5. 멀티모달 학습 데이터 생성 및 활용 연구

## 1. 환경 셋업
```bash
git clone --recursive https://github.com/libary753/SEC_L5.git
cd SEC_L5
conda create -n sec_l5 python=3.10
conda activate sec_l5
pip install -e ./submodules/janus
pip install -r requirements.txt
```

## 2. 데이터셋 준비
루트 경로 내에 SlideVQA 데이터셋 복사

> SEC_LMM  
> ├ SlideVQA  
> │ ├ annotations  
> │ ├ images  
> │ ├ ...  
> ├ ...

## 3. Text-to-Document retrieval

### 3.1. Parsing
python indexing.py --out_dir output/indexing_t2d

### 3.3. Retrieval
python retrieval_t2d.py --out_dir output/indexing_t2d

## 4. Image-to-Document retrieval

### 4.1. Parsing
python indexing.py --out_dir output/indexing_i2d

### 4.2. Retrieval
python retrieval_i2d.py --out_dir output/indexing_t2d