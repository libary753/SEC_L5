# SE_LMM
삼성 멀티모달과제

SE_LMM 폴더 내에 SlideVQA 폴더에 SlideVQA 데이터셋 저장 필요
utils.py에 하드코딩해놨는데, 그냥 맞춰서 가는 게 편할 것 같음

```bash
git clone --recursive https://github.com/libary753/SE_LMM.git
```

```bash
pip install -r requirements.txt
```

되는지 확실하지 않음 하다가 오류 생기면 문의 ㄱㄱ

1. SlideVQA 데이터 인덱싱
```bash
python indexing.py
```

2. Retrieval 평가
```bash
python retrieval.py
```