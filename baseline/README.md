# 2022 인공지능 온라인 경진대회
## [자연어] 스마트카 음성인식 고도화 문제

해당 베이스라인 코드는 https://github.com/sooftware/kospeech 내용을 일부 발췌하고 경진대회에 적절한 형태로 수정하였음을 밝힙니다.

### 코드 구조

```
${PROJECT}
├── config/
│   ├── audio/
│   ├── eval/
│   ├── model/
│   ├── train/
│   ├── eval.yaml
│   └── train.yaml
├── data/
│   ├── trainscripts/
│   │   ├── train.csv
│   │   └── sample_submission.csv
│   └── vocab/
├── kospeech/
├── modules/
│   ├── earlystopper.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── recorders.py
│   ├── trainer.py
│   └── utils.py
├── results/
│   ├── train/
│   └── predict
├── README.md
├── train.py
├── preprocess.py
└── predict.py
```

- config : 학습/추론에 필요한 파라미터 등을 기록하는 yaml 파일
- data  
    - trainscripts/ : 제공한 train.csv와 sample_submission.csv을 포함하는 디렉토리  
    - vocab/ : train.csv을 통해 생성한 character-level vocabulary (vocab.csv)가 저장될 디렉토리
- kospeech : kospeech에서 사용되는 modules
- modules
    - earlystopper.py : loss가 특정 에폭 이상 개선되지 않을 경우 멈춤
    - losses.py : config에서 지정한 loss function을 리턴
    - metrics.py : config에서 지정한 metric을 리턴
    - optimizers.py : config에서 지정한 optimizer를 리턴
    - recorders.py : log, learning curve, best model.pt 등을 기록
    - trainer.py : 에폭 별로 수행할 학습 과정
    - utils.py : 여러 확장자 파일을 불러오거나 여러 확장자로 저장하는 등의 함수
- results
  - train/ : 학습 log를 기록하는 디렉토리
  - predict/ : 추론 log를 기록하는 디렉토리
- train.py : 학습 시 실행하는 코드
- predict.py : 추론 시 실행하는 코드
- preprocess.py : 전처리 코드
- setup.py : kospeech package 설치 


---

### Install
1. `pip install -e .`를 terminal에 실행

### 전처리
1. train 데이터 전처리
   1. `python preprocess.py --dataset_path=./data/transcripts/train.csv`
2. test 데이터 전처리
   2. `python preprocess.py --dataset_path=./data/transcripts/sample_submission.csv`

### 학습

1. `config/train/ds2_train.yaml` 수정
    1. dataset_path : 데이터 경로 지정 (.wav 파일들이 위치한 디렉토리)
    2. 이외 파라미터 조정
2. `python train.py`를 terminal에 실행
3. `results/train/` 내에 결과(weight, log, plot 등)가 저장됨


### 추론

1. `config/eval/defualt.yaml` 수정
    1. dataset_path : 데이터 경로 지정 (.wav 파일들이 위치한 디렉토리)
    2. model_path : 추론을 수행할 모델의 weight 경로
    3. train_serial : 추론을 수행할 모델이 존재하는 train serial 
2. `python predict.py`를 terminal에 실행
3. `results/train/` 내에 결과 파일(prediction.csv)이 저장됨

