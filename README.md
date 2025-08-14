# 감정일기 분석 프로젝트

## Overview

- **목표:** 사용자의 감정을 분석하고 시각화
- **기술:** Python, Hugging Face, FAISS, streamlit, OpenAI GPT
- **기간:** 2025.07.28 ~ 2025.08.14 (약 3주)

## Features

- 일기 입력 → 감정 분석
- 감정 점수 및 분포 그래프 시각화
- 유사일기 검색
- 공감 메시지 생성

## Tech Stack

- **NLP 모델:** Hugging Face `klue-bert-base-sentiment`
- **임베딩 & 검색:** Sentence Transformers + FAISS
- **UI:** Streamlit
- **데이터 처리:** Pandas
- **시각화:** Matplotlib
- **API:** OpenAI GPT

## Screenshots

### UI 화면

![스크린샷 2025-08-14 212353.png](attachment:5e6d1e1f-3149-4824-83cd-ad8b01f278cd:스크린샷_2025-08-14_212353.png)

### 감정 분포 그래프

![스크린샷 2025-08-14 121159.png](attachment:92093af3-10c5-4146-95ce-58dfe8deb7f5:스크린샷_2025-08-14_121159.png)

### 감정 점수 추이 그래프

![스크린샷 2025-08-14 121206.png](attachment:53d2fc14-dd4b-4d66-8bf7-091e6ecf3a4c:스크린샷_2025-08-14_121206.png)

## Outcome & Reflection

- **완성도:** 주요 기능 구현 완료
- **배운 점:**
    - Hugging Face 모델로 한국어 감정 분석 구현
    - FAISS를 활용한 로컬 벡터 검색
    - Gradio를 이용한 UI 구성
- **개선점:**
    - GPT API 호출 없이 로컬 LLM 활용 가능
    - 유사일기 검색 속도 최적화
    - UI 반응형 개선 및 디자인 보완
