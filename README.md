<div align="center">

<img src="assets/logo.png" width="80" alt="Pickstape logo"/>

# PICKSTAPE

**AI가 골라 담은 당신만의 테이프**

![Python](https://img.shields.io/badge/Python_3.11+-FFD4E5?style=flat-square&logoColor=FF6B9D)
![Streamlit](https://img.shields.io/badge/Streamlit-FF6B9D?style=flat-square&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-FFF0F5?style=flat-square&logoColor=FF6B9D)
![Qwen](https://img.shields.io/badge/Qwen3.5--4B-FFD4E5?style=flat-square&logoColor=2D1B33)
![Tests](https://img.shields.io/badge/Tests-126_passed-FF6B9D?style=flat-square)

</div>

---

## 📼 프로젝트 소개

**Pickstape**는 대화형 AI 음악 추천 챗봇입니다. f(x)의 *Pink Tape* 앨범에서 영감을 받은 레트로 VHS 카세트 UI로, 사용자의 감정과 상황에 맞는 음악을 골라 담아드립니다.

> **Pick** (고르다) + **Tape** (카세트테이프) = Pickstape
> 챗봇에게 지금 기분, 어떤 상황인지, 좋아하는 곡을 말하면 됩니다.

---

## 🎵 데모 시나리오

| 유형 | 예시 입력 |
|------|-----------|
| 😢 감정 기반 | `"요즘 우울한 기분인데 위로가 되는 노래 추천해줘"` |
| 🎉 상황 기반 | `"친구들이랑 파티 하는데 신나는 곡 골라줘"` |
| 🔍 유사곡 탐색 | `"Blinding Lights랑 비슷한 분위기의 곡 찾아줘"` |

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|------|------|
| **LLM** | Qwen3.5-4B (OpenAI 호환 API, 사내 서버 호스팅) |
| **에이전트** | LangGraph (고정 상태 머신) |
| **UI** | Streamlit + 커스텀 CSS (레트로 VHS 테마) |
| **추천 엔진** | pandas + numpy + scikit-learn (코사인 유사도) |
| **언어** | Python 3.11+ |

---

## 🏗️ 아키텍처

```
사용자 입력
    ↓
[라우터 노드]  ← Qwen3.5-4B: 의도 분류 + 파라미터 추출 (JSON)
    ↓ (intent: emotion / situation / similar)
[추천 노드]   ← Python 추천 엔진: 필터링 + 코사인 유사도
    ↓
[응답 노드]   ← 템플릿 고정 응답 (LLM 자유 생성 미사용)
    ↓
[app.py 후처리]  ← disambiguation 체크, 블랙리스트 필터, dedup
    ↓
추천 카드 4곡 표시
```

### 설계 원칙

**LLM 역할 최소화** — 4B 소형 모델은 의도 분류 + 파라미터 추출만 담당합니다. 추천 품질에 영향을 주는 모든 로직(필터링, 스코어링, 응답 생성)은 Python 코드로 처리합니다. LLM 출력 불안정성을 전제하고, JSON 파싱 실패 시 regex fallback → 재질문의 3단계 방어 구조를 갖춥니다.

### 추천 전략

| 유스케이스 | 1차 필터 | 2차 랭킹 | 반환 |
|-----------|----------|----------|------|
| 감정 기반 | valence / energy 범위 필터 | 코사인 유사도 | Top 4 |
| 상황 기반 | 장르 + 오디오 피처 범위 | popularity 가중 정렬 | Top 4 |
| 유사곡 탐색 | seed곡 검색 (이름/아티스트) | 8개 피처 코사인 유사도 | Top 4 |

---

## ✨ 주요 기능

- **3가지 추천 모드** — 감정 / 상황 BGM / 유사곡 탐색
- **비슷한 곡 찾기** — 추천 카드에서 바로 유사곡 탐색 (LLM 라우터 우회, 엔진 직접 호출)
- **동명이곡 disambiguation** — 동일 제목 곡이 여러 아티스트에게 있을 경우 재질문
- **오디오 피처 시각화** — 카드 하단 E/D/A/I/V/T 바 차트 (Energy, Danceability, Acousticness, Instrumentalness, Valence, Tempo)
- **Spotify 연동** — 추천 곡을 Spotify에서 바로 열기
- **안전 필터** — 블랙리스트 word-boundary 필터 + dedup

---

## 🚀 실행 방법

### 사전 요구사항

```bash
Python 3.11+
pip install -r requirements.txt
```

### 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성합니다:

```env
LLM_BASE_URL=http://<서버주소>/v1
LLM_MODEL=qwen3.5-4b
LLM_API_KEY=
```

### 실행

```bash
streamlit run app.py
```

---

## 🧪 테스트

```bash
python -m pytest tests/ -v
```

**126개 테스트** (test_ui.py · test_agent.py · test_engine.py · test_recommender.py)

| 파일 | 테스트 수 | 주요 검증 |
|------|-----------|-----------|
| `test_ui.py` | 40 | 챗봇 플로우, 라우팅, 후처리, 피드백, 유사곡 탐색, 엣지케이스 |
| `test_agent.py` | 36 | JSON 파싱, fallback, 전체 파이프라인 통합 |
| `test_engine.py` | 40 | 감정/상황/유사곡 추천, 장르 필터, fallback cascade |
| `test_recommender.py` | 10 | 전처리, 정규화, 피처 매트릭스 |

---

## 📁 프로젝트 구조

```
pickstape/
├── app.py                         # Streamlit 진입점 + 후처리 게이트
├── requirements.txt
├── data/
│   └── Music_recommendation.csv  # 32,833곡 Spotify 데이터셋
├── src/
│   ├── agent/
│   │   ├── graph.py              # LangGraph 파이프라인 컴파일
│   │   ├── nodes.py              # 라우터 / 추천 / 응답 노드
│   │   ├── prompts.py            # 시스템 프롬프트 + few-shot
│   │   └── state.py              # AgentState TypedDict
│   ├── recommender/
│   │   ├── engine.py             # 추천 엔진 (필터링 + 코사인 유사도)
│   │   ├── preprocess.py         # 데이터 전처리 + 정규화
│   │   └── mappings.py           # 감정/상황 → 피처 매핑 테이블
│   └── ui/
│       ├── components.py         # Streamlit 컴포넌트 (카드, 사이드바, 챗 버블)
│       └── styles.py             # 레트로 VHS CSS 테마
├── assets/
│   └── logo.png                  # Pickstape 로고 (투명 PNG)
└── tests/
    ├── conftest.py
    ├── test_ui.py
    ├── test_agent.py
    ├── test_engine.py
    └── test_recommender.py
```

---

## 💿 데이터셋

- **출처**: Spotify 플레이리스트 기반 공개 데이터셋
- **규모**: 32,833행 / 고유 곡 28,356개 / 고유 아티스트 10,692명
- **장르**: edm, rap, pop, r&b, latin, rock (6개)
- **오디오 피처**: valence, energy, danceability, acousticness, speechiness, instrumentalness, liveness, tempo, loudness

---

## 🎨 디자인 컨셉

<div align="center">

f(x) *Pink Tape* 앨범 VHS 카세트 모티프<br/>
크림 배경 `#FFF8F0` · 연핑크 `#FFF0F5` · 핫핑크 포인트 `#FF6B9D`<br/>
Galmuri11 픽셀 폰트

</div>

```
┌──────────────────────────────────────┐
│ ██ SIDE A ██████████████████ ◎  ◎  │  ← 핫핑크 #FF6B9D 헤더
│ ┌────────────────────────────────┐  │
│ │ Blinding Lights                │  │  ← 흰 내부 패널
│ │ The Weeknd · After Hours       │  │
│ │ [pop]  [r&b]                   │  │
│ │ ▶ Spotify에서 열기              │  │
│ └────────────────────────────────┘  │
│  ▮  ▮▮   ▮    ▮    ▮▮             │  ← 오디오 피처 바
│  E   D   A    I    V   T           │
└──────────────────────────────────────┘
```

---

## 💡 기술 의사결정 포인트

- **LangGraph를 자율 에이전트가 아닌 고정 상태 머신으로 사용** — 소형 LLM의 루프/오류 발생률을 낮추기 위해 라우터→추천→응답의 선형 플로우로 고정
- **ChromaDB 대신 pandas + 코사인 유사도** — 32,833행 규모에서 벡터 DB 오버헤드 불필요, 배포 의존성 최소화
- **LLM 자유 생성 제거, 템플릿 전용** — 4B 모델의 한국어 품질 불안정 문제를 근본적으로 해결
- **Mental_Health_Label 비노출** — 내부 검색 신호로만 사용, 사용자에게 진단명 노출 금지
