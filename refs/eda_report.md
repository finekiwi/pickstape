# EDA Report — Music Recommendation Dataset

> Generated: 2026-04-07 | Script: `scripts/eda.py`

---

## 0. Executive Summary

1. **데이터 품질은 양호**: 결측치 0.02%, 중복 track_id 4,477건이지만 90.1%는 1개 플레이리스트에만 등장. `track_id` + `track_popularity max` 기준 dedup으로 28,356행으로 정리.
2. **장르 분포는 균형적**: edm 18.4%~rock 15.1%로 6개 장르가 고르게 분포. 추천 결과 다양성 확보에 유리.
3. **매핑 대부분 양호, 수면만 조정 필요**: MOOD_MAPPING 7개 전부 300건 이상. SITUATION_MAPPING에서 `수면`이 234건으로 유일하게 기준 미달 → loudness 범위 완화 필요.
4. **`excited` → Bipolar (Mania) 98.9% 연관**: Mental_Health_Label이 추천 신호로 실제로 유효함을 확인. 단, 사용자에게 라벨명 직접 노출 금지 원칙 유지.
5. **`loudness` 코사인 유사도 제외 확정**: energy–loudness 상관 0.677. 두 피쳐 동시 포함 시 에너지 방향 과대평가. loudness는 situation 하드 필터로만 사용.

---

## 1. Dataset Overview

| 항목 | 값 |
|------|-----|
| 총 행 수 | 32,833 |
| 컬럼 수 | 25 |
| 고유 track_id | 28,356 |
| 고유 아티스트 | 10,692 |
| 고유 앨범 | 22,545 |
| 고유 플레이리스트 | 471 |

### 결측치

| column | missing_count | pct |
|--------|--------------|-----|
| track_name | 5 | 0.02 |
| track_artist | 5 | 0.02 |
| track_album_name | 5 | 0.02 |

실질적 영향 없음. 전처리 시 해당 5행 삭제.

### 중복 track_id 분석

| n_playlists | n_tracks | pct |
|------------|---------|-----|
| 1 | 25,538 | 90.1% |
| 2 | 2,113 | 7.5% |
| 3 | 481 | 1.7% |
| 4+ | 224 | 0.8% |

**결론**: 4,477행이 중복 track_id지만, 같은 곡이 여러 플레이리스트에 실린 것. 코사인 유사도 검색에서 동일 곡 중복 반환을 막으려면 **dedup 필수**.

**Dedup 전략**: `track_id` 기준 중복 제거, `track_popularity` 최대값 행 유지. 이유: 여러 플레이리스트 중 가장 인기 있는 버전(메타데이터가 정확할 가능성 높음)을 남기고, popularity 기반 정렬에도 올바른 값 사용.

---

## 2. Audio Feature Distributions

| feature | mean | std | min | p25 | median | p75 | max |
|---------|------|-----|-----|-----|--------|-----|-----|
| danceability | 0.655 | 0.145 | 0.0 | 0.563 | 0.672 | 0.761 | 0.983 |
| energy | 0.699 | 0.181 | 0.0 | 0.581 | 0.721 | 0.840 | 1.0 |
| loudness | -6.719 | 2.988 | -46.448 | -8.171 | -6.166 | -4.645 | 1.275 |
| speechiness | 0.107 | 0.101 | 0.0 | 0.041 | 0.062 | 0.132 | 0.918 |
| acousticness | 0.175 | 0.220 | 0.0 | 0.015 | 0.080 | 0.255 | 0.994 |
| instrumentalness | 0.085 | 0.224 | 0.0 | 0.0 | 0.0 | 0.005 | 0.994 |
| liveness | 0.190 | 0.154 | 0.0 | 0.093 | 0.127 | 0.248 | 0.996 |
| valence | 0.511 | 0.233 | 0.0 | 0.331 | 0.512 | 0.693 | 0.991 |
| tempo | 120.881 | 26.904 | 0.0 | 99.96 | 121.98 | 133.92 | 239.44 |

**주요 관찰**:
- `instrumentalness` 중앙값 0.0, p75 0.005 → 대부분의 곡이 보컬 있음. 75번째 백분위 이상만 진정한 인스트루멘탈.
- `acousticness` p75가 0.255로 낮음 → 대부분 일렉트릭/전자 사운드. 어쿠스틱 필터 적용 시 결과 수가 급감할 수 있음.
- `loudness` 평균 -6.7 dB, `tempo` 평균 121 BPM → 두 피쳐는 스케일이 다른 나머지 피쳐들(0~1)과 다름. **min-max 정규화 필수**.

---

## 3. Correlation Analysis

**|r| ≥ 0.4인 피쳐 쌍:**

| feature_a | feature_b | r |
|-----------|-----------|---|
| energy | loudness | 0.677 |
| energy | acousticness | -0.540 |

**코사인 유사도 설계 시사점**:

- **energy–loudness (r=0.677)**: 두 피쳐를 동시에 포함하면 코사인 거리 계산 시 "에너지" 방향이 사실상 2배 가중됨. `loudness`를 코사인 피쳐 셋에서 제외하고 situation mapping의 하드 필터(`수면` 조건)로만 사용.
- **energy–acousticness (r=-0.540)**: 높은 에너지 곡은 어쿠스틱하지 않고, 낮은 에너지 곡은 어쿠스틱한 경향. 두 피쳐가 반대 방향 정보를 담음 → 의미상 중복이 적어 둘 다 유지.
- 나머지 쌍은 |r| < 0.4: 각 피쳐가 독립적 정보를 제공한다고 볼 수 있음.

---

## 4. Category Distributions

### playlist_genre

| genre | count | pct |
|-------|-------|-----|
| edm | 6,043 | 18.4% |
| rap | 5,746 | 17.5% |
| pop | 5,507 | 16.8% |
| r&b | 5,431 | 16.5% |
| latin | 5,155 | 15.7% |
| rock | 4,951 | 15.1% |

6개 장르가 15~18% 범위로 균형 분포. 특정 장르 편향 없이 추천 결과 다양성 자연스럽게 확보 가능.

### Mental_Health_Label

| label | count | pct |
|-------|-------|-----|
| Normal/Unclassified | 22,145 | 67.4% |
| Bipolar (Mania) | 5,016 | 15.3% |
| Anxiety | 4,922 | 15.0% |
| Bipolar (Depression) | 335 | 1.0% |
| Depression | 138 | 0.4% |
| 기타 (Autism, Schizophrenia 등) | 307 | 0.9% |

상위 3개 라벨(Normal, Mania, Anxiety)이 전체의 97.7%. 나머지는 샘플 수 부족으로 검색 신호로 쓰기 어려움.

**사용 방침**: Normal/Mania/Anxiety 3개 라벨은 **하드 게이트가 아니라 소프트 랭킹 신호 또는 fallback 필터로만 활용**. 이유: 이 3개 라벨로 선필터 후 오디오 조건을 적용하면 저커버리지 mood/situation의 후보 풀이 급감함 (예: `코딩` 381 → 284, `수면` 234 → 109 — 섹션 5 참고). 라벨 필터를 먼저 거는 하드 게이트 방식은 검증되지 않았으며, 특히 `수면`·`코딩` 경로에서 결과 없음(no-result) 위험이 있음.

### Mental_Health_Label × Genre 교차 분포 (주요 라벨)

| label | edm | latin | pop | r&b | rap | rock |
|-------|-----|-------|-----|-----|-----|------|
| Anxiety | 53% | 6% | 13% | 5% | 11% | 12% |
| Bipolar (Mania) | 12% | 28% | 17% | 13% | 13% | 18% |
| Normal/Unclassified | 13% | 15% | 18% | 20% | 20% | 15% |

**Anxiety 라벨이 edm에 53% 집중** → `angry` mood 필터 결과가 edm 편향으로 이어지는 직접 원인.

### instrument

| instrument | count | pct |
|------------|-------|-----|
| Unknown | 27,627 | 84.1% |
| Guitar/Drums | 5,015 | 15.3% |
| Synth | 191 | 0.6% |

Unknown이 84%라 추천 피쳐로 활용 불가. **제외 확정**.

---

## 5. Mapping Validation

### MOOD_MAPPING

`post_label_filter`: audio 조건 적용 전, `Mental_Health_Label ∈ {Normal/Unclassified, Bipolar (Mania), Anxiety}` 선필터 후 잔여 행 수. 이 수치가 낮으면 라벨 필터를 하드 게이트로 쓸 수 없음. **`sad`(928→431)와 `empty`(2,108→1,948)는 특히 감소폭이 큼** — `sad`의 경우 top-3 라벨에 해당하지 않는 Depression/Bipolar Depression 성향 음악이 상당 비중을 차지함.

| mood | matched | post_label_filter | coverage_% | status | top_label | genre_bias |
|------|---------|------------------|-----------|--------|-----------|------------|
| happy | 11,212 | 11,176 | 34.2% | ✅ | Normal/Unclassified (54.9%) | none |
| sad | 928 | 431 | 2.8% | ✅ | Normal/Unclassified (46.4%) | none |
| angry | 5,466 | 5,428 | 16.6% | ✅ | **Anxiety (73.1%)** | **edm (48% vs 18%)** |
| calm | 932 | 896 | 2.8% | ✅ | Normal/Unclassified (96.1%) | r&b (38% vs 16%) |
| excited | 5,072 | 5,061 | 15.5% | ✅ | **Bipolar Mania (98.9%)** | none |
| anxious | 4,346 | 4,291 | 13.2% | ✅ | Normal/Unclassified (78.9%) | none |
| empty | 2,108 | 1,948 | 6.4% | ✅ | Normal/Unclassified (92.4%) | none |

**해석**:
- **`excited`**: Bipolar (Mania) 98.9% → valence(0.7~1.0) + energy(0.7~1.0) 범위가 Mania 라벨 음악과 거의 완벽하게 겹침. Mental_Health_Label 신호가 감정 추천에 실제로 유효하다는 근거.
- **`angry`**: Anxiety 라벨이 73.1%고 edm 편향 심함. 음악의 "분노" 느낌이 데이터셋에서는 Anxiety 라벨 + high energy edm으로 표현됨. 의도된 매핑이지만, 추천 시 장르 다양성이 부족할 수 있음 → 엔진에서 genre_pref 파라미터로 보완 가능.
- **`calm`**: r&b 편향(38%). low energy + mid valence가 r&b와 잘 맞음. 큰 문제는 아니나 발표 시 언급할 수 있는 관찰.
- **`sad`, `calm`**: 각각 928, 932건으로 풀 사이즈가 작은 편. 추천 5개를 뽑기엔 충분하지만 반복 추천 시 동일 곡 반환 가능성 있음 → 나중에 고도화 시 범위 조정 여지 있음.

### SITUATION_MAPPING

`post_label_filter`: audio 조건 적용 전, `Mental_Health_Label ∈ {Normal/Unclassified, Bipolar (Mania), Anxiety}` 선필터 후 잔여 행 수. **`코딩`(381→284)과 `수면`(234→109)은 라벨 하드 게이트 적용 불가** — 후보 풀이 300 미만으로 급감함. `카페`(1,219→848)도 40% 감소로 주의 필요.

| situation | matched | post_label_filter | coverage_% | status | genre_bias |
|-----------|---------|------------------|-----------|--------|------------|
| 카페 | 1,219 | 848 | 3.7% | ✅ | r&b (38% vs 16%) |
| 파티 | 4,962 | 4,957 | 15.1% | ✅ | none |
| 코딩 | 381 | 284 | 1.2% | ✅ | none |
| 운동 | 10,836 | 10,807 | 33.0% | ✅ | edm (39% vs 18%) |
| 수면 | 234 | 109 | 0.7% | ⚠️ | none |
| 드라이브 | 6,239 | 6,219 | 19.0% | ✅ | none |

**수면 범위 조정 필요**:

현재: `energy=(0.0, 0.3)`, `acousticness=(0.4, 1.0)`, `loudness=(-46.0, -15.0)`

문제: loudness -15 dB 이하는 전체 데이터에서 극히 일부 (p25가 -8.2 dB). 매우 조용한 곡만 필터링됨.

**조정안**: `loudness=(-46.0, -10.0)`으로 완화
- 이유: 수면용 음악은 "매우 조용함"(-15 이하)보다 "조용함"(-10 이하)으로 기준을 넓혀도 energy/acousticness 조건이 이미 분위기를 충분히 잡아줌.
- 예상 매칭 건수: loudness ≤ -10 dB이 전체의 약 20% → 다른 조건과 교집합으로 300건 이상 확보 가능.

**운동/카페 편향**:
- `운동` edm 39%: high energy + high tempo가 edm과 자연스럽게 맞음. 운동 BGM이 edm에 쏠리는 건 사용자 기대와 일치할 가능성 높음 → 허용.
- `카페` r&b 38%: 편향이지만, low energy + acoustic + no speech 조합이 r&b 특성과 실제로 맞음. 허용.

---

## 6. Preprocessing Recommendations

### 정규화
| feature | 현재 스케일 | 처리 방법 |
|---------|-----------|---------|
| loudness | -46~1 dB | min-max → [0, 1] |
| tempo | 0~239 BPM | min-max → [0, 1] |
| 나머지 7개 | 0~1 | 이미 정규화됨, 그대로 유지 |

### 중복 제거
```
dedup 기준: track_id
남길 행: track_popularity 최대값 (동일 시 첫 번째 행)
결과: 32,833 → 28,356행
```

### 결측치 처리
- track_name, track_artist, track_album_name 각 5건 → 삭제 (총 최대 15행)

### 코사인 유사도 피쳐 셋 (8개)
```
danceability, energy, speechiness, acousticness,
instrumentalness, liveness, valence, tempo (정규화 후)
```
`loudness` 제외: energy와 상관 0.677로 중복 신호. situation 하드 필터로만 사용.

### 제외 컬럼
- `instrument`: 84.1% Unknown, 정보량 없음
- ID 컬럼: `track_album_id`, `playlist_id`, `track_id` (dedup 후 인덱스 재설정)
- `key`, `mode`: 음악 이론적 속성이나 현재 매핑 테이블에서 미사용. 제외.

---

## 7. Key Findings & Design Implications

### PS-03 전처리 확정 사항
1. `track_id` 기준 dedup, popularity max 유지 → 28,356행
2. `loudness`, `tempo` min-max 정규화
3. 결측 15행 삭제
4. 코사인 피쳐 8개 컬럼으로 정리된 `feature_matrix` 생성

### PS-05 추천 엔진 수정 사항
1. **`수면` loudness 범위 완화**: `(-46, -15)` → `(-46, -10)` (CLAUDE.md 업데이트 필요)
2. **Mental_Health_Label 사용 방침 — 소프트 신호 / fallback만 허용**:
   - `Normal/Mania/Anxiety` 3개 라벨은 **하드 게이트(선필터)가 아닌** 코사인 유사도 스코어링 단계의 보조 신호 또는 후처리 fallback으로만 활용.
   - 이유: 라벨 선필터 적용 시 `코딩` 381→284, `수면` 234→109로 급감 — 둘 다 no-result 경로 진입 위험 (섹션 5 post_label_filter 컬럼 참고).
   - 나머지 라벨(Depression, Bipolar Depression 등)은 샘플 부족으로 어떤 방식으로도 활용 불가.
3. **`excited` mood**: Bipolar (Mania) 라벨이 거의 완벽한 신호(98.9%) — 소프트 신호로서는 강력히 활용 가능. 단, 하드 게이트가 아닌 스코어링 가중치로 반영.
4. **`angry` genre 다양성**: 추천 결과가 edm에 쏠릴 수 있음. genre_pref 파라미터로 사용자가 원하는 장르 지정 가능하면 보완됨.

### 발표 어필 포인트
- "Mental_Health_Label은 내부 검색 신호로만 사용하고 사용자에게 노출하지 않는다" — 이 원칙이 실제 데이터에서도 정당화됨 (`excited`–Mania 98.9%)
- 데이터 기반으로 매핑 수치를 검증하고 조정했다 (`수면` 범위 수정)
- loudness 제외 근거를 상관분석으로 설명 가능
