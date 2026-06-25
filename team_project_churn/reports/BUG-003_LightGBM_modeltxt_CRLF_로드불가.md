# BUG-003 — LightGBM `model.txt` 로드 불가 (CRLF 개행 → "expect a tree here")

| 항목 | 값 |
| --- | --- |
| **ID** | BUG-003 |
| **제목** | `models/churn/lightgbm/model.txt`가 LightGBM 네이티브 로드 시 `Model format error, expect a tree here` 치명오류 |
| **심각도** | Medium (해당 파일 추론 불가. 단 **현 서빙 경로 미사용**이라 운영 영향 없음 — 잠복 지뢰) |
| **상태** | ✅ Resolved (2026-06-22) — 방식 1(CRLF→LF) 적용·검증 완료 |
| **영역** | 모델 산출물 `models/churn/lightgbm/` (모델팀 export) · 백엔드는 무관 |
| **보고일** | 2026-06-22 (모델 파일 점검 중 제보) |
| **유입 시점** | LightGBM `model.txt` export 단계(Windows 텍스트모드 저장) |
| **관련 문서** | [19-3](19-3_모델팀_산출물_및_I_O_계약서.md) §5(산출물), [BUG-001](BUG-001_얼굴인증_계약불일치.md)(형식) |

---

## 1. 요약 (TL;DR)
`model.txt`가 **CRLF(`\r\n`) 개행**으로 저장돼 있어 LightGBM C++ 네이티브 파서(LF 전용)가 토큰 경계를 깨뜨려 `Model format error, expect a tree here`를 던진다(멀티스레드라 `[Fatal]` 폭주 + Python 프로세스 abort). **파일 단독 결함**이며 **서버 폴백과 무관**하다 — 백엔드는 `model.txt`를 전혀 로드하지 않고 `prep_LightGBM_v2.joblib`(+정상 `model.pkl`)로 서빙한다. 조치: model.pkl 부스터로 **LF 재익스포트**해 교체하거나 model.txt **삭제**.

## 2. 증상 (What)
```
[LightGBM] [Fatal] Model format error, expect a tree here. met ...
```
- `lgb.Booster(model_file="models/churn/lightgbm/model.txt")` 호출 시 위 오류가 **수십~수백 줄 폭주**, 이어 **프로세스가 예외가 아니라 abort로 종료**(Python try/except로도 못 잡고 결과 파일 기록 전에 죽음).

## 3. 발생 위치 (Where)
- 파일: `SKN32-2nd_GAJIMA_Dev/models/churn/lightgbm/model.txt` (351,630 B, 2,062 lines)
- 트리거: LightGBM 네이티브 텍스트 로더(`Booster(model_file=...)`).

## 4. 근본 원인 (Why) — 결정적 증거
- `cmp model.txt (정상 LF 재익스포트)` → **char 5, line 1에서 분기**.
- `od -c` 1행 비교:
  - **깨진 model.txt**: `t r e e \r \n` ← **CRLF**
  - 정상(model.pkl 부스터 재익스포트): `t r e e \n` ← LF
- 크기 차이: 정상 349,570 B vs 깨짐 351,630 B = **+2,060 B ≈ 2,060행 × `\r` 1바이트** → **모든 줄에 `\r` 삽입**.
- LightGBM 네이티브 파서는 LF 기준 토큰 분해를 하는데, 각 줄 끝 `\r`가 `Tree=N` 등 헤더 인식을 깨 **"트리가 와야 할 자리"를 못 찾음** → `expect a tree here`.

### 4.1 CRLF는 어디서 왔나 (git autocrlf 아님)
- 본 머신 `core.autocrlf=true`지만, **`model.txt`는 git 미추적**(`did not match any file(s) known to git`) → **git 체크아웃 변환이 원인이 아님**.
- 즉 **export 단계에서 CRLF로 저장**된 것(Windows에서 `open(path,'w')` 텍스트모드 저장 또는 텍스트 에디터 경유 시 LF→CRLF 변환). model.pkl/model_config.json은 정상.
- ⚠ 향후 이 파일을 **커밋하면** `.gitattributes` 없이 `autocrlf=true` 환경에서 CRLF가 재유입될 수 있음 → 예방 필요(§9).

## 5. 영향 (Impact)
- `model.txt`로의 LightGBM 네이티브 로드 **불가**. 게다가 단순 에러가 아니라 **호출 프로세스를 죽일 수 있음**.
- **현재 운영 영향 없음**: 아래 §6대로 서버는 이 파일을 사용하지 않음.

## 6. 서버 폴백과 겹친 건가? — **아니오(다른 문제)**
- 백엔드 전체 grep(`model.txt`/`model_file`/`Booster(`/`.txt'`) → **0건**. 백엔드는 `model.txt`를 **어떤 경로로도 로드하지 않음**(폴백 포함).
- 실시간 추론은 `infrastructure/model_inference/python_model_loader.py`가 **`models/preprocessors/prep_LightGBM_v2.joblib`** 번들을 로드(존재 확인). `model.pkl`도 정상(LGBMClassifier, 100 trees).
- 결론: **서버 폴백 충돌이 아니라, model.txt라는 산출물 자체가 잘못 만들어진 단독 결함**. 서버는 정상 경로(prep 번들)로 동작.

## 7. 진단/해결 (How / Method / When) — 2026-06-22
1. 헤더·구조 검사: `Tree=` 100개, `end of trees` 존재 → 구조는 정상으로 보임(육안).
2. 로드 재현 → `[Fatal] expect a tree here` + 프로세스 abort 확인.
3. `model.pkl` 로드 정상(LGBMClassifier, 22 feat, 100 trees, `predict_proba(zeros)=[0.239, 0.761]`).
4. pkl 부스터 → `save_model()`로 **LF 재익스포트(349,570 B)** → `Booster(model_file=...)` **재로드 성공(100 trees)**.
5. `cmp`/`od -c`로 **CRLF가 유일 차이**임을 확정.

### 7.1 적용된 조치 — 방식 1 (CRLF→LF, 그 파일 그대로 수정·재활용)
**2026-06-22 적용 완료.** 모델은 정상이고 개행만 결함이므로, `model.txt`의 `\r`만 제거(`b"\r\n"→b"\n"`)해 LF로 정상화. 새 모델을 만들지 않고 **기존 모델을 그대로 재활용**.
- 적용 전 원본은 `model.txt.crlf.bak`로 백업 후 검증 완료, 이후 백업 제거.
- 대안(미채택): (A') `model.pkl.booster_.save_model()`로 재익스포트 / (B) 삭제. 둘 다 동일 결과지만 방식 1이 "그 파일 수정·재활용" 의도에 부합.
- **예방 동시 적용**: 레포 루트 `.gitattributes` 신설 — `models/**/model.txt -text`, `*.txt -text`, 바이너리 산출물 `binary` 지정 → `autocrlf=true` 환경에서 CRLF 재유입 차단(`git check-attr text` = `unset` 확인).

## 8. 검증 (Verification)
| 케이스 | 결과 |
| --- | --- |
| `Booster(model_file=model.txt)` (수정 전) | ❌ `[Fatal] expect a tree here` + abort |
| `joblib.load(model.pkl)` | ✅ LGBMClassifier, 100 trees, proba 0.761 |
| **CRLF→LF 수정 적용** | ✅ CR 2,062개 제거(351,630→349,568B), 첫 줄 `tree\n` |
| **수정 후 `Booster(model_file=model.txt)`** | ✅ **native_load OK, 100 trees, 22 feat** |
| **수정 txt vs model.pkl 예측 일치** | ✅ **max_abs_diff = 0.000e+00 (동일 모델)** |
| `.gitattributes` text 속성 | ✅ `unset`(autocrlf 변환 차단) |
| 백엔드 `model.txt` 로드 코드 | ✅ 없음(서빙 무관) |
| `prep_LightGBM_v2.joblib` 서빙 아티팩트 | ✅ 존재 |

## 9. 재발 방지 (Prevention)
- **모델 txt export는 LF로**: `booster.save_model(path)` 사용(LightGBM 기본 LF). Python `open()`으로 직접 쓸 때는 `open(path, "w", newline="\n")`로 텍스트모드 변환 차단.
- **`.gitattributes` 추가**(이 산출물들을 커밋 공유할 예정이므로 필수):
  ```
  models/**/model.txt -text
  *.txt -text
  ```
  → `core.autocrlf=true` 환경에서도 CRLF 재유입 방지.
- 모델 로드 헬스체크: 산출물 수령 시 `Booster(model_file=...)` 스모크로 **개행/포맷을 사전 검증**(이번처럼 abort 위험이 있으니 서브프로세스 격리 권장).

## 10. 잔여 (Remaining)
- ~~조치 적용~~ ✅ 완료(방식 1) + `.gitattributes` 신설. 산출물 공유 커밋 시 `.gitattributes`도 함께 커밋(가지마 feature/backend, 명령 시).
- 타 모델(CatBoost/XGBoost 등)은 native `model.txt` 미보유라 동일 이슈 없음(파일 형식: cbm/json/joblib/pt). XGBoost `model.json`·Transformer `model.pt`는 텍스트 개행 영향 없음.
