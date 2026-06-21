# -*- coding: utf-8 -*-
"""preprocessing_project — 과거 전처리 작업 전수를 버전 폴더로 정리 + 데이터셋 백업.
각 작업 → preprocessing_project/<ver>/{output(데이터 백업), src(재현 소스 포인터), README.md}.
대용량(>CAP MB)은 복사 대신 DATASET_BACKUP.md 포인터(경로·크기). `--all` 이면 전부 복사.
실행: python preprocessing_project/backup_datasets.py [--all]
"""
import os, sys, shutil, json

HERE = os.path.dirname(os.path.abspath(__file__))            # preprocessing_project
ROOT = os.path.dirname(HERE)                                  # team_project_churn
CAP_MB = 40
ALL = "--all" in sys.argv

# ver : (source_dir, kind, x_desc, y_desc, reproducer)
WORKS = {
    "legacy_onlineretail":   ("retail/data", "온라인리테일(UCI) 이탈", "tabular.csv(RFM) + seq.npz([N,T,F])", "split.npz(train/test) 라벨", "retail/*.py (analyze/label_features/train_*)"),
    "legacy_rees46_cosmetics":("retail_rees46/data", "REES46 화장품 단월 이탈", "tabular.csv + seq.npz", "split.npz", "retail_rees46/prep_rees46.py 등"),
    "legacy_rees46_multi":   ("retail_rees46_multi/data", "REES46 다개월 시퀀스 이탈", "tabular.csv + seq.npz(다개월)", "split.npz", "retail_rees46_multi/prep_multi.py 등"),
    "legacy_rees46_labels":  ("retail_rees46_labels/data", "REES46 라벨변형(base/A/B/C)", "라벨별 데이터셋", "라벨 설계별 churn", "retail_rees46_labels/prep_labels.py"),
    "legacy_mobile_game":    ("mobile_game_labels/data", "모바일게임 레벨시퀀스 이탈", "레벨 시퀀스 + 집계", "이탈/회귀 라벨", "mobile_game_labels/prep_game.py 등"),
    "aux_fullsession":       ("sample_project/data/processed_full", "풀 세션-레벨 데이터셋", "session_level_full.npz([N,T,F]) + recommend_user_interest.parquet", "session_level_full_meta.json", "scripts/save_full_windows.py, session_full_1734.py"),
    "aux_nextcat":           ("sample_project/data/processed_nextcat", "다음카테고리 예측 데이터셋", "nextcat_dataset(.fullwin).npz", "다음 카테고리 멀티클래스", "scripts/nextcat_1735.py, save_full_windows.py"),
    "aux_rec":               ("sample_project/data/processed_rec", "추천 전용(user×item) 데이터셋", "rec_user_interest_5m / rec_user_top_categories / rec_item_popularity .parquet", "추천(무라벨)", "preprocessing_project/v2_5m/src/rec_extract_5m.py"),
}


def dirsize_mb(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t / 1e6


def main():
    index = []
    for ver, (src, kind, xd, yd, repro) in WORKS.items():
        srcp = os.path.join(ROOT, src)
        outp = os.path.join(HERE, ver, "output")
        os.makedirs(os.path.join(HERE, ver, "src"), exist_ok=True)
        os.makedirs(outp, exist_ok=True)
        exists = os.path.isdir(srcp)
        mb = dirsize_mb(srcp) if exists else 0
        copied = False
        if exists and (ALL or mb <= CAP_MB):
            for f in os.listdir(srcp):
                sp = os.path.join(srcp, f)
                if os.path.isfile(sp):
                    shutil.copy2(sp, os.path.join(outp, f))
            copied = True
        # DATASET_BACKUP.md
        with open(os.path.join(HERE, ver, "output", "DATASET_BACKUP.md"), "w", encoding="utf-8") as f:
            f.write(f"# {ver} 데이터셋 백업\n\n- 원본 위치: `{src}` ({'존재' if exists else '없음'}, {mb:.1f} MB)\n")
            f.write(f"- 백업 상태: {'복사됨(output/)' if copied else ('대용량→포인터(원본 참조)' if exists else '원본 없음')}\n")
            if not copied and exists:
                f.write(f"- 복사하려면: `python preprocessing_project/backup_datasets.py --all`\n")
        # README.md
        with open(os.path.join(HERE, ver, "README.md"), "w", encoding="utf-8") as f:
            f.write(f"# {ver} — {kind}\n\n전처리 버전 관리 항목(과거 작업 백업).\n\n"
                    f"## 구성\n- **종류**: {kind}\n- **X 구성**: {xd}\n- **Y 구성**: {yd}\n"
                    f"- **재현 소스**: `{repro}` (원본 위치 유지)\n- **원본 데이터**: `{src}` ({mb:.1f} MB)\n"
                    f"- **백업**: `output/` ({'복사됨' if copied else '포인터'}) — 상세 `output/DATASET_BACKUP.md`\n\n"
                    f"> 대용량은 git 비대화 방지를 위해 기본 포인터. `--all`로 물리 복사 가능.\n")
        index.append((ver, kind, f"{mb:.1f}MB", "복사" if copied else ("포인터" if exists else "없음")))
        print(f"  {ver:26s} {mb:7.1f}MB  {'COPIED' if copied else ('POINTER' if exists else 'MISSING')}")

    # 마스터 인덱스
    with open(os.path.join(HERE, "DATASET_INDEX.md"), "w", encoding="utf-8") as f:
        f.write("# preprocessing_project 데이터셋 인덱스 (전 버전)\n\n")
        f.write("| 버전 | 종류 | 원본크기 | 백업 |\n| --- | --- | --- | --- |\n")
        f.write("| v1_daily | 배포본 일별14 | (sample_project) | 정본 위치 |\n")
        f.write("| v2_5m | 5개월 10피처·7모델 | 96MB | output/OUTPUT_INDEX.md |\n")
        f.write("| v3_event_additive | 이벤트시퀀스 추가형 | (processed_eventseq) | output/ |\n")
        for ver, kind, sz, st in index:
            f.write(f"| {ver} | {kind} | {sz} | {st} |\n")
        f.write(f"\n> CAP={CAP_MB}MB 초과는 기본 포인터. 전체 물리 백업: `python preprocessing_project/backup_datasets.py --all`\n")
    print(f"\n[backup] 인덱스 → preprocessing_project/DATASET_INDEX.md (ALL={ALL})")


if __name__ == "__main__":
    main()
