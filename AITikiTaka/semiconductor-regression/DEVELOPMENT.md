# Local app implementation

## Boundaries

- Streamlit orchestrates UI; `lab/` owns extraction, persistence and ML operations.
- `src/models.py` and `src/evaluation.py` are reused. All three original estimators remain.
- Each persisted app model is a complete scikit-learn pipeline, independent of the legacy CLI artifact format.
- SQLite project-scopes all record reads. Uploaded filenames are metadata; generated IDs determine filesystem paths.
- Files, dataset snapshots and model versions are append-only. Activating a model changes only a pointer.
- The local Ollama client bypasses proxies and redirects; rejects remote model metadata and cloud model names. The launcher starts Ollama with `OLLAMA_NO_CLOUD=1`.

## Data policies

Inputs and both measured targets must be finite numeric values after explicit unit conversion. Invalid rows remain in the stored import report but are excluded from training. No target imputation. Formula cells are not executed. Excel percentage number formats are preserved as metadata to distinguish 0.98 from 98.

The user chooses the sheet, range and optional Lot/time columns. LLM output is a proposal: locations are validated against the sheet, unit suggestions must appear in the heading (or be derived from Excel percentage format), and the visible explanation is grounded in original cells. The final user confirmation is mandatory. The first release does not autonomously discover arbitrary tables across all sheets.

Training uses one selected imported dataset plus optional measured results from that project. Duplicate content and common Lot IDs are connected into indivisible split components. Complete timestamp data triggers chronological splitting; overlapping Lot times fail explicitly. Without timestamps, grouping is deterministic with seed 42. Without Lot metadata, identical content still cannot cross partitions.

Model selection uses mean MAE normalized by user-configured tolerance, on validation only. A mean-prediction baseline is shown. The winner is evaluated on test data, without refitting. Activation requires validation tolerance compliance and baseline improvement. Test metrics remain visible for review.

Retraining resplits the current snapshot; it does not claim an immutable prospective holdout. Existing-model comparison is omitted if the new test set overlaps the incumbent's training/validation content or Lot groups. Repeated reuse of evaluation data is not proof of deployment performance; measured outcomes provide the forward monitoring channel.

## Test commands

From the workspace root:

```sh
.venv/bin/python -m pytest AITikiTaka/semiconductor-regression/tests -q
```

From this app directory, with local Ollama running and Qwen3 4B installed:

```sh
../../.venv/bin/python scripts/validate_local_llm.py
```

The latter sends only synthetic fixtures to localhost and writes detailed results to ignored `local_data/verification/local_llm.json`. It checks all six column positions and the header row for English, Korean-report and reordered/unit-converted workbooks. Unit extraction is not guaranteed for every arbitrary workbook; absent units require manual confirmation.

Tests cover full train/save/load/predict/measure/retrain lifecycle, cross-project isolation, time/group/duplicate split separation, unit conversion, missing targets, Excel percentage formats, LLM schema errors, and Streamlit user flows (including upload mapping and measurement submission).

## Operational limits

- Single-user local prototype. No authentication, disk encryption or multi-user coordination layer.
- Training runs synchronously with progress. Multi-browser simultaneous training and background scheduling are outside the first release.
- Retraining reminders are evaluated on overview visits; there is no daemon or automatic model replacement.
- Actual results can be entered through the prediction-linked form. Bulk Excel actual-result matching is not implemented.
- Demo workbooks are alternate representations of the same synthetic data. They must not be treated as independent experiments.
- Browser/native UI automation was unavailable in this development session. Streamlit AppTest verified widget flows; visual browser rendering still needs a manual check on the user's screen.


## 확정한 프로젝트 범위 (2026-11-09 대화 기준)

- 현장에서 쓰는 자료를 일부 수정해 연구실을 통해 제공받았다. 원자료 수정 범위와 출력값 보존 여부는 미확인이다.
- 제공 자료의 Si₂H₆/N₂O/N₂ 조합은 고정이며 설정값은 임의로 정한 값이다. 업계 최다 사용 레시피라는 주장은 사용하지 않는다.
- 현재 다섯 입력 모델은 보조가스 N₂O 4000 sccm, N₂ 3000 sccm 고정이라는 제공 조건표의 범위로 사용한다. 다른 보조가스 설정을 숫자 입력 다섯 개만으로 구분할 수 없다.
- 업로드 승인에서 사용자가 가스 조합과 고정 유량을 확인해야 한다. 자동 가스 검증 기능은 아니다. 원본은 변경하지 않는다.
- 출처 구분은 unknown/modified_field/measured/synthetic으로 데이터셋과 모델에 저장한다. 기존 기록은 합성이 아닌 경우에도 출처 미확인으로 표시한다.
- 로컬 LLM은 파일 의미 해석, 프로그램은 행 처리와 검증, 회귀 모델은 두께/반범위 불균일도 예측을 맡는다.
- 실제 장비 실험 전 후보를 비교하여 준비 부담과 반복 실험을 줄이는 것이 목표다. 비용/실험 감소 및 현장 정확도 개선은 검증 전 성과로 주장하지 않는다.
- 회사별 데이터·장비/챔버 확장은 후속 검증 대상이다. 로컬 실행은 외부 API 전송 회피를 위한 설계이며 보안 인증이나 검증 완료를 뜻하지 않는다.
- PTEOS 표기와 Si2H6 표기의 불일치는 제공자 확인 전 미해결로 남긴다.

## 가상 실험 직접 입력 (2026-11-09)

- 가상 실험 파일 업로드를 RF 전력·압력·온도·Si₂H₆ 유량·증착 시간 다섯 개의 숫자 입력으로 교체했다.
- 기존 다섯 입력 회귀 모델을 유지하며, 증착 시간(s)도 사용자가 직접 입력한다. 다섯 입력의 초기값은 학습 구간 중앙값이며 모두 수정할 수 있다.
- 입력을 수정하면 이전 예측을 지워 오래된 결과가 저장되지 않게 한다. 평균 두께·반범위 불균일도, BKM 비교와 5% 실험 목표를 표시한다.
