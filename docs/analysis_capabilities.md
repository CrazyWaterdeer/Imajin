# Imajin 분석 능력 매트릭스 (사용자 가이드)

imajin이 **무엇을**(분석 종류) · **어디에**(대상) · **어떻게**(도구) 수행하고, 각 분석에
어떤 **통계**와 **그래프**가 붙는지 정리한 참조 문서입니다. 챗독에서 자연어로 요청하든
수동 dock에서 버튼으로 하든 같은 도구가 실행됩니다.

> 표기: `도구이름`은 챗 에이전트가 호출하는 tool 함수입니다. "대상"은 그 분석이 실제로
> 소비하는 napari 레이어/채널/표를 뜻합니다.

## 능력 매트릭스

| 분석 종류 | 대상 | 주요 도구 (방식) | 통계 | 그래프 |
|---|---|---|---|---|
| 파일 로드 · 메타데이터 | `.lsm` / `.czi` / OME-TIFF | `load_file`, `reload_file`, `advance_to_file`(순차 언로드) | — | — |
| 채널 주석 · 해석 | image 레이어 | `annotate_channel`, `resolve_target_channel`, `detect_counterstain_channel` | — | — |
| 전처리 | target 채널 | `rolling_ball_background`, `auto_contrast`, `gaussian_denoise` | — | — |
| 세포/객체 분할 | target 채널 | `segment_target_objects`, `auto_segment_target`, `segment_3d_cells_auto`, `cellpose_sam`, `analyze_target_cells`(원샷 분할+측정) | — | QC 오버레이 (`compute_segmentation_qc`) |
| 발현 도메인 분할 | 채널 강도 | `segment_intensity_regions`, `segment_expression_domain` | — | — |
| ROI 제한 분할 | Shapes + 참조 레이어 | `boundary_mask_from_shapes` → `boundary_mask=` 로 전달 | — | — |
| **강도 측정 (객체별)** | labels × 채널 | `measure_intensity`, `measure_projected_intensity`(투영 후 측정), `refresh_measurement` | `describe_table`, `compare_groups` | `plot_group_distribution` |
| **시계열 강도 (ROI over time)** | ROI labels × T | `measure_intensity_over_time`, `extract_timepoint` | `normalize_timecourse`, `extract_timecourse_features` | `plot_timecourse` |
| 콜로컬라이제이션 | 채널 쌍 | `manders_coefficients`(M1/M2), `pearson_correlation` | — | `plot_scatter` |
| **inside / outside 도메인** | 채널 마스크 | `mask_logic`, `partition_inside_outside`, `classify_labels_by_mask` | `compare_groups`(**paired** wilcoxon) | `plot_group_distribution`(`paired=True`) |
| 칼슘 이미징 | ROI × T movie | `assess_calcium_timecourse`(ΔF/F0+게이팅), `correct_calcium_motion`, `stabilize_calcium_dense` | — | `plot_dff_heatmap` |
| 세포 추적 | T축 labels | `track_cells` (btrack) | — | — |
| 신경 형태 (고급, 별도 워크플로) | 스켈레톤 / 트레이스 | `enhance_neural_processes`, `skeletonize`, `prune_skeleton`, `compute_sholl_analysis`, `extract_branch_metrics`, `classify_neuron_type`, `find_similar_neurons`, `export_neural_trace` | 형태 지표 | Sholl 등 전용 |
| **표 정리 · 풀링** | 세션 테이블 | `combine_tables`(파일별 표 합치기+`sample_name` 태깅), `import_table`(외부 CSV→세션), `coalesce_columns`(sparse 강도컬럼 통합), `map_column`(그룹 부여), `select_representative_rows`(주 영역만), `filter_table`, `summarize_table`, `export_table` | — | — |
| **통계** | 측정 표 | `describe_table`, `compare_groups`, `summarize_experiment` | (본체) | `plot_group_distribution` |
| QC · ROI 리뷰 | labels / 표 / 타임코스 | `compute_segmentation_qc`, `compute_measurement_qc`, `compute_timecourse_qc`, `mark_qc_status`, `review_target_roi`, `jump_to_object` | — | 라벨 아웃라인 |
| 배치 · 실험 | 파일 그룹 | `register_files`, `annotate_samples`, `create_analysis_recipe`, `run_recipe_on_samples`, `get_batch_progress`, `plan_resume` → `open_result_bundle`(재개) | `summarize_experiment` | 실험 리포트 |
| 리포트 | 세션 / 실험 | `save_result_bundle`(결과 한 폴더로), `generate_report`, `generate_experiment_report`, `generate_methods` | — | — |
| 시각화 · 3D | 레이어 | `set_view`, `set_colormap`, `max_projection`, `average_projection`, `orthogonal_views`, `animate_z_rotation`, `export_channel_composite_png`, `screenshot` | — | — |

## 그래프 옵션 상세

### `plot_group_distribution` — 그룹 비교 (가장 옵션이 많음)
| 옵션 | 값 | 설명 |
|---|---|---|
| `kind` | `box`(기본) / `bar` / `violin` / `dots` | `dots` = 모든 점 + 평균±SEM crossbar (**소표본 정석**); `bar` = 평균+SEM 막대 |
| `paired` | `True` / `False` | 같은 `sample_name`이 두 그룹에 있을 때 **샘플별 연결선** (inside/outside, 전/후) |
| `show_posthoc` | `True`(기본) | 3그룹↑에서 **보정된 사후검정 유의성 bracket** (Games-Howell / Dunn+Holm) 자동 표시 |
| `weight_col` | `auto`(기본) / `None` / 컬럼명 | 객체→샘플 집계 시 **면적가중**(area 있으면 total/total). `None`=무가중 |
| `level` | `auto` / `sample` / `object` | 샘플단위 집계 vs 객체단위 |
| `palette` | `["#..","#.."]` | 그룹 색 지정 |
| `ymin` / `ymax` / `log_y` / `zero_baseline` | | y축 범위 · 로그 · 0부터 시작 |
| `point_size` / `jitter` / `show_points` | | 점 크기 · 흩뿌림 폭 · 점 표시 여부 |
| `show_n` / `show_stats` / `stats_test` | | n 라벨 · 유의성 표시 · 검정 종류 |
| `format` / `title` / `ylabel` / `width` / `height` / `dpi` | | svg(기본)/pdf/png, 크기 |

### 그 외 플롯
- **`plot_timecourse`** — 평균선 + 구간(`interval`: `sem`/`ci95`/`none`), 개별 트레이스(`show_individual`, `max_individual_traces`).
- **`plot_scatter`** — 두 수치 컬럼 산점도, 그룹 색(`group_col`), 로그(`log10`), 회귀선(`fit_line`).
- **`plot_dff_heatmap`** — 칼슘 ΔF/F0 래스터 히트맵.
- **`export_channel_composite_png`** — 다채널 RGB 합성 이미지(채널별 max/mean 투영, 역할별 컬러맵, 스케일바).

## 통계 선택 가이드

`compare_groups`가 핵심입니다. `test="auto"`(기본)는 **가정을 스스로 점검**합니다.

- **모수 vs 비모수**: Shapiro-Wilk 정규성 검정 → 정규면 **Welch**(2그룹)/**ANOVA**(3+), 비정규면 **Mann-Whitney**/**Kruskal**. 2·3그룹 철학 일관.
- **paired(대응) 설계**: 같은 개체를 두 조건에서 측정(inside/outside, 전/후, 짝지음)했다면 **직접 `test="wilcoxon"`(또는 `"paired_t"`) 지정**. auto는 대응 구조를 감지하면 경고만 하고 자동 전환은 안 함(대응 여부는 실험 설계 주장이므로).
- **면적가중**: 객체별 강도를 샘플로 합칠 때 기본이 **면적가중**(`weight_col="auto"`, area 있으면 total signal/total area). 작은 debris가 개수만큼 표를 행사하지 않음. per-cell처럼 각 객체가 독립 관측치면 `weight_col=None`.
- **3그룹↑ 사후검정**: omnibus에 더해 **다중비교 보정된 쌍별 검정**을 `posthoc`로 반환(ANOVA→Games-Howell, Kruskal→Dunn's+Holm). 보정 없는 쌍별 비교를 직접 돌리지 말 것.
- **주의**: 결과의 `warnings` / `test_selection`을 항상 확인 — 소표본(n<3), 비정규, **pseudoreplication**(세포를 독립 표본 취급) 경고가 여기 담김.

## 대표 워크플로

**① 단일 이미지 세포 측정**
`load_file` → (필요시 `resolve_target_channel`) → `segment_target_objects` → `measure_intensity` → `describe_table` / `plot_group_distribution`.

**② 다중 파일 그룹 비교 (풀링)**
파일별로 분할·측정 → `combine_tables`(sample_name 태깅) → `map_column`(sample→group) → `coalesce_columns`(강도 컬럼 통합) → (선택) `select_representative_rows`(주 영역만) → `compare_groups` → `plot_group_distribution`.
외부에서 합친 CSV는 `import_table`로 되불러오면 동일하게 이어집니다.

**③ inside / outside 도메인**
`segment_intensity_regions("green")` → `partition_inside_outside(green, specimen)` → `measure_intensity(partition, ["red"])` → `compare_groups(group_col="region", test="wilcoxon")` → `plot_group_distribution(paired=True)`.

**④ 시계열 / 칼슘**
`measure_intensity_over_time(ROI, movie)` → `normalize_timecourse` / `extract_timecourse_features` → `plot_timecourse`. 칼슘은 `assess_calcium_timecourse` → `plot_dff_heatmap`.

**⑤ 순차 다중 파일 (한 폴더에 수집)**
`start_analysis(<이름>)` 한 번 호출 → 파일별 분석 → 각 `save_result_bundle`가 그 폴더에 append(파일 사이에 `finalize_analysis` 호출 금지) → 마지막에 `finalize_analysis`.

---
*이 문서는 코드의 실제 등록 도구(110개)를 기준으로 작성되었습니다. 새 도구/옵션이 추가되면 함께 갱신하세요.*
