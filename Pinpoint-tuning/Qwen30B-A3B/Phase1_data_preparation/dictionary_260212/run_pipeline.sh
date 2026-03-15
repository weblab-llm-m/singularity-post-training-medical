#!/bin/bash
# =============================================================
# dictionary_260212: 拡張辞書パイプライン
# Step 1: 辞書生成 (5 seeds × 30 samples × 2並列, GPU 0-3 / 4-7)
# Step 2: 辞書マージ (old 5 + new 5)
# Step 3: replacement_mapping 生成 (2並列, GPU 0-3 / 4-7)
# Step 4: アノテーション
# Step 5: 反実データ生成
# Step 6: Path Patchingデータ構築
# =============================================================
set -e

VENV="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B/venv/bin/python3"
BASE="/home/yuuki.nakamura/singularity-post-training-medical/Pinpoint-tuning/Qwen30B-A3B/Phase1_data_preparation"
DICT_DIR="${BASE}/dictionary_260212"

echo "============================================================"
echo "Step 1: Generate expanded dictionaries (3+2 seeds × 30 samples, 2 parallel)"
echo "============================================================"

# Group A: seeds 2000,3500,4000 on GPU 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 ${VENV} ${DICT_DIR}/generate_expanded_dict.py \
    --seeds 2000 3500 4000 --num_samples 30 \
    > ${DICT_DIR}/gen_groupA.log 2>&1 &
PID_A=$!
echo "  Group A (seeds 2000,3500,4000, GPU 0-3) started: PID=${PID_A}"

# Group B: seeds 4500,5000 on GPU 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 ${VENV} ${DICT_DIR}/generate_expanded_dict.py \
    --seeds 4500 5000 --num_samples 30 \
    > ${DICT_DIR}/gen_groupB.log 2>&1 &
PID_B=$!
echo "  Group B (seeds 4500,5000, GPU 4-7) started: PID=${PID_B}"

echo "  Waiting for both groups to finish..."
FAIL=0
wait ${PID_A} || FAIL=1
if [ ${FAIL} -eq 1 ]; then
    echo "  ERROR: Group A failed! See gen_groupA.log"
    cat ${DICT_DIR}/gen_groupA.log
    exit 1
fi
echo "  Group A done."

wait ${PID_B} || FAIL=1
if [ ${FAIL} -eq 1 ]; then
    echo "  ERROR: Group B failed! See gen_groupB.log"
    cat ${DICT_DIR}/gen_groupB.log
    exit 1
fi
echo "  Group B done."
echo "  Step 1 complete!"

echo ""
echo "============================================================"
echo "Step 2: Merge all dictionaries (old 5 + new 5)"
echo "============================================================"
${VENV} ${DICT_DIR}/merge_all_dicts.py

echo ""
echo "============================================================"
echo "Step 3: Generate replacement mapping for new terms"
echo "============================================================"
CUDA_VISIBLE_DEVICES=0,1,2,3 ${VENV} ${DICT_DIR}/generate_replacement_mapping.py

echo ""
echo "============================================================"
echo "Step 4: Annotate igakuQA with expanded dictionary"
echo "============================================================"
${VENV} ${BASE}/medical_term_annotator.py \
    --medical_dict ${DICT_DIR}/medical_terms_dictionary.json \
    --output_file ${DICT_DIR}/annotated_medical_data_full.jsonl

echo ""
echo "============================================================"
echo "Step 5: Generate counterfactual data (overlap)"
echo "============================================================"
cd ${DICT_DIR}
${VENV} ${DICT_DIR}/generate_overlap_counterfactuals.py

echo ""
echo "============================================================"
echo "Step 6: Build Path Patching data"
echo "============================================================"
${VENV} ${BASE}/path_patching_data_builder.py \
    --annotation_path ${DICT_DIR}/annotated_medical_data_full.jsonl \
    --counterfactual_path ${DICT_DIR}/counterfactual_overlap_medical.jsonl \
    --output_path ${DICT_DIR}/path_patching_data_overlap_medical.jsonl

${VENV} ${BASE}/path_patching_data_builder.py \
    --annotation_path ${DICT_DIR}/annotated_medical_data_full.jsonl \
    --counterfactual_path ${DICT_DIR}/counterfactual_overlap_reasoning.jsonl \
    --output_path ${DICT_DIR}/path_patching_data_overlap_reasoning.jsonl

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo "Output files in: ${DICT_DIR}/"
echo "  medical_terms_dictionary.json"
echo "  replacement_mapping.json"
echo "  annotated_medical_data_full.jsonl"
echo "  counterfactual_overlap_medical.jsonl"
echo "  counterfactual_overlap_reasoning.jsonl"
echo "  path_patching_data_overlap_medical.jsonl"
echo "  path_patching_data_overlap_reasoning.jsonl"
