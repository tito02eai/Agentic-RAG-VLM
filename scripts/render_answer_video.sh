#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/output/complete_task_demo_v2_seed101"
RAW_VIDEO="${RUN_DIR}/guanghua_complete_task_raw.mp4"
SUBTITLES="${PROJECT_DIR}/video/complete_task_success/complete_task_success.ass"
FINAL_VIDEO="${RUN_DIR}/guanghua_agentic_rag_vlm_complete_success.mp4"

if [[ ! -f "${RAW_VIDEO}" ]]; then
  echo "Missing raw rollout: ${RAW_VIDEO}" >&2
  echo "Run MUJOCO_GL=egl python scripts/render_complete_task_demo.py first." >&2
  exit 1
fi

ffmpeg -y -hide_banner -loglevel warning \
  -i "${RAW_VIDEO}" \
  -vf "subtitles=${SUBTITLES}:fontsdir=/usr/share/fonts" \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -movflags +faststart \
  -metadata title="Guanghua Agentic RAG-VLM Complete Task Success" \
  -metadata comment="Qwen3.5-4B NF4 seed 101; kinematic grasp-skill proxy; not a contact-dynamics claim" \
  -an "${FINAL_VIDEO}"

ffprobe -v error \
  -show_entries format=duration,size:stream=codec_name,width,height,r_frame_rate,pix_fmt \
  -of json "${FINAL_VIDEO}"

echo "Answer video: ${FINAL_VIDEO}"
