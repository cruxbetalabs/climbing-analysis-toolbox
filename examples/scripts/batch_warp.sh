#!/bin/bash

DIR="/Users/tommyjtl/Documents/Projects/climbing/climbs/apr-15"
REF_IMG="$DIR/warped.png"

shopt -s nullglob

for video in "$DIR"/*.MOV; do
  echo "Processing: $(basename "$video")"

  python warp_video_demo.py \
    --src_video_path "$video" \
    --ref_img "$REF_IMG"

  echo "Done: $(basename "$video")"
  echo
done

