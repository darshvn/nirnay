#!/bin/bash
# The skill's join.sh at 60 fps, normalised to -14 LUFS for YouTube.
# usage: join60.sh <out.mp4> <fade_seconds> clip1.mp4 clip2.mp4 [...]
set -euo pipefail
OUT="$1"; FADE="$2"; shift 2
N=$#
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
inputs=(); vf=""; af=""; i=0; prev="v0"; aprev="0:a"
for f in "$@"; do inputs+=(-i "$f"); vf+="[$i:v]settb=AVTB,fps=60[v$i];"; i=$((i+1)); done
i=1; offset=$(python -c "print($(dur "$1") - $FADE)")
for f in "${@:2}"; do
  last=$([ $i -eq $((N-1)) ] && echo 1 || echo 0)
  vf+="[$prev][v$i]xfade=transition=fade:duration=$FADE:offset=$offset"
  [ $last -eq 1 ] && vf+=",format=yuv420p[v]" || vf+="[x$i];"
  af+="[$aprev][$i:a]acrossfade=d=$FADE"
  [ $last -eq 1 ] && af+=",loudnorm=I=-14:TP=-1.0:LRA=11,aresample=48000[a]" || af+="[ax$i];"
  aprev="ax$i"; prev="x$i"
  offset=$(python -c "print($offset + $(dur "$f") - $FADE)")
  i=$((i+1))
done
ffmpeg -y -loglevel error "${inputs[@]}" -filter_complex "$vf;$af" -map "[v]" -map "[a]" \
  -c:v libx264 -crf 16 -preset slow -profile:v high -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart "$OUT"
echo "joined $N clips -> $OUT ($(dur "$OUT")s)"
