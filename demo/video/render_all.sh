#!/usr/bin/env bash
# Render every composition to out/<clip>.mp4 at 60 fps.
set -e
cd "$(dirname "$0")"
export HYPERFRAMES_SKIP_SKILLS=1
for c in ${@:-c1_hook c2_refinery c3_plan c4_unload c5_gpu c5b_arch c6_proof c6b_next c7_end}; do
  (cd renders/$c && npx -y hyperframes@0.8.97 render -o ../../out/$c.mp4 --fps 60 -q delivery > ../../out/$c.log 2>&1) && echo "rendered $c" || { echo "FAILED $c"; tail -20 out/$c.log; }
done
