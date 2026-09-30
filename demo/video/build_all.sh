#!/bin/bash
# build every clip: name page duration
set -e
cd "$(dirname "$0")"
while read -r clip page dur; do
  [ -z "$clip" ] && continue
  PYTHONIOENCODING=utf-8 python build_clip.py "$clip" "$page" "$dur" --title "$clip" | tail -1
done <<LIST
c1_hook pages/c1_hook.html ${D_c1_hook:-18.3}
c2_refinery capture/refinery.html ${D_c2_refinery:-29.4}
c3_plan capture/plan4.html ${D_c3_plan:-17.5}
c4_unload capture/unload.html ${D_c4_unload:-14.5}
c5_gpu capture/datt256.html ${D_c5_gpu:-15.5}
c5b_arch pages/c5b_arch.html ${D_c5b_arch:-31.2}
c6_proof capture/datt256.html ${D_c6_proof:-31.4}
c6b_next pages/c6b_next.html ${D_c6b_next:-24.0}
c7_end pages/c7_end.html ${D_c7_end:-9.8}
LIST
