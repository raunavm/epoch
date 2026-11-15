#!/bin/sh
# Screenshot an object page (all "check it" panels open) for the demo video. Needs the site served on :8765.
C="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$C" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1.5 --window-size=960,560 \
  --virtual-time-budget=15000 --screenshot="$PWD/dist/object_shot.png" "http://localhost:8765/object.html?id=${1:-757}&check=all" 2>/dev/null
