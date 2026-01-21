ray stop --force && pkill -9 -f "ray::" 2>/dev/null; pkill -9 -f "raylet" 2>/dev/null; pkill -9 -f "gcs_server" 2>/dev/null; sleep 2 && echo "Ray processes stopped"

rm -rf /tmp/ray/* 2>/dev/null; rm -rf /tmp/ray_tmp_* 2>/dev/null; echo "Cleaned Ray temp files"