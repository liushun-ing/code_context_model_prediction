step=1
if [ $step -eq 1 ]; then
  for i in $(seq 2 12); do
    python3.9 origin_pattern_match.py $step $i
  done
else
  for i in $(seq 0 12); do
    python3.9 origin_pattern_match.py $step $i
  done
fi