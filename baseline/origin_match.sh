step=1
if [ $step == 1 ] ; then
  for i in 1 2 3 4 5 6 7 8 9 10 11 12
  do
      python3.9 origin_pattern_match.py $step $i
  done
else
    for i in 0 1 2 3 4 5 6 7 8 9 10 11 12
  do
      python3.9 origin_pattern_match.py $step $i
  done
fi