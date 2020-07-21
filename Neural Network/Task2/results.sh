#!/bin/bash
a="round_1:  "
b=1000
h=1
c=" ==================== "
d="/"
f=2
g="round_2:  "
# for ((i=1; i<b; i++))
# do
# 	e="$c$a$i$d$b$c"
# 	echo "$e"
# 	python3 best_model.py 1 0
# done
# python3 best_model.py 3 0

for ((i=0; i<h; i++))
do
	echo "$c$g$i$d$h$c"
	python3 best_model.py 2 $i
done
	python3 best_model.py 4 0
