#!/bin/bash

CDIR=$PWD
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR
echo entering $DIR

if [ "$1" = "-f"  ]; then
	echo "cleaning ..."
	rm -f gen_res_script_tokens.cpp 
	rm -f gen_res_script_parser.cpp 
	rm -f gen_effect_script_tokens.cpp 
	rm -f gen_effect_script_parser.cpp 
fi

echo "....."

if [ gen_res_script_tokens.cpp -ot res_script_tokens.l ]; then
	cmd="flex   -Phqengine_res_parser_ -ogen_res_script_tokens.cpp  res_script_tokens.l"
	echo "$cmd"
	eval $cmd
else
	echo "skip res_script_tokens.l"
fi
if [ gen_res_script_parser.cpp -ot res_script_parser.y ]; then
	cmd="bison  -d  -v -p hqengine_res_parser_ -o gen_res_script_parser.cpp res_script_parser.y"
	echo "$cmd"
	eval $cmd
else
	echo "skip res_script_parser.y"
fi

if [ gen_effect_script_tokens.cpp -ot effect_script_tokens.l ]; then
	cmd="flex   -Phqengine_effect_parser_ -ogen_effect_script_tokens.cpp  effect_script_tokens.l"
	echo "$cmd"
	eval $cmd
else
	echo "skip effect_script_tokens.l"
fi
if [ gen_effect_script_parser.cpp -ot effect_script_parser.y ]; then
	cmd="bison  -d  -v -p hqengine_effect_parser_ -o gen_effect_script_parser.cpp effect_script_parser.y"
	echo "$cmd"
	eval $cmd
else
	echo "skip effect_script_parser.y"
fi

echo "....."

echo leaving $DIR
cd $CDIR

