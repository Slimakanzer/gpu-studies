#!/bin/bash -x

USAGE="Usage: $0 -l line -f source_file -o debug_buffer_path -w watches -t counter -v pooling_args -p perl_args"

while getopts "l:f:o:w:t:v:" opt
do
	echo "-$opt $OPTARG"
	case "$opt" in
	l) line=$OPTARG ;;
	f) source_file=$OPTARG ;;
	o) debug_buffer_path=$OPTARG ;;
	w) watches=$OPTARG ;;
	t) counter=$OPTARG ;;
	v) pooling_args=$OPTARG ;;
	p) perl_args=$OPTARG ;;
	esac
done

[[ -z "$line" || -z "$source_file" || -z "$debug_buffer_path" || -z "$watches" ]] && { echo $USAGE; exit 1; }

rm -rf tmp_dir/
mkdir tmp_dir

export DEBUGGER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export ASM_DBG_CONFIG="$DEBUGGER_DIR/config.toml"
export HSA_TOOLS_LIB="$DEBUGGER_DIR/libplugintercept.so"

"$DEBUGGER_DIR/../build/gfx9/fp32_pooling"    	\
	--asm "$source_file"						\
	--include "$DEBUGGER_DIR/../gfx9/include"	\
	--output_path "./tmp_dir/fp32_pooling.co"   \
	$pooling_args

echo