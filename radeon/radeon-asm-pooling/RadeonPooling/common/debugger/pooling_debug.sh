#!/bin/bash -x

while getopts "l:f:o:w:t:p:s:v:" opt
do
	echo "$opt $OPTARG"
	case "$opt" in
	l) line=$OPTARG
	;;
	f) src_path=$OPTARG
	;;
	o) dump_path=$OPTARG
	;;
	w) watches=$OPTARG
	;;
	t) counter=$OPTARG
	;;
	v) pooling_args=$OPTARG
	;;
	p) perl_args=$OPTARG
	;;
	s) debug_size=$OPTARG
	;;
	esac
done

rm -rf tmp_dir
mkdir tmp_dir
num_watches=`echo "${watches}" | awk -F":" '{print NF}'`

tmp=tmp_dir/tmp_gcn_breakpoint_pl.s
export BREAKPOINT_SCRIPT_OPTIONS="-l $line -o $tmp -s 96 -r s0 -t $counter $perl_args"
export BREAKPOINT_SCRIPT_WATCHES="$watches"
export ASM_DBG_BUF_SIZE="$debug_size"
export DEBUGGER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

"$DEBUGGER_DIR/../../build/gfx9/fp32_pooling"        \
	--clang "$DEBUGGER_DIR/dbg_clang_wrapper.sh" \
	--asm "$src_path"                            \
	--include "$DEBUGGER_DIR/../../gfx9/include" \
	--output_path "./tmp_dir/fp32_pooling.co"    \
	--debug_path "$dump_path"                    \
	--debug_size $ASM_DBG_BUF_SIZE               \
	$pooling_args

echo