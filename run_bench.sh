#!/bin/bash

NUM_RUNS=8

function run_bench() {
    CMD=$1
    OUTFILE=$2
    
    echo -n "Running $OUTFILE: "
    sleep 1

    echo "" > /tmp/repeated
    for i in `seq 1 $NUM_RUNS`; do 
        $CMD 2>&1 >> /tmp/repeated; 
        echo -n "."
    done
    echo ""

    cat /tmp/repeated | grep final | awk '{print $NF}' > /tmp/throughputs

    {
    awk '{
        val = $NF
        x[NR] = val
        sum += val
        sumsq += val * val
        if (NR == 1 || val < min) min = val
        if (NR == 1 || val > max) max = val
    }
    END {
        n = NR
        avg = sum / n
        std = sqrt((sumsq - sum * sum / n) / n)
        print "Avg:", avg, "+/-", std, " -- min:", min, "max:", max
    }' /tmp/throughputs
    echo -n "all: "; cat /tmp/throughputs | tr '\n' ' '
    echo -n "\n"
    } > $OUTFILE
}

make loader_stub
# no gpu computation
run_bench "bin/loader --batch_size=64 --max_batches=1024 --n_producers=8 --dry_run=1" "bench_no-libcurl_no-computation.txt"
run_bench "bin/loader --batch_size=64 --max_batches=1024 --n_producers=8 --dry_run=0" "bench_libcurl_no-computation.txt"

make loader_cuda
run_bench "bin/loader --batch_size=64 --max_batches=1024 --n_producers=8 --dry_run=2" "bench_no-libcurl_h2d_no-computation.txt"
run_bench "bin/loader --batch_size=64 --max_batches=1024 --n_producers=8 --dry_run=1" "bench_no-libcurl_h2d_computation.txt"
run_bench "bin/loader --batch_size=64 --max_batches=1024 --n_producers=8 --dry_run=0" "bench_libcurl_h2d_computation.txt"

echo "--------------------------------"
echo "Summary:"
grep "Avg:" bench_*

