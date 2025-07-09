# Benchmarks on Ryzen 5950X, values are MiB/s

```
bench  yy-libcurl  yy-h2d  yy-kernel    Avg:  5191.8 +/-  77.600  min:  5091.0000 max:  5373.0000
bench  yy-libcurl  no-h2d  no-kernel    Avg:  5641.8 +/- 382.957  min:  5112.0000 max:  6134.0000
bench  no-libcurl  yy-h2d  yy-kernel    Avg: 14527.0 +/- 454.536  min: 13502.0000 max: 14965.0000
bench  no-libcurl  yy-h2d  no-kernel    Avg: 15216.0 +/- 410.804  min: 14264.0000 max: 15573.0000
bench  no-libcurl  no-h2d  no-kernel    Avg: 22863.4 +/- 555.385  min: 22175.0000 max: 23564.0000
memory card manufacturer's decl throughput:  24435  (two-channel)
                                             12217  (one-channel)
```

# Just memory copy

My memory card is  Kingston FURY 32GB DDR4 3200MHz CL16 Beast Black with throughput of 25 600 MB/s = 24,435 MiB/s

512 KiB block => 53513.62 MiB/s
512 MiB block => 14210.71 MiB/s
