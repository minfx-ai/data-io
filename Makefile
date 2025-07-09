server:
	miniserve -p 8080 --verbose .

payload:
	dd if=/dev/zero of=data/32MB bs=1M count=32
	dd if=/dev/zero of=data/128MB bs=1M count=128

kernels:
	nvcc -std=c++17 -O3 -c src/kernels.cu -o build/kernels.o

loader_stub:
	gcc src/loader.c -DSTUB_GPU -O3 -lm -Icurl/include -Icurl/lib -Lcurl/lib/.libs -lcurl -o bin/loader

loader_cuda: kernels
	gcc -O3 -mavx2 -lm -Icurl/include -Icurl/lib -c src/loader.c -o build/loader.o
	nvcc -O3 -std=c++17 build/loader.o build/kernels.o -Lcurl/lib/.libs -lcurl -lm -o bin/loader

bench:
	bin/loader --batch_size=64 --max_batches=1024 --n_producers=8

bench_dry_run:
	bin/loader --batch_size=24 --max_batches=2048 --n_producers=24 --dry_run=1 --use_hugepages=0

memory_benchmark:
	gcc -O4 -march=native -Wall -Wextra -o bin/memcpy_benchmark src/memcpy_benchmark.c