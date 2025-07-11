// loader.c — lock-free triple-buffer streaming loader
// ────────────────────────────────────────────────────────────────────────────
//
// Overview
// --------
// A parent **master** process cooperates with **N producer** processes via a
// single, page-locked shared-memory region:
//
//                CPU                    PCIe                     GPU
//         ┌──────────────┐    ┌──────────────────────┐    ┌────────────────┐
//         │ producers  * │ => │ triple buffer (H2D)  │ => │  kernel stream │
//         └──────────────┘    └──────────────────────┘    └────────────────┘
//
// The region holds three buffers (`NBUF = 3`); each buffer stores one
// **batch_bytes** = `batch_size × slot_bytes`.
//
// Life-cycle per buffer
//     █ 0  FILLING    (producers write records; ticket dispenser)
//     █ 1  READY      (last producer flips state)
//     █ 2  IN_GPU     (master enqueues cudaMemcpyAsync + kernel)
//     └──→ back to 0  (master zeroes header, opens buffer for producers)
//
// Two global monotonic counters drive the ring:
//
//     prod_epoch : next batch producers will fill
//     cons_epoch : next batch master will consume
//
// Back-pressure = `prod_epoch - cons_epoch >= NBUF`  → producers spin.
//
// Compile
// -------
// * make loader_cuda -- use nvcc to compile with CUDA
// * make loader_stub -- use gcc to compile without CUDA
//
// All GPU resources (streams, cudaMalloc etc.) are created once in gpu_init();
// the hot path never calls into the OS or CUDA driver except the three async
// operations per batch (`memcpyAsync`, kernel launch, `StreamSynchronize`).
//
// Goals & Constraints
// -------------------
//
// | Aspect     | Constraint / Target                                          |
// |------------|--------------------------------------------------------------|
// | Throughput | One PCIe DMA per batch, overlap with kernel and CPU refill   |
// | Sys-calls  | Only at start-up: memfd_create, fallocate, mmap, mlock       |
// | Sync       | Pure C11 atomics; no futexes, pipes, mutexes in the hot path |
// | Memory     | Fixed 3 x batch_bytes in 2 MiB-huge-page, mlock()’d segment  |
// | Producers  | Independent processes, core-pinned (sched_setaffinity)       |
// | Master     | Busy-waits on one cache-line word (state), recycles buffers  |
//
// Quick-tuning notes
// ------------------
// * Increase `batch_size` (8–16 MiB) to approach peak PCIe bandwidth.  
// * Pin processes to a NUMA node that shares the GPU’s PCIe root complex.  
// * Check `cudaHostRegister()` status; Nsight Systems should mark the host
//   ranges as **Pinned**.  
// * If slot size is tiny and producers fight over a cache line, switch to
//   chunked tickets or per-producer sub-rings.
//
// Debugging
// ---------
// Define `-DDEBUG=1` to enable `DEBUG_PRINT(fmt,…)`, a zero-overhead logger
// that compiles to nothing when `DEBUG==0`.
//
// The resulting binary sustains >90 % of PCIe Gen4 bandwidth on a single GPU
// while keeping the CPU hot path completely lock-free.

#define _GNU_SOURCE

#include <curl/curl.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <immintrin.h>
#include <linux/memfd.h>
#include <math.h>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define NBUF 3                 // triple‑buffering fixed
#define MAX_PRODUCERS  64      // guard against silly config values
#define TAG_BUFSZ      8192    // per-producer line assembly buffer
// Round up to the nearest multiple of `a`
#define ALIGN_UP(x, a)   ( ((x) + (a) - 1) & ~((a) - 1) )

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

#if DEBUG
  #define DEBUG_PRINT(fmt, ...) \
      fprintf(stderr, "[debug] " fmt "\n", ##__VA_ARGS__)
#else
  #define DEBUG_PRINT(fmt, ...) \
      do {} while (0)
#endif

static inline void xperror(const char* msg) {
  perror(msg); // syscall!
  exit(EXIT_FAILURE); // syscall!
}

static inline long xsyscall(long ret, const char* where) {
  if (ret == -1) xperror(where);
  return ret;
}


// -----------------------------------------------------------------------------
// Config – run‑time tunables (64‑byte, cache‑aligned)
// -----------------------------------------------------------------------------

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage: %s [options]\n\n"
          "Options (defaults in brackets):\n"
          "  --local_rank=N    Local rank of GPU (-1 from LOCAL_RANK env) [0]\n"
          "  --window_len=N    Samples per window           [1024]\n"
          "  --feat_size=N     Features per sample          [256]\n"
          "  --float_size=N    Bytes per scalar (2|4)       [2]\n"
          "  --batch_size=N    Records per GPU launch       [512]\n"
          "  --n_producers=N   Producer processes           [12]\n"
          "  --max_batches=N   Max batches to produce       [128]\n"
          "  --use_hugepages   Use hugepages                [0]\n"
          "  --dry_run=0       Normal run (default)\n"
          "  --dry_run=1       Dry run (no libcurl loading)\n"
          "  --dry_run=2       Dry run (no GPU computation)\n"
          "  -h, --help        Show this message\n",
          prog);
  exit(EXIT_FAILURE); // syscall!
}

typedef struct __attribute__((aligned(64))) {
    int32_t local_rank;      // local rank of GPU
    uint32_t window_len;      // samples per window
    uint32_t feat_size;       // features per sample
    uint32_t float_size;      // bytes per scalar (2 = fp16, 4 = fp32)
    uint32_t batch_size;      // records per GPU launch
    uint32_t n_producers;     // producer processes
    uint32_t max_batches;     // max batches to produce
    uint64_t retry_sleep;     // sleep time between retries
    uint32_t dry_run;         // 0 normal run, 
                              // 1 dry run (no libcurl loading), 
                              // 2 dry run (no GPU computation)
    uint32_t use_hugepages;   // 1 if use hugepages, 0 otherwise

    // Derived (filled in by parse_config)
    uint64_t slot_bytes;      // feat_size * window_len * float_size
    uint64_t batch_bytes;   // batch_size * slot_bytes
} Config;

static Config parse_config(int argc, char** argv) {
  Config c = {
      .local_rank    = 0,
      .window_len    = 1024,
      .feat_size     = 256,
      .float_size    = 2,
      .batch_size    = 512,
      .n_producers   = 12,
      .max_batches   = 512,
      .use_hugepages = 0,
      .dry_run       = 0,
  };

  static const struct option opts[] = {
      {"local_rank",     required_argument, 0, 'l'},
      {"window_len",     required_argument, 0, 'w'},
      {"feat_size",      required_argument, 0, 'f'},
      {"float_size",     required_argument, 0, 's'},
      {"batch_size",     required_argument, 0, 'b'},
      {"n_producers",    required_argument, 0, 'n'},
      {"max_batches",    required_argument, 0, 't'},
      {"use_hugepages",  required_argument, 0, 'u'},
      {"dry_run",        required_argument, 0, 'd'},
      {"help",           no_argument,       0, 'h'},
      {0, 0, 0, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "w:f:s:b:n:t:u:d:h", 
                            (struct option *)opts, NULL)) != -1) {
      switch (opt) {
          case 'l': c.local_rank    = (int32_t)  strtol(optarg, NULL, 0); break;
          case 'w': c.window_len    = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 'f': c.feat_size     = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 's': c.float_size    = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 'b': c.batch_size    = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 'n': c.n_producers   = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 't': c.max_batches   = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 'u': c.use_hugepages = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 'd': c.dry_run       = (uint32_t)strtoul(optarg, NULL, 0); break;
          case 'h': default: usage(argv[0]);
      }
  }

  uint64_t raw_slot = (uint64_t)c.window_len * c.feat_size * c.float_size;
  // Round slot to 4 KiB (page) and 32 B (WC) so two different producers
  // never contend for the same cache line inside the payload region.
  c.slot_bytes      = ALIGN_UP(raw_slot, 4096);

  c.batch_bytes = (uint64_t) c.batch_size * c.slot_bytes;
  c.retry_sleep = 100;

  if (!c.slot_bytes || !c.batch_bytes) xperror("invalid zero parameter");
  if (c.n_producers > MAX_PRODUCERS) {
      fprintf(stderr, "n_producers capped at %d\n", MAX_PRODUCERS);
      exit(EXIT_FAILURE); // syscall!
  }

  if (c.local_rank == -1) {
    const char* local_rank_str = getenv("LOCAL_RANK");
    if (local_rank_str) {
        c.local_rank = atoi(local_rank_str);
    } else {
        fprintf(stderr, "Auto discovery of LOCAL_RANK was set, "
                        "but env was not set, using default rank 0\n");
        c.local_rank = 0;
    }
  }

  return c;
}

// -----------------------------------------------------------------------------
// Shared‑memory layout – header + NBUF variable‑length buffers
// -----------------------------------------------------------------------------

// --- new buffer life-cycle flags ----------------------------------------
enum { BUF_FREE = 0, BUF_FILLING = 1, BUF_READY = 2, BUF_IN_GPU = 3 };

// 128-B header = 2 × L1D lines → no false sharing between control words
// and the first 128-aligned payload byte.
typedef struct __attribute__((aligned(64))) {
    _Atomic uint32_t state;      // BUF_*
    _Atomic uint32_t head;       // ticket dispenser
    _Atomic uint32_t written;    // #records finished
    _Atomic uint32_t epoch;      // monotonically increasing batch id
} BufferHeader;

// shared header: split the single epoch counter in two monotonic ones
typedef struct __attribute__((aligned(64))) {
    Config          cfg;
    _Atomic uint32_t prod_epoch;     // next epoch producers will fill
    _Atomic uint32_t cons_epoch;     // next epoch the master will use
    _Atomic uint32_t producers_left; // number of producers left to finish
    uint32_t         url_count;
    _Atomic uint32_t url_idx;
    char             urls[8192];
} SharedHeader;

// make sure compile-time sizes do not break 32-alignment
_Static_assert(sizeof(BufferHeader) % 32 == 0,
               "BufferHeader must be a multiple of 32 B");
_Static_assert(sizeof(SharedHeader)  % 32 == 0,
               "SharedHeader must be a multiple of 32 B");


static inline uint8_t *payload_ptr(BufferHeader *buf) {
    return (uint8_t *)(buf + 1);
}
static inline size_t buffer_bytes(const Config *cfg) {
    return sizeof(BufferHeader) + cfg->batch_bytes;
}
static inline BufferHeader *get_buffer(SharedHeader *shm, const Config *cfg, 
                                       uint32_t idx) {
    uint8_t *base = 
        (uint8_t *)shm + sizeof(SharedHeader) + idx * buffer_bytes(cfg);
    return (BufferHeader *)base;
}
static inline size_t scratch_offset(const Config *c){
    size_t off = sizeof(SharedHeader) + NBUF * buffer_bytes(c);
    return ALIGN_UP(off, 32);
}
static inline size_t scratch_region_bytes(const Config *c){
    return (size_t)c->n_producers * c->slot_bytes;
}
static inline size_t shm_total_bytes(const Config *c){
    return scratch_offset(c) + scratch_region_bytes(c);
}
static inline uint8_t* scratch_base(SharedHeader *shm,const Config *c){
    return (uint8_t*)shm + scratch_offset(c);
}
static inline uint8_t* scratch_ptr(SharedHeader *shm,const Config *c, 
                                   uint32_t pid){
    return scratch_base(shm,c) + (uint64_t)pid * c->slot_bytes;
}

// Allocate shared memory with hugepages.
// This is a wrapper around memfd_create and mmap.
// It also pins the memory so it will not be swapped out.
static SharedHeader* create_pinned_hugetlb_shm(const Config* cfg) {
    size_t bytes = shm_total_bytes(cfg);

    // 1. memfd_create with 2 MiB hugepages
    int flags = MFD_CLOEXEC;
    if (cfg->use_hugepages) flags |= MFD_HUGETLB | MFD_HUGE_2MB;
    
    int fd = syscall(SYS_memfd_create, "gpu_batch", flags); // syscall!
    if (fd == -1) {
        // fall back to normal pages if hugepages not available
        fd = syscall(SYS_memfd_create, "gpu_batch", MFD_CLOEXEC); // syscall!
    }

    // 2. reserve and back the pages so we will not fault later
    if (fallocate(fd, 0, 0, bytes) == -1 && errno == ENOSPC) { // syscall!
        fprintf(stderr, "huge-page pool empty: falling back to normal pages\n");
        close(fd); // syscall!
        fd = memfd_create("gpu_batch", MFD_CLOEXEC); // syscall!
        xsyscall(fallocate(fd, 0, 0, bytes), "fallocate"); // syscall!
    }

    // 3. map into every process (MAP_SHARED)
    void* addr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); // syscall!
    if (addr == MAP_FAILED) xperror("mmap");

    // 4. keep resident (pins for DMA too)
    xsyscall(mlock(addr, bytes), "mlock"); // syscall!

    SharedHeader *shm = (SharedHeader *)addr;
    memcpy(&shm->cfg, cfg, sizeof(Config));

    /* the two monotonic counters */
    atomic_store_explicit(&shm->prod_epoch, 0, memory_order_relaxed);
    atomic_store_explicit(&shm->cons_epoch, 0, memory_order_relaxed);
    atomic_store_explicit(&shm->producers_left, cfg->n_producers, memory_order_relaxed);

    shm->url_count = 0;
    atomic_store_explicit(&shm->url_idx, 0, memory_order_relaxed);
    shm->urls[0] = '\0';

    /* prepare the triple buffer for immediate use */
    for (int i = 0; i < NBUF; ++i) {
        BufferHeader *b = get_buffer(shm, cfg, i);

        atomic_store_explicit(&b->state,   BUF_FILLING, memory_order_relaxed);
        atomic_store_explicit(&b->head,    0,           memory_order_relaxed);
        atomic_store_explicit(&b->written, 0,           memory_order_relaxed);
        atomic_store_explicit(&b->epoch,   0,           memory_order_relaxed);
    }
    return shm;
}



// -----------------------------------------------------------------------------
// GPU stubs (undef STUB_GPU for real CUDA)
// -----------------------------------------------------------------------------

// one “mailbox” is enough in stub mode because the calls
// are strictly gpu_copy() → gpu_run_kernel() in sequence
static unsigned long long pending_sum = 0ULL;

static inline void cpu_sum(int idx, const void *src, const Config *cfg) {
  const uint64_t *p = (const uint64_t *)src;

  size_t n64 = cfg->batch_bytes >> 3; /* bytes / 8          */
  uint64_t acc = 0;
  for (size_t i = 0; i < n64; ++i)
    acc += p[i];

  /* handle up-to-7 trailing bytes */
  const uint8_t *tail = (const uint8_t *)(p + n64);
  for (size_t i = 0; i < (cfg->batch_bytes & 7); ++i)
    acc += tail[i];

  pending_sum = acc;
}

#ifdef STUB_GPU

static inline void gpu_init(SharedHeader *shm) {}

  /* host-to-“device” copy + checksum (64-bit chunks, remainder byte-wise) */
  static inline void gpu_copy(int idx, const void *src, const Config *cfg) {
    // Uncomment to actually do some work.
    // cpu_sum(idx, src, cfg);
  }

  /* called right after gpu_copy(): just print the checksum */
  static inline void gpu_run_kernel(int idx) {
    // Uncomment to actually do some work.
    // fprintf(stderr, "[CPU-stub] buffer %d → sum = %llu\n", idx,
    //         (unsigned long long)pending_sum);
  }

#else

#include <cuda_runtime.h>
#include "kernels.h"

static void *d_buffers[NBUF];
static unsigned long long *d_sums[NBUF]; // device-side sums
static unsigned long long h_sums[NBUF];  // host-side sums
static cudaStream_t streams[NBUF];

static inline void gpu_init(SharedHeader *shm) {
    const Config *cfg = &shm->cfg;
    DEBUG_PRINT("gpu_init: initializing %d buffers\n", NBUF);
    DEBUG_PRINT("gpu_init: allocating in total: %ld MiB\n", 
                NBUF * cfg->batch_bytes / (1 << 20));

    cudaSetDevice(cfg->local_rank);

    for (int i = 0; i < NBUF; ++i) {
        cudaMalloc(&d_buffers[i], cfg->batch_bytes);
        cudaStreamCreate(&streams[i]);

        // Allocate host-side and device-side sums
        cudaHostAlloc((void **)&h_sums[i], sizeof(unsigned long long),
                      cudaHostAllocPortable);
        cudaMalloc((void **)&d_sums[i], sizeof(unsigned long long));
    }

    /* register host buffers once – zero-copy friendly */
    for (int i = 0; i < NBUF; ++i) {
        void *h = payload_ptr(get_buffer(shm, &shm->cfg, i));        /* real */
        cudaHostRegister(h, shm->cfg.batch_bytes, cudaHostRegisterPortable);
    }
}

static inline void gpu_copy(int idx, const void *src, const Config *cfg) {
  // 1. reset device accumulator
  unsigned long long zero = 0ULL;
  cudaMemcpyAsync(d_sums[idx], &zero, sizeof(zero), cudaMemcpyHostToDevice,
                  streams[idx]);

  // 2. copy the payload
  cudaMemcpyAsync(d_buffers[idx], src, cfg->batch_bytes,
                  cudaMemcpyHostToDevice, streams[idx]);

  // 3. launch kernel (same stream ⇒ copy must finish first)
  if (cfg->dry_run < 2) {
    launch_sum_kernel(
        d_buffers[idx], cfg->batch_bytes, d_sums[idx], streams[idx]);
  }

  // 4. bring result back (also on the same stream)
  cudaMemcpyAsync(&h_sums[idx], d_sums[idx], sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost, streams[idx]);
}

static inline void gpu_run_kernel(int idx) { 
    cudaStreamSynchronize(streams[idx]);
    DEBUG_PRINT("[GPU] buffer %d → sum = %llu\n", idx, h_sums[idx]);
}

#endif // STUB_GPU

// -----------------------------------------------------------------------------
//  split_urls – returns malloc‑ed array terminated by NULL, count in *n
// -----------------------------------------------------------------------------
static char **split_urls(const char *s, int *n)
{
    if (!s) return NULL;
    size_t len = strlen(s);
    char *buf  = malloc(len + 1);
    if (!buf) return NULL;
    strcpy(buf, s);

    /* worst‑case count = 1 + #newlines */
    size_t cap = 1;
    for (const char *p = s; *p; ++p) if (*p == '\n') cap++;
    char **list = calloc(cap + 1, sizeof(char*));
    if (!list) { free(buf); return NULL; }

    int idx = 0;
    char *tok, *save;
    for (tok = strtok_r(buf, "\n", &save); tok; tok = strtok_r(NULL, "\n", &save)) {
        list[idx++] = tok;
    }
    list[idx] = NULL;
    *n = idx;
    return list;            /* note: single malloc block for all strings */
}

// -----------------------------------------------------------------------------
// Sync helpers
// -----------------------------------------------------------------------------

static inline int try_claim_slot(BufferHeader *buf, uint32_t batch_size,
                                 uint32_t *slot_out)
{
    // One atomic-add, no CAS:
    //   – fetch_add is already atomic
    //   – relaxed, because readers never depend on its ordering
    uint32_t head = atomic_fetch_add_explicit(&buf->head, 1,
                                              memory_order_relaxed);
    if (head >= batch_size)         /* batch full → ticket discarded */
        return 0;
    *slot_out = head;
    return 1;
}

//  Pin caller to one logical core and switch to SCHED_FIFO priority 1.
static inline void pin_to_core_and_fifo(int core_id)
{
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(core_id, &cs);
    if (sched_setaffinity(0, sizeof(cs), &cs) != 0) // syscall!
        perror("sched_setaffinity"); // syscall!

    struct sched_param sp = { .sched_priority = 1 };
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) // syscall!
        perror("sched_setscheduler (need CAP_SYS_NICE?)"); // syscall!
}


// -----------------------------------------------------------------------------
//  Streaming producer
// -----------------------------------------------------------------------------

// Helper for dry run mode.
// Try to fill slot with AVX2 instructions if possible, memset otherwise.
static inline void fill_slot(void *dst, uint8_t byte, size_t n)
{
#if defined(__AVX2__)
    if (((uintptr_t)dst & 31) == 0 && n >= 4096) {
        __m256i v = _mm256_set1_epi8((char)byte);
        for (size_t i = 0; i < n; i += 64) {
            _mm256_stream_si256((__m256i*)((uint8_t*)dst + i), v);
            _mm256_stream_si256((__m256i*)((uint8_t*)dst + i + 32), v);
        }
        _mm_sfence();                     /* flush WC */
        return;
    }
#endif
    memset(dst, byte, n);
}

// producer_loop_dry_run()
// 
// Lock-free producer for the “dry-run” mode (no libcurl, just memset patterns)
//
//  Life-cycle (one iteration = one attempt to write **one** record)
//  -----------------------------------------------------------------
//
//   0.  **Init & naming**
//       * give the process a readable `ps` name  →  "loader-prod-<pid>"
//       * lookup immutable `Config` once
//       * get a pointer to this producer’s private scratch slot
//
//   1.  **Global termination check**
//       cur_prod = prod_epoch (what I am about to produce)
//       cur_cons = cons_epoch (what the master is about to consume)
//       → exit loop if `cur_prod >= max_batches`
//
//   2.  **Ring-capacity guard**     (back-pressure)
//       if  (cur_prod - cur_cons) ≥ NBUF
//           ring is full → `_mm_pause()` and retry
//
//   3.  **Pick buffer for current epoch**
//       buf = buffers[ cur_prod % NBUF ]
//
//   4.  **Wait until master recycled it**
//       spin until  `buf->state == BUF_FILLING`
//
//   5.  **Ticket dispenser**
//       slot = atomic_fetch_add(buf->head, 1)
//       if slot ≥ batch_size                → batch already full, restart loop
//
//   6.  **Write payload**
//       dst = payload_ptr(buf) + slot * SLOT_BYTES
//       fill_slot(dst, pid)                 (simple memset pattern)
//
//   7.  **Commit**
//       if atomic_fetch_add(buf->written, 1) + 1 == batch_size   ⇒ I’m last
//          * buf->epoch  = cur_prod          (publish epoch #)
//          * buf->state  = BUF_READY         (make visible to master)
//          * prod_epoch++                    (open next epoch)
//
//   8.  **Loop**
//       goto step 1
//
//  Synchronisation & ordering
//  --------------------------
//    * Only producers mutate `prod_epoch`; only master mutates `cons_epoch`.
//    * State machine per buffer:  FILLING → READY → IN_GPU → FILLING
//    * `memory_order_acquire` on loads ensures producer sees master’s recycle
//      before using the buffer; `memory_order_release` when flipping to READY
//      ensures master sees fully-written data once it observes READY.
//
//  Guarantees
//  ----------
//    1. Ring never overruns: distance check in step 2.
//    2. No lost records: buffer recycled only after GPU stage finished.
//    3. Works for any speed skew: slow or fast producers, slow or fast master.
static void producer_loop_dry_run(SharedHeader *shm, uint32_t pid)
{
    char pname[16];
    snprintf(pname, sizeof pname, "loader-prod-%u", pid);
    prctl(PR_SET_NAME, (unsigned long)pname, 0, 0, 0); // syscall!

    const Config *cfg = &shm->cfg;
    fprintf(stderr, "[producer %u] starting\n", pid);
    uint8_t *scratch  = scratch_ptr(shm, cfg, pid);

    while (1) {
        DEBUG_PRINT("[producer %u] entering loop\n", pid);
        /* ---- 1.  stop if the job is finished ------------------------ */
        uint32_t cur_cons = atomic_load_explicit(&shm->cons_epoch,
                                                 memory_order_acquire);
        uint32_t cur_prod = atomic_load_explicit(&shm->prod_epoch,
                                                 memory_order_acquire);
        if (cur_prod >= cfg->max_batches) {
            DEBUG_PRINT("[producer %u] breaking out of loop\n", pid);
            break;
        }

        /* ---- 2.  enforce ring capacity ------------------------------ */
        if (cur_prod - cur_cons >= NBUF) {       /* ring full → back-off */
            DEBUG_PRINT("[producer %u] ring full → back-off\n", pid);
            _mm_pause();
            continue;
        }

        DEBUG_PRINT("[producer %u] getting buffer\n", pid);
        BufferHeader *buf = get_buffer(shm, cfg, cur_prod % NBUF);

        DEBUG_PRINT("[producer %u] waiting for buf to be FILLING\n", pid);
        /* wait until master recycled the buffer */
        if (atomic_load_explicit(&buf->state, memory_order_acquire)
                != BUF_FILLING) {
            _mm_pause();
            continue;
        }

        /* ---- 3.  try to claim a slot -------------------------------- */
        DEBUG_PRINT("[producer %u] trying to claim slot\n", pid);
        uint32_t slot;
        if (!try_claim_slot(buf, cfg->batch_size, &slot))
            continue;                            /* batch already full   */

        /* ---- 4.  write the sample ----------------------------------- */
        DEBUG_PRINT("[producer %u] writing sample\n", pid);
        uint8_t *dst = payload_ptr(buf) + (uint64_t)slot * cfg->slot_bytes;
        fill_slot(dst, (uint8_t)pid, cfg->slot_bytes);   /* or curl copy */

        /* ---- 5.  last writer publishes the buffer ------------------- */
        DEBUG_PRINT("[producer %u] maybe publishing buffer\n", pid);
        if (atomic_fetch_add_explicit(&buf->written, 1,
                                      memory_order_acq_rel) + 1
                == cfg->batch_size)
        {
            DEBUG_PRINT("[producer %u] publishing buffer\n", pid);
            atomic_store_explicit(&buf->epoch, cur_prod,
                                   memory_order_release);
            atomic_store_explicit(&buf->state, BUF_READY,
                                   memory_order_release);

            /* move on to the next epoch                                */
            atomic_fetch_add_explicit(&shm->prod_epoch, 1,
                                      memory_order_acq_rel);
        }
    }
    // Clean up
    atomic_fetch_sub_explicit(&shm->producers_left, 1, memory_order_release);

    fprintf(stderr, "[producer %u] exiting\n", pid);
    _exit(EXIT_SUCCESS); // syscall!
}

/* Per-transfer context passed to the curl write-callback */
typedef struct {
    SharedHeader *shm;
    const Config *cfg;
    uint32_t      pid;

    uint8_t      *scratch;   /* slot-sized temporary buffer          */
    size_t        fill;      /* bytes currently in scratch           */
    int           done;      /* set → abort transfer / quit loop     */
} stream_ctx;

// stream_write_cb – a libcurl WRITEFUNCTION
//
//  * Collect bytes until one SLOT is full
//  * Then atomically claim a ticket in the current buffer and copy it
//  * Last writer (written == batch_size) publishes the buffer and
//    advances prod_epoch.
//  * Back-pressure:
//        – if ring full     → spin (very short)
//        – if buffer not yet recycled → spin
static size_t stream_write_cb(void *ptr, size_t size, size_t nmemb,
                              void *userdata)
{
    stream_ctx *ctx   = userdata;
    SharedHeader *shm = ctx->shm;
    const Config *cfg = ctx->cfg;

    size_t avail = size * nmemb;
    uint8_t *p   = ptr;

    while (avail) {
        size_t need = cfg->slot_bytes - ctx->fill;
        size_t take = avail < need ? avail : need;

        memcpy(ctx->scratch + ctx->fill, p, take);
        ctx->fill += take;
        p         += take;
        avail     -= take;

        /* ─── scratch full → push one record into the ring ────────── */
        if (ctx->fill == cfg->slot_bytes) {

            while (1) {
                uint32_t cons = atomic_load_explicit(&shm->cons_epoch,
                                                     memory_order_acquire);
                uint32_t prod = atomic_load_explicit(&shm->prod_epoch,
                                                     memory_order_acquire);

                /* (1) global termination */
                if (prod >= cfg->max_batches) {
                    ctx->done = 1;
                    return 0;                   /* abort libcurl transfer */
                }

                /* (2) ring capacity guard */
                if (prod - cons >= NBUF) {
                    _mm_pause();
                    continue;
                }

                BufferHeader *buf =
                    get_buffer(shm, cfg, prod % NBUF);

                /* (3) wait until master recycled this buffer */
                if (atomic_load_explicit(&buf->state,
                                         memory_order_acquire)
                        != BUF_FILLING) {
                    _mm_pause();
                    continue;
                }

                /* (4) ticket dispenser */
                uint32_t slot;
                if (!try_claim_slot(buf, cfg->batch_size, &slot)) {
                    /* batch already full – someone else will publish it.
                       Re-check prod_epoch on next loop iteration.          */
                    _mm_pause();
                    continue;
                }

                /* (5) copy the slot */
                uint8_t *dst = payload_ptr(buf) +
                               (uint64_t)slot * cfg->slot_bytes;
                memcpy(dst, ctx->scratch, cfg->slot_bytes);

                /* (6) last writer publishes the buffer */
                if (atomic_fetch_add_explicit(&buf->written, 1,
                                              memory_order_acq_rel) + 1
                        == cfg->batch_size) {
                    atomic_store_explicit(&buf->epoch, prod,
                                          memory_order_release);
                    atomic_store_explicit(&buf->state, BUF_READY,
                                          memory_order_release);

                    atomic_fetch_add_explicit(&shm->prod_epoch, 1,
                                              memory_order_acq_rel);
                }
                break;  /* record successfully written */
            }

            ctx->fill = 0;   /* reset scratch for next window */
        }
    }
    return size * nmemb;      /* tell libcurl we consumed everything */
}

// producer_loop() – real streaming mode
static void producer_loop(SharedHeader *shm, uint32_t pid)
{
    /* --- cosmetic: give the process a short name -------------------- */
    char pname[16];
    snprintf(pname, sizeof pname, "loader-prod-%u", pid);
    prctl(PR_SET_NAME, (unsigned long)pname, 0, 0, 0); // syscall!

    const Config *cfg = &shm->cfg;
    fprintf(stderr, "[producer %u] starting\n", pid);

    /* --- 1. build private URL list ---------------------------------- */
    int n_urls = 0;
    char **urls = split_urls(shm->urls, &n_urls);
    if (!urls || n_urls == 0) {
        fprintf(stderr, "[producer %u] URL list empty – abort\n", pid);
        _exit(EXIT_FAILURE);
    }

    /* --- 2. libcurl session ----------------------------------------- */
    CURL *curl = curl_easy_init();
    if (!curl) xperror("curl_easy_init");

    curl_easy_setopt(curl, CURLOPT_NOSIGNAL,    1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    /* optional bearer token from environment */
    const char *tok = getenv("AUTH");
    struct curl_slist *hdrs = NULL;
    char hdrbuf[1024];
    if (tok) {
        snprintf(hdrbuf, sizeof hdrbuf, "Authorization: Bearer %s", tok);
        hdrs = curl_slist_append(hdrs, hdrbuf);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, hdrs);
    }

    /* --- 3. per-transfer context ------------------------------------ */
    stream_ctx sctx = {
        .shm     = shm,
        .cfg     = cfg,
        .pid     = pid,
        .scratch = scratch_ptr(shm, cfg, pid),
        .fill    = 0,
        .done    = 0
    };

    curl_easy_setopt(curl, CURLOPT_WRITEDATA,    &sctx);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_write_cb);

    /* --- 4. main download loop -------------------------------------- */
    while (!sctx.done) {
        DEBUG_PRINT("[producer %u] entering loop\n", pid);
        uint32_t idx = atomic_fetch_add_explicit(&shm->url_idx, 1,
                                                 memory_order_relaxed);
        if (idx >= (uint32_t)n_urls) break;          /* queue exhausted */

        curl_easy_setopt(curl, CURLOPT_URL, urls[idx]);
        fprintf(stderr, "[producer %u] fetching %u / %u : %s\n",
                pid, idx + 1, n_urls, urls[idx]);

        while (!sctx.done) {
            CURLcode rc = curl_easy_perform(curl);

            if (sctx.done) break;                    /* normal exit   */
            if (rc == CURLE_OK)  break;              /* next URL      */

            fprintf(stderr, "[producer %u] curl: %s – retry\n",
                    pid, curl_easy_strerror(rc));
            usleep(cfg->retry_sleep);                /* brief back-off */ // syscall!
        }
    }

    /* --- 5. cleanup -------------------------------------------------- */
    curl_easy_cleanup(curl);
    curl_slist_free_all(hdrs);
    free(urls[0]);   /* the whole block */
    free(urls);
    atomic_fetch_sub_explicit(&shm->producers_left, 1, memory_order_release);
    
    fprintf(stderr, "[producer %u] exiting\n", pid);
    _exit(EXIT_SUCCESS); // syscall!
}


// -----------------------------------------------------------------------------
//  Master loop - consumer
// -----------------------------------------------------------------------------

// master_loop()
//
// Single-threaded consumer that overlaps
//      * PCIe host→device copy
//      * GPU kernel execution
//      * CPU refill by producers
//
// Triple-buffer dance (steady state)
//  -----------------------------------------------------------------
//    1.  **WAIT  (buf[e % 3] : READY)**
//        Spin on `buf->state` until the last producer sets it to READY.
//
//    2.  **COPY  (buf[e % 3] : READY → IN_GPU)**
//        gpu_copy(idx = e % 3, payload)  ← enqueues async H2D memcpy.
//
//    3.  **ADVANCE consumer pointer**
//        cons_epoch++                    ← opens a slot for producers.
//
//    4.  **RUN kernel + RECYCLE (buf[(e+3) % 3])**
//        gpu_run_kernel((e+3) % 3)       ← waits for kernel of the
//                                          buffer that is *two* epochs behind.
//        zero head/written;  state = FILLING
//                                     ↑
//        producers see this and start refilling immediately.
//
//    5.  **LOOP**    (e++, batches_done++)
//
//  Watch-dogs & termination
//  ------------------------
//    * Every 1024 spins it checks `producers_left`; if all producers have gone,
//      it breaks the main loop early.
//    * Main loop ends when either
//          a)  `batches_done == max_batches`, or
//          b)  `alive == 0`  (all producers exited).
//
//  Atomics & ordering
//  ------------------
//    * Producers own   `prod_epoch`  and set  state = READY.
//    * Master   owns   `cons_epoch`  and sets  state = FILLING.
//    * `memory_order_acquire` when reading READY guarantees payload visible.
//    * `memory_order_release` when writing FILLING guarantees zeroed header
//      visible before producers can claim tickets.
//
//  Overlap picture (NBUF = 3)
//
//      Epoch e-2            Epoch e-1            Epoch e
//   ┌─────────────┐     ┌─────────────┐     ┌───────────────┐
//   │ GPU kernel  │ --> │  PCIe copy  │ --> │  CPU filling  │
//   └─────────────┘     └─────────────┘     └───────────────┘
//
//  Result: one batch in flight on PCIe, one running on GPU,
//          one being refilled — continuously.
//
static int master_loop(SharedHeader *shm, pid_t *pids, uint32_t n_producers)
{
    const Config *cfg = &shm->cfg;
    gpu_init(shm);

    uint32_t batches_done = 0;
    uint32_t alive        = n_producers;

    struct timespec t0;  clock_gettime(CLOCK_MONOTONIC, &t0); // syscall!

    while (batches_done < cfg->max_batches && alive) {

        uint32_t e   = atomic_load_explicit(&shm->cons_epoch,
                                            memory_order_acquire);
        BufferHeader *buf = get_buffer(shm, cfg, e % NBUF);

        /* ---- 1. wait until producers publish BUF_READY ------------- */
        int spins = 0;
        for (;;)
        {
            uint32_t state = atomic_load_explicit(&buf->state,
                                                  memory_order_acquire);

            if (state == BUF_READY)
                break;                             /* good – continue    */

            if (alive == 0)                        /* NEW 1/2 ---------- */
                goto done;                         /* nobody left to     */
                                                    /* make it READY     */

            _mm_pause();

            if (++spins == 1024) {
                alive = atomic_load_explicit(&shm->producers_left,
                                             memory_order_acquire);
                spins = 0;
            }
        }

        /* ---- 2. enqueue PCIe copy ---------------------------------- */
        uint8_t *pay = payload_ptr(buf);
        gpu_copy(e % NBUF, pay, cfg);

        /* ---- 3. open next epoch for producers ---------------------- */
        atomic_fetch_add_explicit(&shm->cons_epoch, 1,
                                  memory_order_acq_rel);

        /* ---- 4. recycle the buffer that is two epochs behind ------- */
        BufferHeader *recycle =
            get_buffer(shm, cfg, (e + NBUF) % NBUF);

        gpu_run_kernel((e + NBUF) % NBUF);         /* wait for kernel    */

        atomic_store_explicit(&recycle->head,    0, memory_order_relaxed);
        atomic_store_explicit(&recycle->written, 0, memory_order_relaxed);
        atomic_store_explicit(&recycle->state,   BUF_FILLING,
                              memory_order_release);

        ++batches_done;
    }

    done: {
        /* ---- 5.  throughput print ------------------------------------- */
        struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1); // syscall!
        double sec = (t1.tv_sec - t0.tv_sec) +
                    (t1.tv_nsec - t0.tv_nsec) / 1e9;

        if (batches_done == 0) {
            fprintf(stderr, "master_loop: exited before any batch completed\n");
            return 0;
        }

        uint64_t bytes = (uint64_t)batches_done * cfg->batch_bytes;
        double   thr   = bytes / (sec * (double)(1ULL << 20));

        printf("Throughput %.3f MiB/s (%u batches in %.2f s)\n",
            thr, batches_done, sec);
        return thr;
    }
}

// -----------------------------------------------------------------------------
//  Relay child
// -----------------------------------------------------------------------------

// A helper process that relays the output of the producer processes to stdout.
static void relay_child_output(int pfds[][2], int n_producers)
{
    prctl(PR_SET_NAME, (unsigned long)"loader-relay", 0, 0, 0); // syscall!

    struct pollfd fds[MAX_PRODUCERS];
    char   linebuf[MAX_PRODUCERS][TAG_BUFSZ] = {{0}};
    size_t lenbuf[MAX_PRODUCERS]             = {0};

    /* Build poll() array with read-ends only */
    for (int i = 0; i < n_producers; ++i) {
        fds[i].fd = pfds[i][0];
        fds[i].events = POLLIN;
    }


    int open_fds = n_producers;

    while (1) {
        if (open_fds == 0)
            break;                      /* --- nothing left to watch — exit */

        if (poll(fds, n_producers, -1) < 0) { // syscall!
            if (errno == EINTR) continue;
            perror("poll"); break; // syscall!
        }

        for (int p = 0; p < n_producers; ++p) {
            if (fds[p].fd < 0) continue;            /* already closed   */
            if (!(fds[p].revents & (POLLIN | POLLHUP))) continue;


            char buf[4096];
            ssize_t n = read(fds[p].fd, buf, sizeof buf); // syscall!
            if (n <= 0) {               /* EOF or error → stop watching     */
                close(fds[p].fd); // syscall!
                fds[p].fd = -1;
                --open_fds;
                continue;
            }

            /* Assemble complete lines */
            for (ssize_t i = 0; i < n; ++i) {
                char c = buf[i];
                linebuf[p][lenbuf[p]++] = c;

                if (c == '\n' || lenbuf[p] == TAG_BUFSZ-1) {
                    linebuf[p][lenbuf[p]] = 0;
                    fprintf(stdout, "%s", linebuf[p]);
                    lenbuf[p] = 0;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
//  Main
// -----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    prctl(PR_SET_NAME, (unsigned long)"loader-master", 0, 0, 0); // syscall!

    Config cfg = parse_config(argc, argv);

    uint64_t expected_bytes = (uint64_t)cfg.max_batches * cfg.batch_bytes;
    fprintf(stdout, "DataLoader. Expected to produce %.4f MiB of data\n", 
            (double) expected_bytes / (1 << 20));

    fprintf(stdout, "---------- Configuration ----------\n");
    fprintf(stdout, "Window length       : %u samples\n", cfg.window_len);
    fprintf(stdout, "Feature size        : %u features/sample\n", cfg.feat_size);
    fprintf(stdout, "Scalar size         : %u bytes (e.g. fp16=2, fp32=4)\n", cfg.float_size);
    fprintf(stdout, "Batch size          : %u\n", cfg.batch_size);
    fprintf(stdout, "Producer processes  : %u\n", cfg.n_producers);
    fprintf(stdout, "Max batches         : %u\n", cfg.max_batches);
    fprintf(stdout, "Use hugepages       : %u\n", cfg.use_hugepages);
    fprintf(stdout, "Dry run             : %u\n", cfg.dry_run);
    fprintf(stdout, "Local rank          : %d\n", cfg.local_rank);
    fprintf(stdout, "----------- Inferred -------------\n");
    fprintf(stdout, "Slot size           : %.2f KiB\n", (double)cfg.slot_bytes / 1024);
    fprintf(stdout, "Payload per batch   : %.2f MiB (%lu bytes)\n",
           (double)cfg.batch_bytes / (1024 * 1024), cfg.batch_bytes);
    fprintf(stdout, "Max data to download: %.4f GiB\n",
           (double)cfg.max_batches * cfg.batch_bytes / (1024.0 * 1024 * 1024));
    fprintf(stdout, "Total memory alloc  : %.2f MiB (triple buffer + headers)\n",
           (sizeof(SharedHeader) + NBUF * buffer_bytes(&cfg)) / (1024.0 * 1024));
    fprintf(stdout, "-----------------------------------\n");

    // Allocate memory according to the configuration.
    SharedHeader* shm = create_pinned_hugetlb_shm(&cfg);

    // Copy URL list from env → shared blob
    FILE *fp = fopen("data/urls.txt", "r"); // syscall!
    if (!fp) {
        perror("Failed to open URL file"); // syscall!
        return EXIT_FAILURE;
    }
    size_t total_read = fread(shm->urls, 1, sizeof(shm->urls) - 1, fp); // syscall!
    if (ferror(fp)) {
        perror("Error reading URL file"); // syscall!
        fclose(fp); // syscall!
        return EXIT_FAILURE;
    }
    fclose(fp); // syscall!
    if (total_read == 0) {
        fprintf(stderr, "URL file is empty – exiting\n");
        return EXIT_FAILURE;
    }
    shm->urls[total_read] = '\0';
    // Count URLs once
    uint32_t cnt = 1; 
    for (const char *p = shm->urls; *p; ++p) if (*p == '\n') cnt++;
    shm->url_count = cnt;

    // Create pipes for producer processes
    int pfds[MAX_PRODUCERS][2];
    pid_t producer_pids[MAX_PRODUCERS] = {0};

    for (int p = 0; p < cfg.n_producers; ++p) {
        if (pipe(pfds[p]) < 0) xperror("pipe"); // syscall!
    }

    // Fork producer processes
    for (int p = 0; p < cfg.n_producers; ++p) {
        pid_t pid = fork(); // syscall!
        if (pid == 0) {                   /* --- child --- */
            /* Make stdout/stderr line-buffered so every \n flushes */
            setvbuf(stdout, NULL, _IOLBF, 0); // syscall!
            setvbuf(stderr, NULL, _IOLBF, 0); // syscall!

            /* close read-end, dup write-end onto 1 and 2 */
            close(pfds[p][0]); // syscall!
            dup2(pfds[p][1], STDOUT_FILENO); // syscall!
            dup2(pfds[p][1], STDERR_FILENO); // syscall!
            for (int i = 0; i < cfg.n_producers; ++i) close(pfds[i][1]); // syscall!

            if (cfg.dry_run == 0) {
                producer_loop(shm, p);
            } else {
                producer_loop_dry_run(shm, p);
            }
            _exit(0); // syscall!
        } else if (pid > 0) {             /* --- parent --- */
            close(pfds[p][1]);            /* parent uses only read-end */ // syscall!
            producer_pids[p] = pid;
        } else {
            xperror("fork"); // syscall!
        }
    }

    // Create a helper process that relays the output of the producer processes 
    // to stdout. This is used to avoid deadlocks if the master loop is blocked.
    pid_t relay_pid = fork(); // syscall!
    if (relay_pid == 0) {                 /* --- relay child --- */
        relay_child_output(pfds, cfg.n_producers);
        _exit(0); // syscall!
    }

    // Original parent continues to be the master
    double throughput = master_loop(shm, producer_pids, cfg.n_producers);
    fprintf(stdout, "[final] throughput: %.4f\n", throughput);

    // Make sure the relay is gone before we leave
    kill(relay_pid, SIGTERM); // syscall!
    waitpid(relay_pid, NULL, 0); // syscall!

    return 0;
}
