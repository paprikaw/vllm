#include <nccl.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#define CHECK_CUDA(cmd)                                                     \
  do {                                                                      \
    cudaError_t e = (cmd);                                                  \
    if (e != cudaSuccess) {                                                 \
      fprintf(stderr, "[%s:%d] CUDA error %s: %s\n", __FILE__, __LINE__,    \
              #cmd, cudaGetErrorString(e));                                \
      exit(2);                                                              \
    }                                                                       \
  } while (0)

#define CHECK_NCCL(cmd)                                                     \
  do {                                                                      \
    ncclResult_t r = (cmd);                                                 \
    if (r != ncclSuccess) {                                                 \
      fprintf(stderr, "[%s:%d] NCCL error %s: %s\n", __FILE__, __LINE__,    \
              #cmd, ncclGetErrorString(r));                                \
      exit(3);                                                              \
    }                                                                       \
  } while (0)

static double now_ms() {
  using clock = std::chrono::steady_clock;
  static const auto t0 = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static std::string join_path(const std::string &a, const std::string &b) {
  return a + "/" + b;
}

static bool exists(const std::string &path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static void wait_for_file(const std::string &path) {
  while (!exists(path)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

static void write_blob_atomic(const std::string &path, const void *data, size_t n) {
  std::string tmp = path + ".tmp." + std::to_string(getpid());
  int fd = open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
  if (fd < 0) {
    perror("open");
    exit(4);
  }
  const char *p = static_cast<const char *>(data);
  size_t off = 0;
  while (off < n) {
    ssize_t w = write(fd, p + off, n - off);
    if (w < 0) {
      perror("write");
      exit(4);
    }
    off += static_cast<size_t>(w);
  }
  close(fd);
  if (rename(tmp.c_str(), path.c_str()) != 0) {
    perror("rename");
    exit(4);
  }
}

static void read_blob(const std::string &path, void *data, size_t n) {
  wait_for_file(path);
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    perror("open read");
    exit(4);
  }
  char *p = static_cast<char *>(data);
  size_t off = 0;
  while (off < n) {
    ssize_t r = read(fd, p + off, n - off);
    if (r <= 0) {
      perror("read");
      exit(4);
    }
    off += static_cast<size_t>(r);
  }
  close(fd);
}

static void touch_file(const std::string &path) {
  int fd = open(path.c_str(), O_CREAT | O_WRONLY, 0600);
  if (fd >= 0) close(fd);
}

static void barrier(const std::string &dir, const char *name, int rank, int count) {
  touch_file(join_path(dir, std::string(name) + "." + std::to_string(rank)));
  for (int r = 0; r < count; ++r) {
    wait_for_file(join_path(dir, std::string(name) + "." + std::to_string(r)));
  }
}

static void log_line(int rank, const char *msg) {
  char host[256] = {0};
  gethostname(host, sizeof(host) - 1);
  printf("rank=%d host=%s t=%.3fms %s\n", rank, host, now_ms(), msg);
  fflush(stdout);
}

static void allreduce_check(ncclComm_t comm, cudaStream_t stream, int rank,
                            float input, float expected, const char *label) {
  float *d_send = nullptr;
  float *d_recv = nullptr;
  float h_recv = 0.0f;
  CHECK_CUDA(cudaMalloc(&d_send, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_recv, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_send, &input, sizeof(float), cudaMemcpyHostToDevice));
  double start = now_ms();
  CHECK_NCCL(ncclAllReduce(d_send, d_recv, 1, ncclFloat, ncclSum, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  double elapsed = now_ms() - start;
  CHECK_CUDA(cudaMemcpy(&h_recv, d_recv, sizeof(float), cudaMemcpyDeviceToHost));
  char buf[256];
  snprintf(buf, sizeof(buf),
           "%s allreduce result=%.1f expected=%.1f elapsed_ms=%.3f",
           label, h_recv, expected, elapsed);
  log_line(rank, buf);
  if (h_recv != expected) exit(5);
  CHECK_CUDA(cudaFree(d_send));
  CHECK_CUDA(cudaFree(d_recv));
}

static int arg_int(int argc, char **argv, const char *name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == name) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

static std::string arg_str(int argc, char **argv, const char *name,
                           const std::string &fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == name) return argv[i + 1];
  }
  return fallback;
}

int main(int argc, char **argv) {
  int rank = arg_int(argc, argv, "--rank", -1);
  int local_device = arg_int(argc, argv, "--local-device", -1);
  int world_size = arg_int(argc, argv, "--world-size", 8);
  int parent_size = arg_int(argc, argv, "--parent-size", 4);
  std::string run_dir = arg_str(argc, argv, "--run-dir", "");
  std::string mode = arg_str(argc, argv, "--mode", "grow-shrink");
  if (rank < 0 || local_device < 0 || run_dir.empty()) {
    fprintf(stderr, "usage: %s --rank R --local-device D --world-size 8 --parent-size 4 --run-dir DIR\n", argv[0]);
    return 1;
  }

  int version = 0;
  CHECK_NCCL(ncclGetVersion(&version));
  if (version < 23000) {
    fprintf(stderr, "rank %d: need NCCL 2.30+, got %d\n", rank, version);
    return 1;
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (local_device >= device_count) {
    fprintf(stderr, "rank %d: local device %d >= visible device count %d\n",
            rank, local_device, device_count);
    return 1;
  }
  CHECK_CUDA(cudaSetDevice(local_device));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  char start_msg[256];
  snprintf(start_msg, sizeof(start_msg), "start local_device=%d nccl_version=%d",
           local_device, version);
  log_line(rank, start_msg);

  ncclComm_t parent = nullptr;
  ncclComm_t grown = nullptr;
  ncclComm_t shrunk = nullptr;

  if (mode == "init8") {
    std::string uid_path = join_path(run_dir, "init8.uid");
    ncclUniqueId uid;
    if (rank == 0) {
      CHECK_NCCL(ncclGetUniqueId(&uid));
      write_blob_atomic(uid_path, &uid, sizeof(uid));
      log_line(rank, "published init8 unique id");
    } else {
      read_blob(uid_path, &uid, sizeof(uid));
    }
    barrier(run_dir, "before_init8", rank, world_size);
    double start = now_ms();
    CHECK_NCCL(ncclCommInitRank(&grown, world_size, uid, rank));
    char buf[160];
    snprintf(buf, sizeof(buf), "init8 completed elapsed_ms=%.3f", now_ms() - start);
    log_line(rank, buf);
    allreduce_check(grown, stream, rank, static_cast<float>(rank + 1), 36.0f,
                    "init8");
    barrier(run_dir, "final_done", rank, world_size);
    CHECK_NCCL(ncclCommDestroy(grown));
    CHECK_CUDA(cudaStreamDestroy(stream));
    log_line(rank, "done");
    return 0;
  }

  std::string initial_uid_path = join_path(run_dir, "initial.uid");
  ncclUniqueId initial_id;
  if (rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&initial_id));
    write_blob_atomic(initial_uid_path, &initial_id, sizeof(initial_id));
    log_line(rank, "published initial unique id");
  } else {
    read_blob(initial_uid_path, &initial_id, sizeof(initial_id));
  }

  if (rank < parent_size) {
    double start = now_ms();
    CHECK_NCCL(ncclCommInitRank(&parent, parent_size, initial_id, rank));
    char buf[128];
    snprintf(buf, sizeof(buf), "parent init completed elapsed_ms=%.3f",
             now_ms() - start);
    log_line(rank, buf);
    allreduce_check(parent, stream, rank, static_cast<float>(rank + 1), 10.0f,
                    "parent4");
  } else {
    log_line(rank, "not part of parent communicator");
  }

  barrier(run_dir, "parent_phase_done", rank, world_size);

  std::string grow_uid_path = join_path(run_dir, "grow.uid");
  ncclUniqueId grow_id;
  if (rank == 0) {
    CHECK_NCCL(ncclCommGetUniqueId(parent, &grow_id));
    write_blob_atomic(grow_uid_path, &grow_id, sizeof(grow_id));
    log_line(rank, "published grow unique id");
  } else {
    read_blob(grow_uid_path, &grow_id, sizeof(grow_id));
  }

  barrier(run_dir, "before_grow", rank, world_size);
  double grow_start = now_ms();
  if (rank < parent_size) {
    const ncclUniqueId *uid_arg = (rank == 0) ? &grow_id : nullptr;
    CHECK_NCCL(ncclCommGrow(parent, world_size, uid_arg, -1, &grown, nullptr));
  } else {
    CHECK_NCCL(ncclCommGrow(nullptr, world_size, &grow_id, rank, &grown, nullptr));
  }
  char grow_msg[128];
  snprintf(grow_msg, sizeof(grow_msg), "grow8 completed elapsed_ms=%.3f",
           now_ms() - grow_start);
  log_line(rank, grow_msg);

  allreduce_check(grown, stream, rank, static_cast<float>(rank + 1), 36.0f,
                  "grown8");
  barrier(run_dir, "grown_phase_done", rank, world_size);

  if (rank < parent_size) {
    int exclude[4] = {4, 5, 6, 7};
    double shrink_start = now_ms();
    CHECK_NCCL(ncclCommShrink(grown, exclude, 4, &shrunk, nullptr,
                              NCCL_SHRINK_DEFAULT));
    char shrink_msg[128];
    snprintf(shrink_msg, sizeof(shrink_msg), "shrink4 completed elapsed_ms=%.3f",
             now_ms() - shrink_start);
    log_line(rank, shrink_msg);
    allreduce_check(shrunk, stream, rank, static_cast<float>(rank + 1), 10.0f,
                    "shrunk4");
  } else {
    log_line(rank, "excluded from shrink");
  }

  barrier(run_dir, "final_done", rank, world_size);
  if (shrunk) CHECK_NCCL(ncclCommDestroy(shrunk));
  if (grown) CHECK_NCCL(ncclCommDestroy(grown));
  if (parent) CHECK_NCCL(ncclCommDestroy(parent));
  CHECK_CUDA(cudaStreamDestroy(stream));
  log_line(rank, "done");
  return 0;
}
