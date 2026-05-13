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
#include <sys/types.h>
#include <sys/wait.h>
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
  auto dt = std::chrono::duration<double, std::milli>(clock::now() - t0);
  return dt.count();
}

static void log_rank(int rank, const char *msg) {
  fprintf(stdout, "[%9.3f ms][pid=%d][rank=%d] %s\n",
          now_ms(), getpid(), rank, msg);
  fflush(stdout);
}

static std::string path_join(const std::string &a, const std::string &b) {
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
    perror("open tmp");
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

static void file_barrier(const std::string &dir, const char *name,
                         int rank, int nranks) {
  touch_file(path_join(dir, std::string(name) + "." + std::to_string(rank)));
  for (int r = 0; r < nranks; ++r) {
    wait_for_file(path_join(dir, std::string(name) + "." + std::to_string(r)));
  }
}

static void allreduce_check(ncclComm_t comm, cudaStream_t stream, int rank,
                            float input, float expected, const char *label) {
  float *d_send = nullptr;
  float *d_recv = nullptr;
  float h_recv = 0.0f;
  CHECK_CUDA(cudaMalloc(&d_send, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_recv, sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_send, &input, sizeof(float), cudaMemcpyHostToDevice));
  CHECK_NCCL(ncclAllReduce(d_send, d_recv, 1, ncclFloat, ncclSum, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaMemcpy(&h_recv, d_recv, sizeof(float), cudaMemcpyDeviceToHost));
  char buf[256];
  snprintf(buf, sizeof(buf), "%s allreduce result %.1f expected %.1f",
           label, h_recv, expected);
  log_rank(rank, buf);
  if (h_recv != expected) {
    fprintf(stderr, "rank %d: %s mismatch\n", rank, label);
    exit(5);
  }
  CHECK_CUDA(cudaFree(d_send));
  CHECK_CUDA(cudaFree(d_recv));
}

static void child_main(int rank, int initial_ranks, int grown_ranks,
                       const std::string &run_dir, ncclUniqueId initial_id) {
  int visible_devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&visible_devices));
  if (visible_devices <= rank) {
    fprintf(stderr, "rank %d needs GPU %d but only %d GPUs are visible\n",
            rank, rank, visible_devices);
    exit(1);
  }
  CHECK_CUDA(cudaSetDevice(rank));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  ncclComm_t parent = nullptr;
  ncclComm_t grown = nullptr;
  ncclComm_t shrunk = nullptr;

  if (rank < initial_ranks) {
    log_rank(rank, "initializing parent communicator");
    CHECK_NCCL(ncclCommInitRank(&parent, initial_ranks, initial_id, rank));
    allreduce_check(parent, stream, rank, static_cast<float>(rank + 1),
                    3.0f, "parent");
    file_barrier(run_dir, "parent_done", rank, initial_ranks);
  } else {
    log_rank(rank, "not part of parent communicator");
    wait_for_file(path_join(run_dir, "parent_done.0"));
  }

  ncclUniqueId grow_id;
  std::string grow_path = path_join(run_dir, "grow.uid");
  if (rank == 0) {
    CHECK_NCCL(ncclCommGetUniqueId(parent, &grow_id));
    write_blob_atomic(grow_path, &grow_id, sizeof(grow_id));
    log_rank(rank, "published grow unique id");
  } else {
    read_blob(grow_path, &grow_id, sizeof(grow_id));
  }

  file_barrier(run_dir, "before_grow", rank, grown_ranks);
  double grow_start = now_ms();
  if (rank < initial_ranks) {
    const ncclUniqueId *uid_arg = (rank == 0) ? &grow_id : nullptr;
    CHECK_NCCL(ncclCommGrow(parent, grown_ranks, uid_arg, -1, &grown, nullptr));
  } else {
    CHECK_NCCL(ncclCommGrow(nullptr, grown_ranks, &grow_id, rank, &grown, nullptr));
  }
  char grow_msg[128];
  snprintf(grow_msg, sizeof(grow_msg), "ncclCommGrow completed in %.3f ms",
           now_ms() - grow_start);
  log_rank(rank, grow_msg);

  allreduce_check(grown, stream, rank, static_cast<float>(rank + 1),
                  6.0f, "grown");
  file_barrier(run_dir, "grown_done", rank, grown_ranks);

  if (rank < initial_ranks) {
    int exclude[] = {2};
    double shrink_start = now_ms();
    CHECK_NCCL(ncclCommShrink(grown, exclude, 1, &shrunk, nullptr,
                              NCCL_SHRINK_DEFAULT));
    char shrink_msg[128];
    snprintf(shrink_msg, sizeof(shrink_msg),
             "ncclCommShrink completed in %.3f ms", now_ms() - shrink_start);
    log_rank(rank, shrink_msg);

    allreduce_check(shrunk, stream, rank, static_cast<float>(rank + 1),
                    3.0f, "shrunk");
    touch_file(path_join(run_dir, "shrink_done." + std::to_string(rank)));
  } else {
    log_rank(rank, "excluded from shrink; waiting for survivors");
    wait_for_file(path_join(run_dir, "shrink_done.0"));
    wait_for_file(path_join(run_dir, "shrink_done.1"));
  }

  if (shrunk) CHECK_NCCL(ncclCommDestroy(shrunk));
  if (grown) CHECK_NCCL(ncclCommDestroy(grown));
  if (parent) CHECK_NCCL(ncclCommDestroy(parent));
  CHECK_CUDA(cudaStreamDestroy(stream));
  log_rank(rank, "done");
}

int main(int argc, char **argv) {
  int nccl_version = 0;
  CHECK_NCCL(ncclGetVersion(&nccl_version));
  printf("Using NCCL version int: %d\n", nccl_version);
  if (nccl_version < 23000) {
    fprintf(stderr, "This demo requires NCCL 2.30+ for ncclCommGrow.\n");
    return 1;
  }

  std::string run_dir = "/tmp/nccl_grow_shrink_" + std::to_string(getpid());
  if (argc > 1) run_dir = argv[1];
  if (mkdir(run_dir.c_str(), 0700) != 0 && errno != EEXIST) {
    perror("mkdir run_dir");
    return 1;
  }
  printf("Run dir: %s\n", run_dir.c_str());
  fflush(stdout);

  ncclUniqueId initial_id;
  CHECK_NCCL(ncclGetUniqueId(&initial_id));
  fflush(stdout);

  const int initial_ranks = 2;
  const int grown_ranks = 3;
  pid_t pids[grown_ranks];
  for (int rank = 0; rank < grown_ranks; ++rank) {
    pid_t pid = fork();
    if (pid < 0) {
      perror("fork");
      return 1;
    }
    if (pid == 0) {
      child_main(rank, initial_ranks, grown_ranks, run_dir, initial_id);
      return 0;
    }
    pids[rank] = pid;
  }

  int rc = 0;
  for (int i = 0; i < grown_ranks; ++i) {
    int status = 0;
    waitpid(pids[i], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      rc = 1;
      fprintf(stderr, "child pid %d failed with status %d\n", pids[i], status);
    }
  }
  if (rc == 0) {
    printf("PASS: NCCL grow and shrink demo completed.\n");
  }
  return rc;
}
