// Fast DCT/DST without an FFT: real radix-2 recursions down to odd sizes,
// which are summed directly.
//
// Five kernels cover all sixteen transforms in orthel.transforms.dct:
//
//   dct1   X_k = x_0/2 + (-1)^k x_{n-1}/2 + sum_{0<j<n-1} x_j cos(pi k j / (n-1))
//   dst1   X_k = sum_j x_j sin(pi (k+1)(j+1) / (n+1))
//   dct2   X_k = sum_j x_j cos(pi k (2j+1) / 2n)
//   idct2  the inverse of dct2
//   dct4   X_k = sum_j x_j cos(pi (2k+1)(2j+1) / 4n)
//
// The rest are these with flipped or sign-alternated inputs and outputs and
// a scale factor (see `recipe`). Even sizes split in half:
//
//   dct2(2m)    dct2(m) of x_j + x_{n-1-j} gives the even outputs and
//               dct4(m) of x_j - x_{n-1-j} the odd ones
//   idct2(2m)   the same steps in reverse, with idct2(m) and dct4(m)
//   dct4(2m)    rotate (x_j, x_{n-1-j}) by pi (2j+1) / 4n, then dct2(m) of
//               each half (the second as a DST-II) and one butterfly
//   dct1(2m+1)  dct1(m+1) of the symmetric part and a DCT-III(m), that is
//               idct2(m), of the rest
//   dst1(2m-1)  dst1(m-1) of the antisymmetric part and a DST-III(m), from
//               idct2(m), of the rest
//
// Every split is orthogonal up to scale, so rounding errors grow like
// O(log n). Lines are transformed kLanes at a time: a block is n lane
// vectors, vector j holding element j of every line, so each step of a
// kernel is whole-vector arithmetic.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace orthel {

enum class Kernel { Dct1, Dst1, Dct2, Idct2, Dct4 };

constexpr std::size_t kLanes = 8;
// Direct sums keep an n x n table up to this size and recompute entries above.
constexpr std::size_t kMaxTable = 256;

namespace detail {

// kLanes values of T, with elementwise arithmetic and scalar broadcasting.
#if defined(__GNUC__) || defined(__clang__)
template <class T>
struct VecOf;
template <>
struct VecOf<float> {
  typedef float type __attribute__((vector_size(kLanes * sizeof(float))));
};
template <>
struct VecOf<double> {
  typedef double type __attribute__((vector_size(kLanes * sizeof(double))));
};
template <class T>
using Vec = typename VecOf<T>::type;
#else
template <class T>
struct Vec {
  T v[kLanes];
  T& operator[](std::size_t l) { return v[l]; }
  const T& operator[](std::size_t l) const { return v[l]; }
#define ORTHEL_VEC_OP(op)                                                  \
  friend Vec operator op(Vec a, const Vec& b) {                            \
    for (std::size_t l = 0; l < kLanes; ++l) a.v[l] op## = b.v[l];         \
    return a;                                                              \
  }                                                                        \
  friend Vec operator op(Vec a, T b) {                                     \
    for (std::size_t l = 0; l < kLanes; ++l) a.v[l] op## = b;              \
    return a;                                                              \
  }                                                                        \
  friend Vec operator op(T a, Vec b) {                                     \
    for (std::size_t l = 0; l < kLanes; ++l) b.v[l] = a op b.v[l];         \
    return b;                                                              \
  }
  ORTHEL_VEC_OP(+)
  ORTHEL_VEC_OP(-)
  ORTHEL_VEC_OP(*)
#undef ORTHEL_VEC_OP
  friend Vec operator-(Vec a) {
    for (std::size_t l = 0; l < kLanes; ++l) a.v[l] = -a.v[l];
    return a;
  }
};
#endif

constexpr double kPi = 3.14159265358979323846;

// cos(pi p / q) and sin(pi p / q), with p reduced exactly modulo 2q first.
inline double cos_pi(std::int64_t p, std::int64_t q) {
  std::int64_t r = p % (2 * q);
  return std::cos(kPi * static_cast<double>(r) / static_cast<double>(q));
}
inline double sin_pi(std::int64_t p, std::int64_t q) {
  std::int64_t r = p % (2 * q);
  return std::sin(kPi * static_cast<double>(r) / static_cast<double>(q));
}

// Entry (row, col) of a kernel's n x n matrix: output row, input col.
inline double entry(Kernel kind, std::int64_t n, std::int64_t row,
                    std::int64_t col) {
  switch (kind) {
    case Kernel::Dct2:
      return cos_pi(row * (2 * col + 1), 2 * n);
    case Kernel::Idct2:
      return (col == 0 ? 1.0 : 2.0) / static_cast<double>(n) *
             cos_pi(col * (2 * row + 1), 2 * n);
    case Kernel::Dct4:
      return cos_pi((2 * row + 1) * (2 * col + 1), 4 * n);
    case Kernel::Dct1:
      return (col == 0 || col == n - 1 ? 0.5 : 1.0) * cos_pi(row * col, n - 1);
    case Kernel::Dst1:
      return sin_pi((row + 1) * (col + 1), n + 1);
  }
  return 0.0;
}

// Whether a kernel of length n splits in half, as described above.
inline bool splits(Kernel kind, std::size_t n) {
  switch (kind) {
    case Kernel::Dct2:
    case Kernel::Idct2:
    case Kernel::Dct4:
      return n >= 2 && n % 2 == 0;
    case Kernel::Dct1:  // n = 2m + 1 with m >= 1
    case Kernel::Dst1:  // n = 2m - 1 with m >= 2
      return n >= 3 && n % 2 == 1;
  }
  return false;
}

}  // namespace detail

// One node of a plan: either a direct sum or a split into child plans.
template <class T>
struct Plan {
  Kernel kind;
  std::size_t n = 0;
  bool direct = false;
  std::vector<T> table;   // direct: n x n, empty when n > kMaxTable
  std::vector<T> cs, sn;  // dct4: the rotation by pi (2j+1) / 4n
  std::shared_ptr<const Plan> a, b;
  std::size_t work = 0;        // scratch vectors needed by run()
  std::size_t max_direct = 0;  // largest direct sum in the tree
};

template <class T>
std::shared_ptr<const Plan<T>> get_plan(Kernel kind, std::size_t n);

namespace detail {

template <class T>
std::shared_ptr<const Plan<T>> build_plan(Kernel kind, std::size_t n) {
  auto p = std::make_shared<Plan<T>>();
  p->kind = kind;
  p->n = n;
  if (!splits(kind, n)) {
    p->direct = true;
    p->work = n;
    p->max_direct = n;
    if (n <= kMaxTable) {
      p->table.resize(n * n);
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
          p->table[r * n + c] = static_cast<T>(entry(kind, n, r, c));
    }
    return p;
  }
  switch (kind) {
    case Kernel::Dct2:
      p->a = get_plan<T>(Kernel::Dct2, n / 2);
      p->b = get_plan<T>(Kernel::Dct4, n / 2);
      break;
    case Kernel::Idct2:
      p->a = get_plan<T>(Kernel::Idct2, n / 2);
      p->b = get_plan<T>(Kernel::Dct4, n / 2);
      break;
    case Kernel::Dct4: {
      std::size_t m = n / 2;
      p->a = get_plan<T>(Kernel::Dct2, m);
      p->cs.resize(m);
      p->sn.resize(m);
      for (std::size_t j = 0; j < m; ++j) {
        p->cs[j] = static_cast<T>(cos_pi(2 * j + 1, 4 * n));
        p->sn[j] = static_cast<T>(sin_pi(2 * j + 1, 4 * n));
      }
      break;
    }
    case Kernel::Dct1:
      p->a = get_plan<T>(Kernel::Dct1, (n - 1) / 2 + 1);
      p->b = get_plan<T>(Kernel::Idct2, (n - 1) / 2);
      break;
    case Kernel::Dst1:
      p->a = get_plan<T>(Kernel::Dst1, (n + 1) / 2 - 1);
      p->b = get_plan<T>(Kernel::Idct2, (n + 1) / 2);
      break;
  }
  std::size_t child_work = p->a->work;
  p->max_direct = p->a->max_direct;
  if (p->b) {
    child_work = std::max(child_work, p->b->work);
    p->max_direct = std::max(p->max_direct, p->b->max_direct);
  }
  p->work = n + child_work;
  return p;
}

// Transform the block x (n vectors) in place, using w for scratch.
template <class T>
void run(const Plan<T>& p, Vec<T>* x, Vec<T>* w);

template <class T>
void run_direct(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  const std::size_t n = p.n;
  if (n == 1) {
    x[0] = p.table[0] * x[0];
    return;
  }
  std::copy(x, x + n, w);
  for (std::size_t r = 0; r < n; ++r) {
    Vec<T> acc = T(0) * w[0];
    for (std::size_t c = 0; c < n; ++c) {
      const T m = p.table.empty() ? static_cast<T>(entry(p.kind, n, r, c))
                                  : p.table[r * n + c];
      acc = acc + m * w[c];
    }
    x[r] = acc;
  }
}

template <class T>
void run_dct2(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  const std::size_t n = p.n, m = n / 2;
  Vec<T>* u = w;
  Vec<T>* v = w + m;
  for (std::size_t j = 0; j < m; ++j) {
    u[j] = x[j] + x[n - 1 - j];
    v[j] = x[j] - x[n - 1 - j];
  }
  run(*p.a, u, w + n);
  run(*p.b, v, w + n);
  for (std::size_t q = 0; q < m; ++q) {
    x[2 * q] = u[q];
    x[2 * q + 1] = v[q];
  }
}

template <class T>
void run_idct2(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  const std::size_t n = p.n, m = n / 2;
  Vec<T>* u = w;
  Vec<T>* v = w + m;
  for (std::size_t q = 0; q < m; ++q) {
    u[q] = x[2 * q];
    v[q] = x[2 * q + 1];
  }
  run(*p.a, u, w + n);
  run(*p.b, v, w + n);
  // x_j = (u_j + v_j) / 2 and x_{n-1-j} = (u_j - v_j) / 2, with
  // v = (2 / m) dct4(odd outputs).
  const T half = T(0.5), s = T(1) / static_cast<T>(m);
  for (std::size_t j = 0; j < m; ++j) {
    x[j] = half * u[j] + s * v[j];
    x[n - 1 - j] = half * u[j] - s * v[j];
  }
}

template <class T>
void run_dct4(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  const std::size_t n = p.n, m = n / 2;
  Vec<T>* a = w;
  Vec<T>* b = w + m;
  for (std::size_t j = 0; j < m; ++j) {
    const T c = p.cs[j], s = p.sn[j];
    // b is negated on odd rows so that dct2(b) gives the DST-II, reversed.
    const T sign = (j % 2) ? T(-1) : T(1);
    a[j] = c * x[j] + s * x[n - 1 - j];
    b[j] = (sign * c) * x[n - 1 - j] - (sign * s) * x[j];
  }
  run(*p.a, a, w + n);
  run(*p.a, b, w + n);
  // With C = dct2(a) and D = dct2(alt b):
  //   X_{2q} = C_q + D_{m-q},   X_{n-1-2q} = C_{m-q} - D_q,
  // where C_m = 0 and D_m = 0 stand for the missing terms at q = 0.
  x[0] = a[0];
  x[n - 1] = -b[0];
  for (std::size_t q = 1; q < m; ++q) {
    x[2 * q] = a[q] + b[m - q];
    x[n - 1 - 2 * q] = a[m - q] - b[q];
  }
}

template <class T>
void run_dct1(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  // n = N + 1 with N = 2m. Even outputs: dct1(m+1) of u; odd outputs:
  // DCT-III(m) of v, which is (m / 2) idct2(m).
  const std::size_t n = p.n, N = n - 1, m = N / 2;
  Vec<T>* u = w;
  Vec<T>* v = w + m + 1;
  for (std::size_t j = 0; j < m; ++j) {
    u[j] = x[j] + x[N - j];
    v[j] = x[j] - x[N - j];
  }
  u[m] = T(2) * x[m];
  run(*p.a, u, w + n);
  run(*p.b, v, w + n);
  const T h = static_cast<T>(m) / T(2);
  for (std::size_t q = 0; q <= m; ++q) x[2 * q] = u[q];
  for (std::size_t q = 0; q < m; ++q) x[2 * q + 1] = h * v[q];
}

template <class T>
void run_dst1(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  // n = N - 1 with N = 2m. Odd outputs: dst1(m-1) of r; even outputs:
  // DST-III(m) of the symmetric part, computed as alternating signs times
  // (m / 2) idct2(m) of that part reversed, f.
  const std::size_t n = p.n, m = (n + 1) / 2;
  Vec<T>* f = w;
  Vec<T>* r = w + m;
  for (std::size_t i = 0; i < m; ++i) f[i] = x[m - 1 - i] + x[m - 1 + i];
  for (std::size_t t = 0; t + 1 < m; ++t) r[t] = x[t] - x[n - 1 - t];
  run(*p.b, f, w + n);
  run(*p.a, r, w + n);
  const T h = static_cast<T>(m) / T(2);
  for (std::size_t q = 0; q < m; ++q) x[2 * q] = ((q % 2) ? -h : h) * f[q];
  for (std::size_t q = 0; q + 1 < m; ++q) x[2 * q + 1] = r[q];
}

template <class T>
void run(const Plan<T>& p, Vec<T>* x, Vec<T>* w) {
  if (p.direct) return run_direct(p, x, w);
  switch (p.kind) {
    case Kernel::Dct2:
      return run_dct2(p, x, w);
    case Kernel::Idct2:
      return run_idct2(p, x, w);
    case Kernel::Dct4:
      return run_dct4(p, x, w);
    case Kernel::Dct1:
      return run_dct1(p, x, w);
    case Kernel::Dst1:
      return run_dst1(p, x, w);
  }
}

}  // namespace detail

// Plans are built once per (kernel, length, type) and shared.
template <class T>
std::shared_ptr<const Plan<T>> get_plan(Kernel kind, std::size_t n) {
  static std::recursive_mutex mutex;
  static std::map<std::pair<int, std::size_t>, std::shared_ptr<const Plan<T>>>
      cache;
  std::lock_guard<std::recursive_mutex> lock(mutex);
  const auto key = std::make_pair(static_cast<int>(kind), n);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  auto plan = detail::build_plan<T>(kind, n);
  cache.emplace(key, plan);
  return plan;
}

// A transform as: load (flip or alternate signs, weight the end points),
// kernel, store (weight the end points, scale, flip or alternate signs).
struct Recipe {
  Kernel kernel = Kernel::Dct2;
  bool load_flip = false, load_alt = false;
  double load_first = 1, load_last = 1;
  double store_first = 1, store_last = 1;
  bool store_flip = false, store_alt = false;
  double scale = 1;
};

inline Kernel kernel_of(const std::string& op) {
  if (op == "dct1" || op == "idct1") return Kernel::Dct1;
  if (op == "dst1" || op == "idst1") return Kernel::Dst1;
  if (op == "dct2" || op == "idct3" || op == "dst2" || op == "idst3")
    return Kernel::Dct2;
  if (op == "idct2" || op == "dct3" || op == "idst2" || op == "dst3")
    return Kernel::Idct2;
  if (op == "dct4" || op == "idct4" || op == "dst4" || op == "idst4")
    return Kernel::Dct4;
  throw std::invalid_argument("unknown transform '" + op + "'");
}

// The recipe for `op` on n points, or for its transpose (used for autograd).
inline Recipe recipe(const std::string& op, std::size_t n, bool transpose) {
  enum Edge { kNone, kFlip, kAlt };
  Edge pre = kNone, post = kNone;
  const double dn = static_cast<double>(n);
  double scale = 1;
  if (op == "idct1") scale = 2 / (dn - 1);
  if (op == "idst1") scale = 2 / (dn + 1);
  if (op == "dst2") pre = kAlt, post = kFlip;
  if (op == "idst2") pre = kFlip, post = kAlt;
  if (op == "dct3") scale = dn / 2;
  if (op == "idct3") scale = 2 / dn;
  if (op == "dst3") pre = kFlip, post = kAlt, scale = dn / 2;
  if (op == "idst3") pre = kAlt, post = kFlip, scale = 2 / dn;
  if (op == "idct4") scale = 2 / dn;
  if (op == "dst4" || op == "idst4") pre = kFlip, post = kAlt;
  if (op == "idst4") scale = 2 / dn;

  Recipe r;
  r.kernel = kernel_of(op);
  r.scale = scale;
  Edge load = pre, store = post;
  if (transpose) {
    // (post K pre)^T = pre^T K^T post^T. Flips and signs are symmetric;
    // dct1, dct2 and idct2 transpose into kernels with end-point weights.
    std::swap(load, store);
    switch (r.kernel) {
      case Kernel::Dct1:
        // dct1^T = W dct1 W^-1 with W = diag(1/2, 1, ..., 1, 1/2).
        r.load_first = r.load_last = 2;
        r.store_first = r.store_last = 0.5;
        break;
      case Kernel::Dct2:  // dct2^T = (n/2) idct2 diag(2, 1, ..., 1)
        r.kernel = Kernel::Idct2;
        r.load_first = 2;
        r.scale *= dn / 2;
        break;
      case Kernel::Idct2:  // idct2^T = (2/n) diag(1/2, 1, ..., 1) dct2
        r.kernel = Kernel::Dct2;
        r.store_first = 0.5;
        r.scale *= 2 / dn;
        break;
      case Kernel::Dst1:
      case Kernel::Dct4:
        break;
    }
  }
  r.load_flip = load == kFlip;
  r.load_alt = load == kAlt;
  r.store_flip = store == kFlip;
  r.store_alt = store == kAlt;
  return r;
}

// Apply `r` along the middle axis of a C-contiguous (outer, n, inner) array.
template <class T>
void transform(const T* in, T* out, std::size_t outer, std::size_t n,
               std::size_t inner, const Recipe& r, std::size_t nthreads) {
  using V = detail::Vec<T>;
  const std::size_t lines = outer * inner;
  if (n == 0 || lines == 0) return;
  if (r.kernel == Kernel::Dct1 && n < 2)
    throw std::invalid_argument("dct1 needs at least 2 points");
  const auto plan = get_plan<T>(r.kernel, n);
  const std::size_t blocks = (lines + kLanes - 1) / kLanes;

  // Per-point load and store factors: sign, end-point weight and scale.
  std::vector<T> load(n), store(n);
  for (std::size_t j = 0; j < n; ++j) {
    double f = (r.load_alt && j % 2) ? -1.0 : 1.0;
    double g = (r.store_alt && j % 2) ? -r.scale : r.scale;
    if (j == 0) f *= r.load_first, g *= r.store_first;
    if (j == n - 1) f *= r.load_last, g *= r.store_last;
    load[j] = static_cast<T>(f);
    store[j] = static_cast<T>(g);
  }

  auto work = [&](std::size_t first, std::size_t last) {
    std::vector<V> buffer(n + plan->work);
    V* x = buffer.data();
    V* w = x + n;
    std::size_t base[kLanes];
    for (std::size_t block = first; block < last; ++block) {
      const std::size_t count = std::min(kLanes, lines - block * kLanes);
      for (std::size_t l = 0; l < count; ++l) {
        const std::size_t t = block * kLanes + l;
        base[l] = (t / inner) * n * inner + t % inner;
      }
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t src = (r.load_flip ? n - 1 - j : j) * inner;
        for (std::size_t l = 0; l < count; ++l)
          x[j][l] = load[j] * in[base[l] + src];
        for (std::size_t l = count; l < kLanes; ++l) x[j][l] = T(0);
      }
      detail::run(*plan, x, w);
      for (std::size_t j = 0; j < n; ++j) {
        const std::size_t dst = (r.store_flip ? n - 1 - j : j) * inner;
        for (std::size_t l = 0; l < count; ++l)
          out[base[l] + dst] = store[j] * x[j][l];
      }
    }
  };

  // Threads only pay off once each has a few hundred thousand points.
  const std::size_t by_size = std::max<std::size_t>(1, lines * n / (1 << 17));
  const std::size_t threads =
      std::max<std::size_t>(1, std::min({nthreads, blocks, by_size}));
  if (threads == 1) return work(0, blocks);
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  const std::size_t per = (blocks + threads - 1) / threads;
  for (std::size_t i = 1; i < threads; ++i) {
    const std::size_t first = std::min(blocks, i * per);
    const std::size_t last = std::min(blocks, first + per);
    pool.emplace_back(work, first, last);
  }
  work(0, std::min(blocks, per));
  for (auto& thread : pool) thread.join();
}

}  // namespace orthel
