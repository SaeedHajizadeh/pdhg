//! PDHG (Chambolle–Pock) for convex–concave saddle problems
//!
//!     min_x max_y  f(x) + <K x, y> - g*(y)
//!
//! Iteration (theta = 1):
//!     x_{k+1} = prox_{tau f}   (x_k - tau K^T y_k)
//!     xbar    = 2 x_{k+1} - x_k
//!     y_{k+1} = prox_{sigma g*}(y_k + sigma K xbar)
//! Converges for tau * sigma * ||K||^2 < 1. Ergodic averages get O(1/k) gap.
//!
//! Std-only, no crates. Build: `rustc -O pdhg.rs && ./pdhg`
 
// ---------------------------------------------------------------- linear algebra
 
/// Dense row-major matrix, rows x cols (K: R^cols -> R^rows).
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}
 
impl Mat {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Mat { rows, cols, data }
    }
 
    /// out = K x
    pub fn matvec(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.rows {
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            out[i] = row.iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
 
    /// out = K^T y
    pub fn matvec_t(&self, y: &[f64], out: &mut [f64]) {
        out.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..self.rows {
            let yi = y[i];
            let row = &self.data[i * self.cols..(i + 1) * self.cols];
            for (o, a) in out.iter_mut().zip(row) {
                *o += a * yi;
            }
        }
    }
}
 
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|a| a * a).sum::<f64>().sqrt()
}
 
/// ||K||_2 via power iteration on K^T K.
pub fn op_norm(k: &Mat, iters: usize) -> f64 {
    let mut v = vec![1.0 / (k.cols as f64).sqrt(); k.cols];
    let mut kv = vec![0.0; k.rows];
    let mut ktkv = vec![0.0; k.cols];
    let mut s = 0.0;
    for _ in 0..iters {
        k.matvec(&v, &mut kv);
        k.matvec_t(&kv, &mut ktkv);
        let n = norm(&ktkv);
        if n == 0.0 {
            return 0.0;
        }
        s = n.sqrt();
        v.iter_mut().zip(&ktkv).for_each(|(a, b)| *a = b / n);
    }
    s
}
 
// ---------------------------------------------------------------- proximal operators
 
/// out = prox_{step * h}(v) = argmin_u h(u) + ||u - v||^2 / (2 step)
pub trait Prox {
    fn prox(&self, v: &[f64], step: f64, out: &mut [f64]);
}
 
/// h = 0
pub struct Zero;
impl Prox for Zero {
    fn prox(&self, v: &[f64], _: f64, out: &mut [f64]) {
        out.copy_from_slice(v);
    }
}
 
/// h = indicator of the probability simplex
pub struct Simplex;
impl Prox for Simplex {
    fn prox(&self, v: &[f64], _: f64, out: &mut [f64]) {
        // Sort-based Euclidean projection (Held et al. / Duchi et al.), O(n log n).
        let mut u = v.to_vec();
        u.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let (mut cum, mut theta) = (0.0, 0.0);
        for (j, &uj) in u.iter().enumerate() {
            cum += uj;
            let t = (cum - 1.0) / (j as f64 + 1.0);
            if uj - t > 0.0 {
                theta = t;
            }
        }
        for (o, &vi) in out.iter_mut().zip(v) {
            *o = (vi - theta).max(0.0);
        }
    }
}
 
/// h = indicator of [lo, hi]^n
pub struct Boxed {
    pub lo: f64,
    pub hi: f64,
}
impl Prox for Boxed {
    fn prox(&self, v: &[f64], _: f64, out: &mut [f64]) {
        for (o, &vi) in out.iter_mut().zip(v) {
            *o = vi.clamp(self.lo, self.hi);
        }
    }
}
 
/// h = lambda * ||.||_1
pub struct L1 {
    pub lambda: f64,
}
impl Prox for L1 {
    fn prox(&self, v: &[f64], step: f64, out: &mut [f64]) {
        let t = self.lambda * step;
        for (o, &vi) in out.iter_mut().zip(v) {
            *o = vi.signum() * (vi.abs() - t).max(0.0);
        }
    }
}
 
/// Wraps prox_h into prox_{h*} via Moreau:
///     prox_{s h*}(v) = v - s * prox_{h/s}(v / s)
/// Lets you specify g instead of g*.
pub struct Conjugate<P: Prox>(pub P);
impl<P: Prox> Prox for Conjugate<P> {
    fn prox(&self, v: &[f64], step: f64, out: &mut [f64]) {
        let scaled: Vec<f64> = v.iter().map(|a| a / step).collect();
        self.0.prox(&scaled, 1.0 / step, out);
        for (o, &vi) in out.iter_mut().zip(v) {
            *o = vi - step * *o;
        }
    }
}
 
// ---------------------------------------------------------------- solver
 
pub struct PdhgConfig {
    pub max_iter: usize,
    pub tol: f64,
    /// None => tau = sigma = 0.95 / ||K||
    pub tau: Option<f64>,
    pub sigma: Option<f64>,
    pub check_every: usize,
    pub verbose: bool,
}
 
impl Default for PdhgConfig {
    fn default() -> Self {
        PdhgConfig {
            max_iter: 100_000,
            tol: 1e-8,
            tau: None,
            sigma: None,
            check_every: 50,
            verbose: false,
        }
    }
}
 
pub struct PdhgResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub x_avg: Vec<f64>,
    pub y_avg: Vec<f64>,
    pub iters: usize,
    pub residual: f64,
    pub converged: bool,
}
 
/// Solves min_x max_y f(x) + <Kx, y> - g*(y).
///
/// Stopping rule uses the PDHG optimality residual:
///     p = (x_k - x_{k+1})/tau - K^T (y_k - y_{k+1})  in  df(x_{k+1}) + K^T y_{k+1}
///     d = (y_k - y_{k+1})/sigma + K (x_{k+1} - x_k)  in  dg*(y_{k+1}) - K x_{k+1}
/// so (p, d) -> 0 certifies a saddle point.
pub fn pdhg<F: Prox, G: Prox>(
    k: &Mat,
    f: &F,
    g_star: &G,
    x0: &[f64],
    y0: &[f64],
    cfg: &PdhgConfig,
) -> PdhgResult {
    let (n, m) = (k.cols, k.rows);
    assert_eq!(x0.len(), n);
    assert_eq!(y0.len(), m);
 
    let l = op_norm(k, 200).max(1e-12);
    let tau = cfg.tau.unwrap_or(0.95 / l);
    let sigma = cfg.sigma.unwrap_or(0.95 / l);
    assert!(
        tau * sigma * l * l < 1.0,
        "step sizes violate tau*sigma*||K||^2 < 1"
    );
 
    let mut x = x0.to_vec();
    let mut y = y0.to_vec();
    let mut x_new = vec![0.0; n];
    let mut y_new = vec![0.0; m];
    let mut xbar = vec![0.0; n];
    let mut kty = vec![0.0; n];
    let mut kxbar = vec![0.0; m];
    let mut tmp_n = vec![0.0; n];
    let mut tmp_m = vec![0.0; m];
    let mut dx = vec![0.0; n];
    let mut dy = vec![0.0; m];
    let mut x_avg = vec![0.0; n];
    let mut y_avg = vec![0.0; m];
 
    let mut residual = f64::INFINITY;
 
    k.matvec_t(&y, &mut kty);
 
    for it in 1..=cfg.max_iter {
        // primal step
        for i in 0..n {
            tmp_n[i] = x[i] - tau * kty[i];
        }
        f.prox(&tmp_n, tau, &mut x_new);
 
        // extrapolation
        for i in 0..n {
            xbar[i] = 2.0 * x_new[i] - x[i];
        }
 
        // dual step
        k.matvec(&xbar, &mut kxbar);
        for j in 0..m {
            tmp_m[j] = y[j] + sigma * kxbar[j];
        }
        g_star.prox(&tmp_m, sigma, &mut y_new);
 
        // ergodic averages (running mean)
        let w = 1.0 / it as f64;
        for i in 0..n {
            x_avg[i] += w * (x_new[i] - x_avg[i]);
        }
        for j in 0..m {
            y_avg[j] += w * (y_new[j] - y_avg[j]);
        }
 
        let check = it % cfg.check_every == 0 || it == cfg.max_iter;
        if check {
            for i in 0..n {
                dx[i] = x_new[i] - x[i];
            }
            for j in 0..m {
                dy[j] = y_new[j] - y[j];
            }
            // p = -dx/tau + K^T dy ; d = -dy/sigma + K dx
            k.matvec_t(&dy, &mut tmp_n);
            k.matvec(&dx, &mut tmp_m);
            let p: f64 = (0..n)
                .map(|i| (-dx[i] / tau + tmp_n[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let d: f64 = (0..m)
                .map(|j| (-dy[j] / sigma + tmp_m[j]).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale = 1.0 + norm(&x_new).max(norm(&y_new));
            residual = (p + d) / scale;
            if cfg.verbose {
                println!("iter {:>7}  rel. residual {:.3e}", it, residual);
            }
        }
 
        std::mem::swap(&mut x, &mut x_new);
        std::mem::swap(&mut y, &mut y_new);
        k.matvec_t(&y, &mut kty);
 
        if check && residual < cfg.tol {
            return PdhgResult { x, y, x_avg, y_avg, iters: it, residual, converged: true };
        }
    }
 
    PdhgResult { x, y, x_avg, y_avg, iters: cfg.max_iter, residual, converged: false }
}
 
// ---------------------------------------------------------------- examples
 
/// Tiny LCG so the demo stays dependency-free.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
}
 
/// Exploitability of the matrix game min_{x in Δn} max_{y in Δm} y^T A x:
///     max_j (A x)_j - min_i (A^T y)_i  >= 0, zero iff Nash.
fn game_gap(a: &Mat, x: &[f64], y: &[f64]) -> f64 {
    let mut ax = vec![0.0; a.rows];
    let mut aty = vec![0.0; a.cols];
    a.matvec(x, &mut ax);
    a.matvec_t(y, &mut aty);
    ax.iter().cloned().fold(f64::MIN, f64::max) - aty.iter().cloned().fold(f64::MAX, f64::min)
}
 
/// g*(y) = 0.5||y||^2 + <b, y>  (conjugate of 0.5||z - b||^2)
/// prox_{s g*}(v) = (v - s b) / (1 + s)
struct LsqConj {
    b: Vec<f64>,
}
impl Prox for LsqConj {
    fn prox(&self, v: &[f64], s: f64, out: &mut [f64]) {
        for ((o, &vi), &bi) in out.iter_mut().zip(v).zip(&self.b) {
            *o = (vi - s * bi) / (1.0 + s);
        }
    }
}
 
fn main() {
    // --- 1. Rock-paper-scissors: unique Nash is uniform, value 0.
    let rps = Mat::new(3, 3, vec![0., 1., -1., -1., 0., 1., 1., -1., 0.]);
    let x0 = vec![1.0, 0.0, 0.0];
    let y0 = vec![0.0, 1.0, 0.0];
    let res = pdhg(&rps, &Simplex, &Simplex, &x0, &y0, &PdhgConfig::default());
    println!("RPS: iters={} converged={}", res.iters, res.converged);
    println!("  x = {:.4?}", res.x);
    println!("  y = {:.4?}", res.y);
    println!("  gap(last) = {:.2e}", game_gap(&rps, &res.x, &res.y));
 
    // --- 2. Random 60x40 zero-sum game.
    let (m, n) = (60, 40);
    let mut rng = Lcg(42);
    let a = Mat::new(m, n, (0..m * n).map(|_| rng.next()).collect());
    let res = pdhg(
        &a,
        &Simplex,
        &Simplex,
        &vec![1.0 / n as f64; n],
        &vec![1.0 / m as f64; m],
        &PdhgConfig { tol: 1e-9, ..Default::default() },
    );
    println!("\nRandom game {}x{}: iters={} converged={}", m, n, res.iters, res.converged);
    println!("  gap(last) = {:.2e}", game_gap(&a, &res.x, &res.y));
    println!("  gap(avg)  = {:.2e}", game_gap(&a, &res.x_avg, &res.y_avg));
 
    // --- 3. LASSO as a saddle: min_x lam||x||_1 + max_y <Ax, y> - (0.5||y||^2 + <b,y>)
    let (m, n) = (50, 100);
    let a = Mat::new(m, n, (0..m * n).map(|_| rng.next() / (m as f64).sqrt()).collect());
    let mut x_true = vec![0.0; n];
    for i in [3, 17, 42, 77] {
        x_true[i] = if i % 2 == 0 { 2.0 } else { -1.5 };
    }
    let mut b = vec![0.0; m];
    a.matvec(&x_true, &mut b);
    let res = pdhg(
        &a,
        &L1 { lambda: 0.01 },
        &LsqConj { b },
        &vec![0.0; n],
        &vec![0.0; m],
        &PdhgConfig { tol: 1e-10, max_iter: 200_000, ..Default::default() },
    );
    let support: Vec<usize> = (0..n).filter(|&i| res.x[i].abs() > 1e-3).collect();
    println!("\nLASSO: iters={} converged={}", res.iters, res.converged);
    println!("  recovered support = {:?}", support);
    println!("  x[support] = {:.3?}", support.iter().map(|&i| res.x[i]).collect::<Vec<_>>());
}