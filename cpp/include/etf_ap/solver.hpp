#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace etf_ap {

struct Params {
    double kappa = 2.0;
    double sigma_x = 0.08;
    double A = 140.0;
    double k_lambda = 12.0;
    double eta = 0.02;
    double phi = 0.015;
    double fee = 0.08;
    int K = 4;
    double T = 1.0;
    double x_max = 0.35;
    int n_x = 41;
    int q_e_max = 5;
    int q_i_max = 5;
    int n_t = 80;
    double delta_min = 1e-4;
    double delta_max = 0.50;
    double nu_max = 25.0;

    double dx() const { return 2.0 * x_max / (n_x - 1); }
    double dt() const { return T / n_t; }
    int n_qe() const { return 2 * q_e_max + 1; }
    int n_qi() const { return 2 * q_i_max + 1; }
    int n_time() const { return n_t + 1; }

    int idx(int ix, int jq, int kq) const { return (ix * n_qe() + jq) * n_qi() + kq; }
};

inline void thomas(const std::vector<double>& lo, const std::vector<double>& di,
                   const std::vector<double>& up, std::vector<double>& d) {
    const int n = static_cast<int>(di.size());
    std::vector<double> b = di;
    std::vector<double> c = up;
    for (int i = 1; i < n; ++i) {
        const double w = lo[i] / b[i - 1];
        b[i] -= w * c[i - 1];
        d[i] -= w * d[i - 1];
    }
    d[n - 1] /= b[n - 1];
    for (int i = n - 2; i >= 0; --i) d[i] = (d[i] - c[i] * d[i + 1]) / b[i];
}

struct Solution {
    Params p;
    std::vector<double> V;  // (n_x, n_qe, n_qi) at t=0 after solve
    std::vector<double> delta_b, delta_a, nu;
    std::vector<int> region;
    double V0 = 0.0;
};

inline Solution solve(const Params& p) {
    if (p.n_x % 2 == 0) throw std::runtime_error("n_x must be odd");
    const int nx = p.n_x, nqe = p.n_qe(), nqi = p.n_qi(), nt = p.n_t;
    const double dx = p.dx(), dt = p.dt();
    const int N = nx * nqe * nqi;

    std::vector<double> x(nx);
    for (int i = 0; i < nx; ++i) x[i] = -p.x_max + i * dx;

    auto at = [&](std::vector<double>& a, int i, int j, int k) -> double& {
        return a[p.idx(i, j, k)];
    };
    auto cat = [&](const std::vector<double>& a, int i, int j, int k) -> double {
        return a[p.idx(i, j, k)];
    };

    std::vector<double> V(N, 0.0), Vn(N, 0.0);
    std::vector<double> db(N, 0.0), da(N, 0.0), nu(N, 0.0);
    std::vector<int> region(N, 0);

    std::vector<double> lo(nx, 0.0), di(nx, 0.0), up(nx, 0.0);
    const double alpha = (p.sigma_x * p.sigma_x) / (2.0 * dx * dx);
    for (int i = 0; i < nx; ++i) {
        const double mu = -p.kappa * x[i];
        const double am = std::abs(mu) / dx;
        if (i == 0) {
            di[0] = 1.0 / dt + 2.0 * alpha + am;
            up[0] = -2.0 * alpha - am;
            continue;
        }
        if (i == nx - 1) {
            di[nx - 1] = 1.0 / dt + 2.0 * alpha + am;
            lo[nx - 1] = -2.0 * alpha - am;
            continue;
        }
        double adv_im1 = 0, adv_i = 0, adv_ip1 = 0;
        if (mu >= 0) {
            adv_i = mu / dx;
            adv_im1 = -mu / dx;
        } else {
            adv_i = -mu / dx;
            adv_ip1 = mu / dx;
        }
        lo[i] = -adv_im1 - alpha;
        di[i] = -adv_i + 2.0 * alpha + 1.0 / dt;
        up[i] = -adv_ip1 - alpha;
    }

    auto clip = [](double z, double a, double b) { return std::max(a, std::min(b, z)); };

    for (int n = nt - 1; n >= 0; --n) {
        std::vector<double> H(N, 0.0);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < nqe; ++j) {
                for (int k = 0; k < nqi; ++k) {
                    const bool can_buy = (j + 1 < nqe);
                    const bool can_sell = (j - 1 >= 0);
                    const double dplus = can_buy ? cat(Vn, i, j + 1, k) - cat(Vn, i, j, k) : 0.0;
                    const double dminus = can_sell ? cat(Vn, i, j - 1, k) - cat(Vn, i, j, k) : 0.0;
                    double dqi = 0.0;
                    if (k == 0) dqi = cat(Vn, i, j, 1) - cat(Vn, i, j, 0);
                    else if (k == nqi - 1) dqi = cat(Vn, i, j, nqi - 1) - cat(Vn, i, j, nqi - 2);
                    else dqi = 0.5 * (cat(Vn, i, j, k + 1) - cat(Vn, i, j, k - 1));

                    const double delb = clip(1.0 / p.k_lambda + x[i] - dplus, p.delta_min, p.delta_max);
                    const double dela = clip(1.0 / p.k_lambda - x[i] - dminus, p.delta_min, p.delta_max);
                    const double lb = p.A * std::exp(-p.k_lambda * delb);
                    const double la = p.A * std::exp(-p.k_lambda * dela);
                    const double hb = can_buy ? lb * (dplus + delb - x[i]) : 0.0;
                    const double ha = can_sell ? la * (dminus + dela + x[i]) : 0.0;
                    double nv = clip(dqi / (2.0 * p.eta), -p.nu_max, p.nu_max);
                    const double hh = nv * dqi - p.eta * nv * nv;
                    const int qe = j - p.q_e_max;
                    const int qi = k - p.q_i_max;
                    const double pen = p.phi * (qe * qe + qi * qi);
                    at(H, i, j, k) = hb + ha + hh - pen;
                    at(db, i, j, k) = delb;
                    at(da, i, j, k) = dela;
                    at(nu, i, j, k) = nv;
                }
            }
        }

        std::vector<double> Vcont(N, 0.0);
        for (int j = 0; j < nqe; ++j) {
            for (int k = 0; k < nqi; ++k) {
                std::vector<double> rhs(nx);
                for (int i = 0; i < nx; ++i) rhs[i] = cat(Vn, i, j, k) / dt + cat(H, i, j, k);
                thomas(lo, di, up, rhs);
                for (int i = 0; i < nx; ++i) at(Vcont, i, j, k) = rhs[i];
            }
        }

        V = Vcont;
        std::fill(region.begin(), region.end(), 0);
        const int K = p.K;
        for (int sweep = 0; sweep < 4; ++sweep) {
            std::vector<double> nxt = V;
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < nqe; ++j) {
                    for (int k = 0; k < nqi; ++k) {
                        double best = cat(V, i, j, k);
                        int choice = 0;
                        if (j + K < nqe && k - K >= 0) {
                            const double c = cat(V, i, j + K, k - K) - p.fee;
                            if (c > best) {
                                best = c;
                                choice = 1;
                            }
                        }
                        if (j - K >= 0 && k + K < nqi) {
                            const double r = cat(V, i, j - K, k + K) - p.fee;
                            if (r > best) {
                                best = r;
                                choice = -1;
                            }
                        }
                        at(nxt, i, j, k) = best;
                        region[p.idx(i, j, k)] = choice;
                    }
                }
            }
            V.swap(nxt);
        }
        Vn.swap(V);
        V = Vn;
    }

    Solution s;
    s.p = p;
    s.V = V;
    s.delta_b = db;
    s.delta_a = da;
    s.nu = nu;
    s.region = region;
    const int i0 = (nx - 1) / 2;
    s.V0 = cat(V, i0, p.q_e_max, p.q_i_max);
    return s;
}

}  // namespace etf_ap
