#include "etf_ap/solver.hpp"

#include <iomanip>
#include <iostream>

int main() {
    etf_ap::Params p;
    p.n_x = 41;
    p.q_e_max = 5;
    p.q_i_max = 5;
    p.n_t = 80;
    p.K = 3;
    const auto sol = etf_ap::solve(p);
    std::cout << std::setprecision(8) << "V(0,0,0,0) = " << sol.V0 << "\n";
    int n_create = 0, n_redeem = 0, n_cont = 0;
    for (int r : sol.region) {
        if (r == 1) ++n_create;
        else if (r == -1) ++n_redeem;
        else ++n_cont;
    }
    std::cout << "QVI cells  continuation=" << n_cont << "  create=" << n_create
              << "  redeem=" << n_redeem << "\n";
    return 0;
}
