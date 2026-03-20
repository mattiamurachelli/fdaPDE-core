// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_MORE_THUENTE_LINE_SEARCH_H__
#define __FDAPDE_MORE_THUENTE_LINE_SEARCH_H__

#include "header_check.h"
#include <limits>
#include <cmath>
#include <algorithm>

namespace fdapde {

class MoreThuenteLineSearch {
private:
    static constexpr int max_iter_ = 20;    // max number of iterations
    double alpha_ = 1.0;                    // initial step guess
    double c1_ = 1e-4;                      // sufficient decrease constant (Armijo condition)
    double c2_ = 0.9;                       // curvature constant (strong Wolfe condition)
    double min_step_ = 1e-20;               // minimum step value
    double max_step_ = 1e+20;               // maximum step value

    // Helper functions for polynomial minimization
    // Quadratic minimizer 1st variant
    static double quadratic_minimizer(double a, double b, double fa, double ga, double fb) {
        const double ba = b - a;
        const double w = 0.5 * ba * ga / (fa - fb + ba * ga);
        return a + w * ba;
    }

    // Quadratic minimizer 2nd variant (using gradients only)
    static double quadratic_minimizer(double a, double b, double ga, double gb) {
        const double w = ga / (ga - gb);
        return a + w * (b - a);
    }

    // Cubic minimizer
    static double cubic_minimizer(double a, double b, double fa, double fb,
                                  double ga, double gb, bool& exists) {
        using std::abs;
        using std::sqrt;

        const double apb = a + b;
        const double ba = b - a;
        const double ba2 = ba * ba;
        const double fba = fb - fa;
        const double gba = gb - ga;
        
        const double z3 = (ga + gb) * ba - 2.0 * fba;
        const double z2 = 0.5 * (gba * ba2 - 3.0 * apb * z3);
        const double z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;

        const double eps = std::numeric_limits<double>::epsilon();
        if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1)) {
            exists = (z2 * ba > 0.0);
            return exists ? (-0.5 * z1 / z2) : b;
        }

        const double u = z2 / (3.0 * z3), v = z1 / z2;
        const double vu = v / u;
        exists = (vu <= 1.0);
        if (!exists) return b;

        double r1 = 0.0, r2 = 0.0;
        if (abs(u) >= abs(v)) {
            const double w = 1.0 + sqrt(1.0 - vu);
            r1 = -u * w;
            r2 = -v / w;
        } else {
            const double sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1.0 - u / v);
            r1 = -u - sqrtd;
            r2 = -u + sqrtd;
        }
        return (z3 * ba > 0.0) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
    }

    // Select the next step size according to the current step sizes,
    // function values, and derivatives
    static double step_selection(double al, double au, double at,
                                double fl, double fu, double ft,
                                double gl, double gu, double gt) {
        using std::abs;
        using std::isfinite;

        if (al == au) return al;

        if (!isfinite(ft) || !isfinite(gt))
            return (al + at) / 2.0;

        bool ac_exists;
        const double ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
        const double aq = quadratic_minimizer(al, at, fl, gl, ft);

        if (ft > fl) {
            if (!ac_exists) return aq;
            return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / 2.0);
        }

        const double as = quadratic_minimizer(al, at, gl, gt);
        
        if (gt * gl < 0.0)
            return (abs(ac - at) >= abs(as - at)) ? ac : as;

        const double deltal = 1.1, deltau = 0.66;
        if (abs(gt) < abs(gl)) {
            const double res = (ac_exists &&
                              (ac - at) * (at - al) > 0.0 &&
                              abs(ac - at) < abs(as - at)) ?
                ac : as;
            return (at > al) ?
                std::min(at + deltau * (au - at), res) :
                std::max(at + deltau * (au - at), res);
        }

        if ((!isfinite(au)) || (!isfinite(fu)) || (!isfinite(gu)))
            return at + deltal * (at - al);

        bool ae_exists;
        const double ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
        return (at > al) ?
            std::min(at + deltau * (au - at), ae) :
            std::max(at + deltau * (au - at), ae);
    }

public:
    MoreThuenteLineSearch() = default;
    MoreThuenteLineSearch(double alpha, double c1, double c2) : alpha_(alpha), c1_(c1), c2_(c2) { }

    void set_max_step(double s) { if (s > 0.0) max_step_ = s; }
    void set_min_step(double s) { if (s > 0.0) min_step_ = s; }
    void set_wolfe_constants(double c1, double c2) { c1_ = c1; c2_ = c2; }

    template <typename Opt, typename Obj> 
    bool adapt_hook(Opt& opt, Obj& obj) {
        fdapde_static_assert(is_gradient_based_opt_v<Opt>, THIS_METHOD_IS_FOR_GRADIENT_BASED_OPTIMIZATION_ONLY);
        
        // Get current function value and gradient projection
        const double fx_init = obj(opt.x_old);
        const double dg_init = opt.grad_old.dot(opt.update);
        
        // In bounded L-BFGS, the search direction might not be a perfect descent direction
        // due to active constraints, so we use a fallback instead of throwing
        if (dg_init >= 0.0) {
            // Use gradient descent direction as fallback
            opt.update = -opt.grad_old;
            const double new_dg_init = opt.grad_old.dot(opt.update);
            
            // If even gradient descent is not a descent direction, use a very small step
            if (new_dg_init >= 0.0) {
                opt.h = min_step_;
                return false;
            }
            
            // Recompute with gradient descent direction
            return adapt_hook(opt, obj); // Recursive call with new direction
        }

        // Tolerance for convergence test (wolfe constants)
        const double test_decr = c1_ * dg_init;
        const double test_curv = -c2_ * dg_init;

        // Bracketing interval initialization
        double I_lo = 0.0, I_hi = std::numeric_limits<double>::infinity();
        double fI_lo = 0.0, fI_hi = std::numeric_limits<double>::infinity();
        double gI_lo = (1.0 - c1_) * dg_init, gI_hi = std::numeric_limits<double>::infinity();
        
        // Save best point so far
        typename std::decay_t<decltype(opt.x_old)> x_best = opt.x_old;
        typename std::decay_t<decltype(opt.grad_old)> grad_best = opt.grad_old;
        double fx_best = fx_init;
        double dg_best = dg_init;
        double step_best = 0.0;

        // Current trial step
        double step = alpha_;
        auto grad = obj.gradient();
        
        // Use max_step_ directly
        double alpha_max = max_step_;
        
        // Evaluate function and gradient at initial step
        typename std::decay_t<decltype(opt.x_old)> x_trial = (opt.x_old + step * opt.update).eval();
        double fx = obj(x_trial);
        typename std::decay_t<decltype(opt.grad_old)> grad_trial = grad(x_trial);
        double dg = grad_trial.dot(opt.update);

        // Check strong Wolfe conditions
        if (fx <= fx_init + step * test_decr && std::abs(dg) <= test_curv) {
            opt.x_new = x_trial;
            opt.grad_new = grad_trial;
            opt.h = step;
            return false;
        }

        // Extrapolation factor
        const double delta = 1.1;
        
        // Main line search loop
        int iter;
        for (iter = 0; iter < max_iter_; iter++) {
            const double ft = fx - fx_init - step * test_decr;
            const double gt = dg - c1_ * dg_init;

            double new_step;
            if (ft > fI_lo) {
                // Case 1: ft > fl
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);
                if (new_step <= min_step_)
                    new_step = (I_lo + step) / 2.0;

                I_hi = step;
                fI_hi = ft;
                gI_hi = gt;
            } else if (gt * (I_lo - step) > 0.0) {
                // Case 2: ft <= fl, gt * (al - at) > 0
                new_step = std::min(alpha_max, step + delta * (step - I_lo));

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                // Update best point
                x_best = x_trial;
                grad_best = grad_trial;
                fx_best = fx;
                dg_best = dg;
                step_best = step;
            } else {
                // Case 3: ft <= fl, gt * (al - at) <= 0
                new_step = step_selection(I_lo, I_hi, step, fI_lo, fI_hi, ft, gI_lo, gI_hi, gt);

                I_hi = I_lo;
                fI_hi = fI_lo;
                gI_hi = gI_lo;

                I_lo = step;
                fI_lo = ft;
                gI_lo = gt;
                // Update best point
                x_best = x_trial;
                grad_best = grad_trial;
                fx_best = fx;
                dg_best = dg;
                step_best = step;
            }

            // Check if we hit maximum step
            if (step == alpha_max && new_step >= alpha_max) {
                opt.x_new = x_trial;
                opt.grad_new = grad_trial;
                opt.h = step;
                return false;
            }

            step = new_step;

            if (step < min_step_) {
                opt.h = min_step_;
                return false;
            }

            if (step > max_step_) {
                opt.h = max_step_;
                return false;
            }

            // Evaluate at new step
            x_trial = (opt.x_old + step * opt.update).eval();
            fx = obj(x_trial);
            grad_trial = grad(x_trial);
            dg = grad_trial.dot(opt.update);

            // Check strong Wolfe conditions
            if (fx <= fx_init + step * test_decr && std::abs(dg) <= test_curv) {
                opt.x_new = x_trial;
                opt.grad_new = grad_trial;
                opt.h = step;
                return false;
            }

            // Check if we hit bound and this is the best we can do
            if (step >= alpha_max) {
                const double ft_bound = fx - fx_init - step * test_decr;
                if (ft_bound <= fI_lo) {
                    opt.x_new = x_trial;
                    opt.grad_new = grad_trial;
                    opt.h = step;
                    return false;
                }
            }
        }

        // If we exceeded iterations
        if (iter >= max_iter_) {
            const double ft = fx - fx_init - step * test_decr;
            if (ft <= fI_lo) {
                opt.x_new = x_trial;
                opt.grad_new = grad_trial;
                opt.h = step;
                return false;
            }

            if (step_best > 0.0) {
                opt.x_new = x_best;
                opt.grad_new = grad_best;
                opt.h = step_best;
            } else {
                // Use a conservative fallback step
                opt.h = std::min(alpha_, 0.1);
            }
        } else {
            opt.x_new = x_trial;
            opt.grad_new = grad_trial;
            opt.h = step;
        }

        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_MORE_THUENTE_LINE_SEARCH_H__