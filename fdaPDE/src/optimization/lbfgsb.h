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

#ifndef __FDAPDE_LBFGSB_H__
#define __FDAPDE_LBFGSB_H__

#include "header_check.h"

namespace fdapde {

// Helper traits to detect Eigen column vector (used to solve template instantiation issues
// associated to the third argument)
template <typename T>
struct is_eigen_col_vector : std::false_type {};
template <typename Scalar, int Rows, int Options, int MaxRows>
struct is_eigen_col_vector<Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, 1>> : std::true_type {};

template <int N> class LBFGSB {
   private:
    using vector_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_;                 // objective value at optimum
    int n_iter_ = 0;               // current iteration number
    std::vector<double> values_;   // explored objective values during optimization

    int max_iter_;        // maximum number of iterations before forced stop
    double tol_;          // tolerance on error before forced stop
    double step_;         // initial update step (initial alpha guess)
    int mem_size_ = 6;    // number of vector pairs used for approximating the inverse Hessian
    Eigen::Matrix<double, Dynamic, Dynamic> grad_mem_, x_mem_; // circular buffers: columns store y_k and s_k

   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;

    // constructors
    LBFGSB() : max_iter_(10000), tol_(1e-6), step_(1.0) {}
    LBFGSB(int max_iter, double tol, double step, int mem_size) :
        max_iter_(max_iter), tol_(tol), step_(step), mem_size_(mem_size) {
        fdapde_assert(mem_size_ >= 0);
    }
    LBFGSB(const LBFGSB& other) :
        max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_), mem_size_(other.mem_size_) { }
    LBFGSB& operator=(const LBFGSB& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        mem_size_ = other.mem_size_;
        return *this;
    }

    // Unbounded version (bounds automatically set to +/- infty)
    template <typename ObjectiveT, typename... Callbacks,
            typename = std::enable_if_t<
                sizeof...(Callbacks) == 0 ||
                !is_eigen_col_vector<std::decay_t<std::tuple_element_t<0, std::tuple<Callbacks...>>>>::value
            >>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        constexpr int rows = vector_t::RowsAtCompileTime;
        vector_t l, u;
        if constexpr (rows == Eigen::Dynamic) {
            l = vector_t::Constant(x0.rows(), -std::numeric_limits<double>::infinity());
            u = vector_t::Constant(x0.rows(), std::numeric_limits<double>::infinity());
        } else {
            l = vector_t::Constant(-std::numeric_limits<double>::infinity());
            u = vector_t::Constant(std::numeric_limits<double>::infinity());
        }
        return optimize(std::forward<ObjectiveT>(objective), x0, l, u, std::forward<Callbacks>(callbacks)...);
    }

// Bounded (classical) version
template <typename ObjectiveT, typename T1, typename T2, typename... Callbacks,
        typename = std::enable_if_t<
            is_eigen_col_vector<std::decay_t<T1>>::value &&
            is_eigen_col_vector<std::decay_t<T2>>::value
        >>
vector_t optimize(ObjectiveT&& objective, const vector_t& x0, T1&& l, T2&& u, Callbacks&&... callbacks) {
    fdapde_static_assert(
      std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
      INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    std::tuple<Callbacks...> callbacks_ {callbacks...};

    // parameters for line-search
    const double c1 = 1e-4;       // Armijo constant
    const double backtrack = 0.5; // shrinkage factor on failed armijo
    const double alpha_min = 1e-20;
    const double f_tol = 1e-12;   // function value change tolerance

    // other parameters (same as in LBFGS)
    bool stop = false;   // asserts true in case of forced stop
    double error = std::numeric_limits<double>::max();
    double gamma = 1.0;
    int size = N == Dynamic ? x0.rows() : N;
    auto grad = objective.gradient();
    h = step_;
    n_iter_ = 0;

    // ensure initial point is feasible: clamp to bounds if necessary
    x_old = x0;
    for (int i = 0; i < size; ++i) {
        if (x_old[i] < l[i]) x_old[i] = l[i];
        if (x_old[i] > u[i]) x_old[i] = u[i];
    }

    // initial values for _new and _old variables (same as in LBFGS)
    x_new = vector_t::Constant(size, NaN);
    grad_old = grad(x_old);
    grad_new = vector_t::Constant(size, NaN);
    update = -grad_old; // initial descent

    // (same as in LBFGS)
    stop |= internals::exec_grad_hooks(*this, objective, callbacks_);
    // Compute projected gradient norm for stopping
    double proj_grad_norm = 0.0;
    for (int i = 0; i < size; ++i) {
        double gi = grad_old[i];
        if ((x_old[i] <= l[i] && gi > 0) || (x_old[i] >= u[i] && gi < 0)) {
            gi = 0.0;
        }
        proj_grad_norm += gi * gi;
    }
    error = std::sqrt(proj_grad_norm);

    values_.clear();
    values_.push_back(objective(x_old));
    x_mem_.resize(x0.rows(), mem_size_);
    grad_mem_.resize(x0.rows(), mem_size_);

    // initialize counter for filled memory slots for s_k and y_k
    int filled = 0;

    // helper function to compute free_mask given x (current point) and g (gradient eval at x)
    auto compute_free_mask = [&](const vector_t& x, const vector_t& g, std::vector<char>& free_mask) {
        free_mask.assign(size, 1); // default value (all directions are free)
        for (int i = 0; i < size; ++i) {
            if ((x[i] <= l[i] && g[i] >= 0.0) || (x[i] >= u[i] && g[i] <= 0.0)) {
                free_mask[i] = 0; // i-th coordinate fixed, not free this iteration (cannot move in that direction)
            }
        }
    };

    while (n_iter_ < max_iter_ && error > tol_ && !stop) {
        stop |= internals::exec_adapt_hooks(*this, objective, callbacks_);

        // identify free variables
        std::vector<char> free_mask;
        compute_free_mask(x_old, grad_old, free_mask);

        // Check if there are any free variables
        bool any_free = false;
        for (char c : free_mask) { if (c) { any_free = true; break; } }
        if (!any_free) {
            update.setZero();
            values_.push_back(objective(x_old));
            break;
        }

        // Compute the search direction in reduced the space
        std::vector<int> free_idx;
        free_idx.reserve(size);
        for (int i = 0; i < size; ++i) if (free_mask[i]) free_idx.push_back(i);
        const int nfree = static_cast<int>(free_idx.size());

        Eigen::VectorXd g_free(nfree);
        for (int i = 0; i < nfree; ++i) g_free[i] = grad_old[free_idx[i]];

        std::vector<Eigen::VectorXd> s_list; s_list.reserve(filled);
        std::vector<Eigen::VectorXd> y_list; y_list.reserve(filled);
        for (int idx = 0; idx < filled; ++idx) {
            int col = (n_iter_ + mem_size_ - 1 - idx) % mem_size_;
            Eigen::VectorXd s_free(nfree), y_free(nfree);
            for (int j = 0; j < nfree; ++j) {
                int orig = free_idx[j];
                s_free[j] = x_mem_(orig, col);
                y_free[j] = grad_mem_(orig, col);
            }
            s_list.push_back(std::move(s_free));
            y_list.push_back(std::move(y_free));
        }

        // Two-loop recursion
        Eigen::VectorXd q = g_free;
        std::vector<double> alpha(filled, 0.0);
        for (int i = 0; i < filled; ++i) {
            double denom = y_list[i].dot(s_list[i]);
            if (denom == 0.0) { alpha[i] = 0.0; continue; }
            alpha[i] = s_list[i].dot(q) / denom;
            q -= alpha[i] * y_list[i];
        }

        double gamma_red = 1.0;
        if (filled > 0) {
            double ydoty = y_list[0].squaredNorm();
            double sdoty = s_list[0].dot(y_list[0]);
            if (ydoty > 0.0) gamma_red = sdoty / ydoty;
        }

        Eigen::VectorXd update_free = -gamma_red * q;
        for (int i = filled - 1; i >= 0; --i) {
            double denom = y_list[i].dot(s_list[i]);
            double beta = 0.0;
            if (denom != 0.0) beta = y_list[i].dot(update_free) / denom;
            update_free -= s_list[i] * (alpha[i] + beta);
        }

        update.setZero();
        for (int i = 0; i < nfree; ++i) update[free_idx[i]] = update_free[i];

        double dirdotgrad = update.dot(grad_old);
        if (dirdotgrad >= 0.0) {
            for (int i = 0; i < nfree; ++i) update[free_idx[i]] = -g_free[i];
        }

        // Bound-aware line search
        double alpha_max = std::numeric_limits<double>::infinity();
        for (int i = 0; i < size; ++i) {
            if (update[i] > 0.0) {
                double a = (u[i] - x_old[i]) / update[i];
                if (a < alpha_max) alpha_max = a;
            } else if (update[i] < 0.0) {
                double a = (l[i] - x_old[i]) / update[i];
                if (a < alpha_max) alpha_max = a;
            }
        }
        if (!(alpha_max > 0.0)) alpha_max = 0.0;

        double alpha_trial = std::min(h, alpha_max);
        if (alpha_trial <= alpha_min || alpha_max <= 0.0) {
            values_.push_back(objective(x_old));
            break;
        }

        double f_old = values_.back();
        double gTp = grad_old.dot(update);
        bool armijo_ok = false;
        vector_t x_trial;
        double f_trial = 0.0;

        double alpha_cur = alpha_trial;
        while (alpha_cur > alpha_min) {
            x_trial = x_old + alpha_cur * update;
            for (int i = 0; i < size; ++i) {
                if (x_trial[i] < l[i]) x_trial[i] = l[i];
                if (x_trial[i] > u[i]) x_trial[i] = u[i];
            }
            f_trial = objective(x_trial);
            if (f_trial <= f_old + c1 * alpha_cur * gTp) {
                armijo_ok = true;
                break;
            }
            alpha_cur *= backtrack;
            if (alpha_cur > alpha_max) alpha_cur = alpha_max;
        }

        if (!armijo_ok) {
            if (alpha_max > alpha_min) {
                x_trial = x_old + alpha_max * update;
                for (int i = 0; i < size; ++i) {
                    if (x_trial[i] < l[i]) x_trial[i] = l[i];
                    if (x_trial[i] > u[i]) x_trial[i] = u[i];
                }
                f_trial = objective(x_trial);
                if (f_trial <= f_old) {
                    alpha_cur = alpha_max;
                    armijo_ok = true;
                } else {
                    values_.push_back(f_old);
                    break;
                }
            } else {
                values_.push_back(f_old);
                break;
            }
        }

        x_new = x_old + alpha_cur * update;
        for (int i = 0; i < size; ++i) {
            if (x_new[i] < l[i]) x_new[i] = l[i];
            if (x_new[i] > u[i]) x_new[i] = u[i];
        }

        grad_new = grad(x_new);

        vector_t s_k = x_new - x_old;
        vector_t y_k = grad_new - grad_old;

        const double curvature_eps = 1e-10; // looser curvature condition
        double sty = s_k.dot(y_k);
        if (sty > curvature_eps) {
            int col_idx = n_iter_ % mem_size_;
            x_mem_.col(col_idx) = s_k;
            grad_mem_.col(col_idx) = y_k;
            if (filled < mem_size_) ++filled;
            double ynorm = grad_mem_.col(col_idx).norm();
            if (ynorm > 0.0) gamma = x_mem_.col(col_idx).dot(grad_mem_.col(col_idx)) / ynorm;
        }

        ++n_iter_;
        x_old = x_new;
        grad_old = grad_new;

        // Compute projected gradient norm for stopping
        proj_grad_norm = 0.0;
        for (int i = 0; i < size; ++i) {
            double gi = grad_new[i];
            if ((x_new[i] <= l[i] && gi > 0) || (x_new[i] >= u[i] && gi < 0)) {
                gi = 0.0;
            }
            proj_grad_norm += gi * gi;
        }
        error = std::sqrt(proj_grad_norm);

        values_.push_back(f_trial);

        // Additional stopping: function value change
        double f_change = std::abs(f_trial - f_old);
        if (f_change < f_tol) break;

        stop |= (internals::exec_grad_hooks(*this, objective, callbacks_) || internals::exec_stop_if(*this, objective));
    }

    optimum_ = x_old;
    value_ = values_.empty() ? objective(x_old) : values_.back();
    return optimum_;
}

// observers
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_LBFGSB_H__