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
#include "more_thuente.h"
#include <algorithm>
#include "lbfgsb_params.h"

namespace fdapde {

// Helper traits to detect Eigen column vector
template <typename T>
struct is_eigen_col_vector : std::false_type {};
template <typename Scalar, int Rows, int Options, int MaxRows>
struct is_eigen_col_vector<Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, 1>> : std::true_type {};

template <int N> class LBFGSB {
   public:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;
    using GradientFunction = std::function<vector_t(const vector_t&)>; // exact gradient function type

   private:
    vector_t optimum_;                                          // minimum point
    double value_;                                              // value of objective function in minimum point
    int n_iter_ = 0;                                            // number of iterations
    std::vector<double> values_;                                // value of objective function in points visited during minimization
    std::vector<vector_t> trajectory_;                          // points visited during minimization
    vector_t l_, u_;                                            // lower and upper bounds
    vector_t last_direction_;                                   // most recent direction in case fallback is needed
    Eigen::Matrix<double, Dynamic, Dynamic> grad_mem_, x_mem_;  // memory for Hessian approximation

    LBFGSBParams params_;                                       // parameters for the LBFGSB algorithm

   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;

    // constructors
    LBFGSB(LBFGSBParams params) : params_(params) {
        params_.check_params();
    }
    LBFGSB(const LBFGSB& other) {
        params_.max_iter_ = other.params_.max_iter_;
        params_.epsilon_ = other.params_.epsilon_;
        params_.step_ = other.params_.step_;
        params_.mem_size_ = other.params_.mem_size_;
        params_.c1_ = other.params_.c1_;
        params_.c2_ = other.params_.c2_;
        params_.backtrack_ = other.params_.backtrack_;
        params_.alpha_min_ = other.params_.alpha_min_;
        params_.f_tol_ = other.params_.f_tol_;
    }
    LBFGSB& operator=(const LBFGSB& other) {
        params_.max_iter_ = other.params_.max_iter_;
        params_.epsilon_ = other.params_.epsilon_;
        params_.step_ = other.params_.step_;
        params_.mem_size_ = other.params_.mem_size_;
        params_.c1_ = other.params_.c1_;
        params_.c2_ = other.params_.c2_;
        params_.backtrack_ = other.params_.backtrack_;
        params_.alpha_min_ = other.params_.alpha_min_;
        params_.f_tol_ = other.params_.f_tol_;
        return *this;
    }

    // Unbounded version (set bounds to +/- infty and then rely on the bounded version)
    template <typename ObjectiveT, typename... Callbacks,
            typename = std::enable_if_t<
                sizeof...(Callbacks) == 0 ||
                !is_eigen_col_vector<std::decay_t<std::tuple_element_t<0, std::tuple<Callbacks...>>>>::value
            >>
    vector_t optimize(ObjectiveT&& objective, GradientFunction exact_gradient, const vector_t& x0, Callbacks&&... callbacks) {
        constexpr int rows = vector_t::RowsAtCompileTime;
        if constexpr (rows == Eigen::Dynamic) {
            l_ = vector_t::Constant(x0.rows(), -std::numeric_limits<double>::infinity());
            u_ = vector_t::Constant(x0.rows(), std::numeric_limits<double>::infinity());
        } else {
            l_ = vector_t::Constant(-std::numeric_limits<double>::infinity());
            u_ = vector_t::Constant(std::numeric_limits<double>::infinity());
        }
        return optimize(std::forward<ObjectiveT>(objective), exact_gradient, x0, l_, u_, std::forward<Callbacks>(callbacks)...);
    }

    template <typename ObjectiveT, typename T1, typename T2, typename... Callbacks,
            typename = std::enable_if_t<
                is_eigen_col_vector<std::decay_t<T1>>::value &&
                is_eigen_col_vector<std::decay_t<T2>>::value
            >>
    vector_t optimize(ObjectiveT&& objective, GradientFunction exact_gradient, const vector_t& x0, T1&& l, T2&& u, Callbacks&&... callbacks) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        std::tuple<Callbacks...> callbacks_ {callbacks...};

        // Store bounds
        l_ = l;
        u_ = u;      

        bool stop = false;
        double error = std::numeric_limits<double>::max();
        double gamma = 1.0;
        int size = N == Dynamic ? x0.rows() : N;
        
        // use exact gradient if provided otherwise use finite differences
        auto grad = exact_gradient ? exact_gradient : objective.gradient();
        h = params_.step_;
        n_iter_ = 0;

        // ensure initial point is feasible (if not, project to bounds)
        x_old = x0;
        project_to_feasible(x_old);

        // initialize variables
        x_new = vector_t::Constant(size, NaN);
        grad_old = grad(x_old);
        grad_new = vector_t::Constant(size, NaN);
        update = -grad_old;
        last_direction_ = update;

        stop |= internals::exec_grad_hooks(*this, objective, callbacks_);
        
        // Compute projected gradient norm for stopping
        double proj_grad_norm = compute_projected_gradient_norm(x_old, grad_old);
        error = proj_grad_norm;
        
        // Set up storages for values and visited points and memory
        values_.clear();
        values_.push_back(objective(x_old));
        trajectory_.clear();
        trajectory_.push_back(x_old);
        x_mem_.resize(x0.rows(), params_.mem_size_);
        grad_mem_.resize(x0.rows(), params_.mem_size_);

        int filled = 0;

        // Main optimization loop
        while (n_iter_ < params_.max_iter_ && error > params_.epsilon_ && !stop) {
            stop |= internals::exec_adapt_hooks(*this, objective, callbacks_);

            // Compute search direction
            compute_search_direction(filled);

            // Check if direction is valid
            double dg = update.dot(grad_old);
            if (dg >= 0.0) {
                update = -grad_old;
                filled = 0;
                dg = update.dot(grad_old);
            }

            // Compute maximum step size
            double alpha_max = compute_max_step_size(x_old, update);
            if (!(alpha_max > 0.0)) alpha_max = 0.0;

            double alpha_trial = std::min(h, alpha_max);
            if (alpha_trial <= params_.alpha_min_ || alpha_max <= 0.0) {
                values_.push_back(objective(x_old));
                trajectory_.push_back(x_old);
                break;
            }

            // Use More-Thuente line search
            MoreThuenteLineSearch mtls(alpha_trial, params_.c1_, params_.c2_);
            mtls.set_max_step(std::max(alpha_max, alpha_trial));
            mtls.set_min_step(params_.alpha_min_);
            mtls.set_wolfe_constants(params_.c1_, params_.c2_);

            mtls.adapt_hook(*this, objective);

            // Determine whether line search produced a valid step
            bool ls_ok = true;
            if (!(h > params_.alpha_min_) || !std::isfinite(h)) ls_ok = false;
            for (int i = 0; i < size && ls_ok; ++i) {
                if (!std::isfinite(x_new[i])) ls_ok = false;
            }

            if (!ls_ok) {
                // Fallback backtracking Armijo
                double f_old = values_.back();
                double gTp = grad_old.dot(update);
                bool armijo_ok = false;
                vector_t x_trial;
                double f_trial = 0.0;

                double alpha_cur = std::min(h > 0.0 ? h : alpha_trial, alpha_max > 0.0 ? alpha_max : alpha_trial);
                if (alpha_cur <= 0.0) alpha_cur = alpha_trial;
                while (alpha_cur > params_.alpha_min_) {
                    x_trial = x_old + alpha_cur * update;
                    project_to_feasible(x_trial);
                    f_trial = objective(x_trial);
                    if (f_trial <= f_old + params_.c1_ * alpha_cur * gTp) {
                        armijo_ok = true;
                        break;
                    }
                    alpha_cur *= params_.backtrack_;
                    if (alpha_cur > alpha_max) alpha_cur = alpha_max;
                }

                if (!armijo_ok) {
                    if (alpha_max > params_.alpha_min_) {
                        x_trial = x_old + alpha_max * update;
                        project_to_feasible(x_trial);
                        f_trial = objective(x_trial);
                        if (f_trial <= f_old) {
                            alpha_cur = alpha_max;
                            armijo_ok = true;
                        } else {
                            values_.push_back(f_old);
                            trajectory_.push_back(x_old);
                            break;
                        }
                    } else {
                        values_.push_back(f_old);
                        trajectory_.push_back(x_old);
                        break;
                    }
                }

                x_new = x_old + alpha_cur * update;
                project_to_feasible(x_new);
                grad_new = grad(x_new);
            }

            double f_old = values_.back();
            double f_trial = objective(x_new);

            // Update BFGS memory
            vector_t s_k = x_new - x_old;
            vector_t y_k = grad_new - grad_old;

            const double curvature_eps = 1e-12;
            double sty = s_k.dot(y_k);
            if (sty > curvature_eps) {
                int col_idx = n_iter_ % params_.mem_size_;
                x_mem_.col(col_idx) = s_k;
                grad_mem_.col(col_idx) = y_k;
                if (filled < params_.mem_size_) {
                    ++filled;
                }
                double ynorm = grad_mem_.col(col_idx).norm();
                if (ynorm > 0.0) {
                    gamma = x_mem_.col(col_idx).dot(grad_mem_.col(col_idx)) / ynorm;
                }
            }

            ++n_iter_;
            x_old = x_new;
            grad_old = grad_new;
            last_direction_ = update;

            proj_grad_norm = compute_projected_gradient_norm(x_new, grad_new);
            error = proj_grad_norm;

            values_.push_back(f_trial);
            trajectory_.push_back(x_new);
            
            // Additional stopping: function value change
            double f_change = std::abs(f_trial - f_old);
            if (f_change < params_.f_tol_) {
                break;
            }

            stop |= (internals::exec_grad_hooks(*this, objective, callbacks_) || internals::exec_stop_if(*this, objective));
        }

        optimum_ = x_old;
        value_ = values_.empty() ? objective(x_old) : values_.back();
        return optimum_;
    }
    
private:
    // Helper to ensure point is within bounds
    void project_to_feasible(vector_t& x) {
        for (int i = 0; i < x.size(); ++i) {
            if (x[i] < l_[i]) x[i] = l_[i];
            if (x[i] > u_[i]) x[i] = u_[i];
        }
    }
    
    double compute_projected_gradient_norm(const vector_t& x, const vector_t& g) {
        return ((x - g).cwiseMax(l_).cwiseMin(u_) - x).cwiseAbs().maxCoeff();
    }

    double compute_max_step_size(const vector_t& x, const vector_t& direction) {
        double alpha_max = std::numeric_limits<double>::infinity();
        for (int i = 0; i < x.size(); ++i) {
            if (direction[i] > 0.0) {
                double a = (u_[i] - x[i]) / direction[i];
                if (a < alpha_max) alpha_max = a;
            } else if (direction[i] < 0.0) {
                double a = (l_[i] - x[i]) / direction[i];
                if (a < alpha_max) alpha_max = a;
            }
        }
        return alpha_max;
    }

    void compute_search_direction(int filled) {
        int size = x_old.size();
        
        std::vector<char> free_mask(size, 1);
        std::vector<int> free_idx;
        free_idx.reserve(size);
        
        for (int i = 0; i < size; ++i) {
            if ((x_old[i] <= l_[i] && grad_old[i] >= 0.0) || (x_old[i] >= u_[i] && grad_old[i] <= 0.0)) {
                free_mask[i] = 0;
            } else {
                free_idx.push_back(i);
            }
        }

        const int nfree = static_cast<int>(free_idx.size());
        if (nfree == 0) {
            update.setZero();
            return;
        }

        Eigen::VectorXd g_free(nfree);
        for (int i = 0; i < nfree; ++i) {
            g_free[i] = grad_old[free_idx[i]];
        }

        Eigen::VectorXd q = g_free;
        std::vector<double> alpha(filled, 0.0);
        
        for (int i = 0; i < filled; ++i) {
            int col_idx = (n_iter_ + params_.mem_size_ - 1 - i) % params_.mem_size_;
            Eigen::VectorXd s_free(nfree), y_free(nfree);
            for (int j = 0; j < nfree; ++j) {
                int orig_idx = free_idx[j];
                s_free[j] = x_mem_(orig_idx, col_idx);
                y_free[j] = grad_mem_(orig_idx, col_idx);
            }
            double denom = y_free.dot(s_free);
            if (std::abs(denom) < 1e-20) { alpha[i] = 0.0; continue; }
            double rho = 1.0 / denom;
            alpha[i] = rho * s_free.dot(q);
            q -= alpha[i] * y_free;
        }

        Eigen::VectorXd r = q;
        if (filled > 0) {
            int latest_col = (n_iter_ + params_.mem_size_ - 1) % params_.mem_size_;
            Eigen::VectorXd s_latest(nfree), y_latest(nfree);
            for (int j = 0; j < nfree; ++j) {
                int orig_idx = free_idx[j];
                s_latest[j] = x_mem_(orig_idx, latest_col);
                y_latest[j] = grad_mem_(orig_idx, latest_col);
            }
            double yty = y_latest.squaredNorm();
            double sty = s_latest.dot(y_latest);
            if (yty > 0) {
                r = (sty / yty) * q;
            }
        }

        for (int i = filled - 1; i >= 0; --i) {
            int col_idx = (n_iter_ + params_.mem_size_ - 1 - i) % params_.mem_size_;
            Eigen::VectorXd s_free(nfree), y_free(nfree);
            for (int j = 0; j < nfree; ++j) {
                int orig_idx = free_idx[j];
                s_free[j] = x_mem_(orig_idx, col_idx);
                y_free[j] = grad_mem_(orig_idx, col_idx);
            }
            double denom = y_free.dot(s_free);
            if (std::abs(denom) < 1e-20) {
                r += s_free * (alpha[i]); // skip beta safely
            } else {
                double rho = 1.0 / denom;
                double beta = rho * y_free.dot(r);
                r += s_free * (alpha[i] - beta);
            }
        }

        update.setZero();
        for (int i = 0; i < nfree; ++i) {
            update[free_idx[i]] = -r[i];
        }
    }

public:
    // observers
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
    const std::vector<double>& values() const { return values_; }
    const std::vector<vector_t>& trajectory() const { return trajectory_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_LBFGSB_H__