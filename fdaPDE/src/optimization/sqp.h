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

#ifndef __FDAPDE_SQP_H__
#define __FDAPDE_SQP_H__

#include <vector>
#include <type_traits>
#include <Eigen/Core>

namespace fdapde {

template<int N, typename ObjectiveT, typename ConstraintT>
class MeritFunction {
private:
    using vector_t = std::conditional_t<N == Eigen::Dynamic, Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    const ObjectiveT objective_;                        // Objective function f(x)
    const ConstraintT& constraints_;                    // Constraint functions c_k(x)
    double mu_;                                         // Penalty parameter

public:
    // Constructor
    MeritFunction(const ObjectiveT& objective, const ConstraintT& constraints) 
        : objective_(objective), constraints_(constraints) {}
    // Call operator 
    double operator()(const vector_t& x) const {

        double res = objective(x);

        for(int i = 0; i < constraints.size(); ++i) {
            res += (1/mu_) * std::abs(constraints[i](x));
        }

        return res;
    }
    // Directional derivative in p direction method
    double derivative(const vector_t& x, const vector_t p) {

        // TO BE DONE
        double res = 
    }
    
    // Setters for lambda and mu
    void set_mu(const double mu) {mu_ = mu;}
};

template<int N, typename Optimizer>
class SQP {
private:
    using vector_t = std::conditional_t<N == Eigen::Dynamic,Eigen::Matrix<double, Eigen::Dynamic, 1>,Eigen::Matrix<double, N, 1>>;

    double mu_ =;                       // Initial (maximum) penalty parameter
    double min_mu_ =;                  // Minimum penalty parameter
    size_t num_iter_{};                 // Number of iterations for each subproblem
    int max_iter_ = ;                    // Maximum number of subproblems
    double tol_ = ;                     // Tolerance for convergence check on Lagrangian gradient update
    double tau_ =;                     // Parameter for mu update based on residual decrease
    double eta_ =;

    std::vector<vector_t> optimum_{};       // Optimal solution for each subproblem
    std::vector<double> values_{};          // Objective function values at the optimal solution for each subproblem

    // Function to compute the gradient of the Lagrangian (required in solve() method)
    template <ObjectiveT, ConstraintT>
    vector_t lagrangian_gradient(const vector_t& x, const std::vector>double>& lambda,
        const ObjectiveT& objective, const ConstraintT& constraints) {

        double res = objective.gradient(x);

        for(int i = 0; i < constraints.size(); ++i) {
            if(constraints[i].is_inequality) {              // Inequality constraints
                res += lambda[i]*constraints[i].gradient(x);
            } else {                                        // Equality constraints
                res -= lambda[i]*constraints[i].gradient(x);
            }
        }

        return res;
    }

public:
    // Constructors
    // TO BE DONE!

    // Solve method for problem resolution
    template <typename ObjectiveT, typename ConstraintT>
    vector_t solve(ObjectiveT&& objective, const ConstraintT& constraints, const vector_t& x0) {

        // Static assertions to check that the objective function is callable at vector_t and return double
        fdapde_static_assert(
            std::is_same<decltype(std::declval<std::decay_t<ObjectiveT>>().operator()(vector_t())) FDAPDE_COMMA double>
            ::value,INVALID_CALL_TO_SOLVE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE
        );

        // Extract the type of a single constraint from the constraints container
        using constraint_t = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<const ConstraintT&>()
            [std::declval<std::size_t>()])>>;

        // Static assertions to check that the constraint functor is callable at vector_t and return double
        fdapde_static_assert(
            std::is_same<decltype(std::declval<constraint_t>().operator()(vector_t())) FDAPDE_COMMA double>
            ::value,INVALID_CALL_TO_SOLVE__CONSTRAINT_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE
        );

        // Copy x0 to a local variable since x0 is passed by const reference
        // and declare also x_new
        vector_t x_old = x0;
        vector_t x_new = x_old;

        // Declare the variable for the step
        vector_t p_k;

        // Declare the L1 merit function for the problem
        MeritFunction<N, ObjectiveT, ConstraintT> phi(objective, constraints);

        // Other necessary variables for the algorithm
        double gamma = 0;                           // Used in mu update
        double alpha_k = 1;                         // Used for step length computation
        double theta_k = 1;                         // Used in Algorithm 18.2
        double temp1;                               // Temporary support double variable 1
        double temp2;                               // Temporary support double variable 2
        Eigen::VectorXd rhs;                        // Useful for lambda update computation
        Eigen::MatrixXd M;                          // Useful for lambda update computation
        vector_t s_k;                               // Useful for Hessian approximation (BFGS update)
        vector_t y_k;                               // Useful for Hessian approximation (BFGS update)
        vector_t r_k;                               // Useful for Hessian approximation (BFGS update)

        // Create a vector of Lagrange multipliers, we initialize it to zero for all constraints
        // This is a common choice in many libraries, but other initializations could be performed
        std::vector<double> lambda(constraints.size(), 0.0);

        // Create the approximation of the Hessian of the Lagrangian, we initialize it to the identity matrix for the first iteration
        Eigen::Matrix<double, N, N> B_k = Eigen::Matrix<double, N, N>::Identity();

        // Evaluate f(x0), grad(f(x0)), c_i(x0) and A(x0)
        double f_k = objective(x_old);
        vector_t grad_f_k = objective.gradient()(x_old);
        std::vector<double> c_k(constraints.size());
        for(std::size_t i = 0; i < constraints.size(); ++i) { c_k[i] = constraints[i](x_old); }
        Eigen::Matrix<double, constraints.size(), N> A_k;
        for(std::size_t i = 0; i < constraints.size(); ++i) { A_k.row(i) = constraints[i].gradient(x_old).transpose(); }

        // Main loop of the SQP method
        for (int k = 0; k < max_iter_; ++k) {
            // Check termination condition
            // to be determined, ideas : length of the step, gradient of lagrangian, complementarity conditions, constraint residual
            if() {break;}

            // Solve the current quadratic problem
            p_k = solve_problem(B_k, grad_f_k, c_k, A_k);

            // Choose mu such that p_k is a descent direction for the merit function at x_k
            // We begin by computing gamma
            for(int i = 0; i < constraints.size(); ++i) {
                if(std::abs(lambda[i]) > gamma) {gamma = std::abs(lambda[i]); }
            }

            // Now we can compute the update for mu
            if(1/mu_ < gamma + 1e-2) { mu_ = 1/(gamma + 2e-2); }

            // Compute step length
            alpha_k = 1;
            phi.set_mu(mu_);    // Update the merit function if necessary
            while(phi(x_old + alpha_k * p_k) > phi(x_old) + phi.derivative(x_old, p_k)) { alpha_k = alpha_k * tau_;}

            // Compute next point
            x_new = x_old + alpha_k * p_k;

            // Evaluate f(x_k), grad(f(x_k)), c_i(x_k) and A(x_k)
            f_k = objective(x_new);
            grad_f_k = objective.gradient()(x_new);
            for(std::size_t i = 0; i < constraints.size(); ++i) { c_k[i] = constraints[i](x_new); }
            for(std::size_t i = 0; i < constraints.size(); ++i) { A_k.row(i) = constraints[i].gradient()(x_new).transpose(); }

            // Compute Lagrange multipliers update
            rhs = - A_k * grad_f_k;
            M = A_k * A_k.transpose();
            lambda = M.ldlt().solve(rhs);

            // Hessian approximation (BFGS update)
            s_k = x_new - x_old;
            y_k = lagrangian_gradient<ObjectiveT, ConstraintT>(x_new, lambda, objective, constraints)
                - lagrangian_gradient<ObjectiveT, ConstraintT>(x_old, lambda, objective, constraints);

            // Algorithm 18.2 (Damped BFGS Updating for SQP)
            // First compute theta_k
            // temp1 = s_k^t y_k
            // temp2 = s_k^t B_k s_k
            temp1 = s_k.transpose() * y_k;
            temp2 = s_k.transpose() * B_k * s_k;

            if(temp1 < 0.2 * temp2) {
                theta_k = (0.8 * temp2)/(temp2 - temp1);
            } else {
                theta_k = 1;
            }

            // Compute r_k
            r_k = theta_k * y_k + (1 - theta_k) * B_k * s_k;

            // Compute B_k

            B_k = B_k - (B_k * (s_k * s_k.transpose()) * B_k)/temp2 + (r_k * r_k.tranpose())/(s_k.transpose() * r_k);

            // Update x_old
            x_old = x_new;

        }

        return x_k;
    }

    // Observers
    size_t num_iter() const { return num_iter_; }                       // Number of iterations
    const std::vector<vector_t>& optimum() const { return optimum_; }   // Optimal solutions
    const std::vector<double>& values() const { return values_; }       // Objective function values at the optimal solutions
};

} // namespace fdapde

#endif