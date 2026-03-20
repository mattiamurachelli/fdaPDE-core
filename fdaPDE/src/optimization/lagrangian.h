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


#ifndef __FDAPDE_Lagrangian_H__
#define __FDAPDE_Lagrangian_H__

#include "lagrangian_objective.h"

namespace fdapde {

template<int N, typename Optimizer>
class Lagrangian {
private:
    using vector_t = std::conditional_t<N == Eigen::Dynamic,Eigen::Matrix<double, Eigen::Dynamic, 1>,Eigen::Matrix<double, N, 1>>;

    double mu_ = 1e2;                       // Initial (maximum) penalty parameter
    double min_mu_ = 1e-2;                  // Minimum penalty parameter
    double scaling_factor_ = 0.75;          // Scaling factor for penalty parameter update
    std::vector<int> num_iter_{};           // Number of iterations for each subproblem
    int max_iter_ = 500;                    // Maximum number of subproblems
    double tol_ = 1e-8;                     // Tolerance for convergence check on Lagrangian gradient update
    double tau_ = 0.25;                     // Parameter for mu update based on residual decrease

    std::vector<vector_t> optimum_{};       // Optimal solution for each subproblem
    std::vector<double> values_{};          // Objective function values at the optimal solution for each subproblem

    Optimizer optimizer_;                   // Optimizer for solving uncostrained subproblems

public:
    // Constructor 1 (Only optimizer, default parameters)
    Lagrangian(const Optimizer& optimizer) : optimizer_(optimizer) {}
    // Constructor 2 (Set also max_iter and tolerance)
    Lagrangian(const Optimizer& optimizer, const int max_iter, const double tol)
        : optimizer_(optimizer), max_iter_(max_iter), tol_(tol) {}
    // Constructor 3 (Set also mu, min_mu scaling_factor and tau)
    Lagrangian(const Optimizer& optimizer, const int max_iter, const double tol,
        const double mu, double const min_mu, double const scaling_factor, double const tau)
        : optimizer_(optimizer), max_iter_(max_iter), tol_(tol),
        mu_(mu), min_mu_(min_mu), scaling_factor_(scaling_factor), tau_(tau) {}
    // Solve method for problem resolution
    template <typename ObjectiveT, typename ConstraintT, typename... Callbacks>
    vector_t solve(ObjectiveT&& objective, const ConstraintT& constraint, const std::vector<bool>& ineq_constr, 
        const vector_t& x0, Callbacks&&... callbacks) {
        // Copy x0 to a local variable since x0 is passed by const reference
        vector_t x = x0;

        // Create a variable to store constraint residual for mu updates and
        // one to store the gradient norm for convergence check
        // We set them at the maximum value
        double r_k = std::numeric_limits<double>::max();
        double res = 0.0;
        double g_k = std::numeric_limits<double>::max();

        // Compute the constrained violation for the initial point
        for(std::size_t i = 0; i < constraint.size(); ++i) {
            if (ineq_constr[i] == true) {               // Inequality constraint
                if (constraint[i](x) > 0 ) {            // Add a term only if constraint is violated
                    res += constraint[i](x)*constraint[i](x);
                }
            } else {                                    // Equality constraint
                res += constraint[i](x)*constraint[i](x);
            }
        }
        res = std::sqrt(res);

        // Create a vector of Lagrange multipliers, we initialize it to zero for all constraints
        // This is a common choice in many libraries, but other initializations could be performed
        std::vector<double> lambda(constraint.size(), 0.0);

        // Main loop of the Augmented Lagrangian method
        for (int k = 0; k < max_iter_; ++k) {
            // Create the Lagrangian objective function for the current iteration
            // Remark : We minimize this function wrt x, lambda and mu are fixed for the current iteration
            LagrangianObjective<N, std::decay_t<ObjectiveT>> lagrangian_objective(objective, lambda, constraint, ineq_constr, mu_);

            // Adjust the tolerance of the optimizer for the current subproblem
            // We want to solve subproblems with increasing accuracy to avoid getting caught in local minima of the
            // unconstrained problem that may keep us away from the solution of the constrained problem
            optimizer_.set_tol(std::min(1e-4, std::max(res, tol_)));
            //Debugging step
            std::cout << "Using tolerance " << std::min(1e-4, std::max(res, tol_)) << "\n";
            // Solve the current subproblem using the optimizer
            optimizer_.optimize(lagrangian_objective, x, std::forward<Callbacks>(callbacks)...);
            
            // Extract and store results for the current iteraiton
            num_iter_.push_back(optimizer_.n_iter());       // Number of iterations for the current subproblem
            optimum_.push_back(optimizer_.optimum());       // Optimal solution for the current subproblem
            x = optimizer_.optimum();                       // Optimal solution for current subproblem is starting point for next iteration
            values_.push_back(objective(x));                // Objective function value at the optimal solution for current subproblem
            
            // Compute the constraint residual for updates on mu_ and tol
            res = 0.0;
            for(std::size_t i = 0; i < constraint.size(); ++i) {
                if (ineq_constr[i] == true) {               // Inequality constraint
                    if (constraint[i](x) > 0 ) {            // Add a term only if constraint is violated
                        res += constraint[i](x)*constraint[i](x);
                    }
                } else {                                    // Equality constraint
                    res += constraint[i](x)*constraint[i](x);
                }
            }    
            res = std::sqrt(res);

            // Update Lagrange multipliers (Nocedal & Wright, 17.49)
            for (std::size_t j = 0; j < constraint.size(); ++j) {
                if (ineq_constr[j] == true) {                   // Inequality constraint
                    lambda[j] = std::max(0.0 , lambda[j] + mu_ * constraint[j](x));
                } else {                                        // Equality constraint
                    lambda[j] = lambda[j] - constraint[j](x) / mu_;
                }
            }
                
            // Update penalty parameter (Avoid it becoming too small to prevent numerical issues!)
            
            if(res > tau_*r_k) { // the residual has not decreased sufficiently, we increase the penalty (reduce mu_)
                mu_ = std::max(mu_*scaling_factor_, min_mu_);
            } // else do nothing, mu stays the same
            r_k = res;
                     
            // Check convergence using gradient of Lagrangian objective function and constraint violation
            // We would like to be in a stationary point of the Lagrangian, which is a necessary condition for optimality
            auto grad = lagrangian_objective.gradient();
            const double grad_norm = grad(x).norm();

            // Debugging step
            std::cout << "Constraint violation: " << res << ", mu: " << mu_ << ", Gradient norm: " << grad_norm << std::endl;

            // End condition
            if ( grad_norm + res < tol_) { break; }
            g_k = grad_norm;

        }

        return x;
    }
    // Observers
    std::vector<int> num_iter() const { return num_iter_; }             // Number of iterations
    const std::vector<vector_t>& optimum() const { return optimum_; }   // Optimal solutions
    const std::vector<double>& values() const { return values_; }       // Objective function values at the optimal solutions
};

} // namespace fdapde

#endif