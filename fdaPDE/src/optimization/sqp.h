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

        double res = objective_(x);

        for(int i = 0; i < constraints_.size(); ++i) {
            if(constraints_[i].is_inequality_) {            // Inequality constraints
                res += (1/mu_) * std::max(0.0, constraints_[i](x));
            } else {                                      // Equality constraints
                res += (1/mu_) * std::abs(constraints_[i](x));
            }  
        }

        return res;
    }

    // Directional derivative in p direction method
    double derivative(const vector_t& x, const vector_t& p) const {
        double res = objective_.gradient(x).transpose() * p;
        const double eps = 1e-12;

        for (int i = 0; i < constraints_.size(); ++i) {
            double ci  = constraints_[i](x);
            double dci = constraints_[i].gradient(x).transpose() * p;

            if (constraints_[i].is_inequality_) {
                if (ci > eps) {
                    res += (1.0 / mu_) * dci;
                } else if (std::abs(ci) <= eps) {
                    res += (1.0 / mu_) * std::max(0.0, dci);
                }
            } else {
                if (ci > eps) {
                    res += (1.0 / mu_) * dci;
                } else if (ci < -eps) {
                    res += (1.0 / mu_) * (-dci);
                } else {
                    res += (1.0 / mu_) * std::abs(dci);
                }
            }
        }

        return res;
    }   
    
    // Setter for mu
    void set_mu(const double mu) {mu_ = mu;}
};

template<int N>
class SQP {
private:
    using vector_t = std::conditional_t<N == Eigen::Dynamic, Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    double mu_ = 1e2;                       // Initial penalty parameter
    int max_iter_ = 500;                    // Maximum number of subproblems
    int max_iter_subproblem_ = 20;          // Maximum number of iterations per subproblem
    double feasibility_tol_ = 1e-8;         // Tolerance for convergence check on residual
    double stationarity_tol_ = 1e-6;        // Tolerance for convergence check on lagrangian gradient
    double tau_ = 0.75;                     // Parameter for alpha reduction during LineSearch
    double eta_ = 0.25;                     // Parameter for LineSearch

    std::vector<int> num_iter_{};           // Number of iterations for each subproblem
    std::vector<vector_t> optimum_{};       // Optimal solution for each subproblem
    std::vector<double> values_{};          // Objective function values at the optimal solution for each subproblem

    // Function to clear data in order to perform multiple simulations in a row
    void clearData() {
        num_iter_.clear();
        optimum_.clear();
        values_.clear();
    }

    // ACCESSORY FUNCTIONS FOR SOLVE METHOD

    // Function to compute the gradient of the Lagrangian
    template <typename ObjectiveT, typename ConstraintT>
    vector_t lagrangian_gradient(const vector_t& x, const Eigen::Matrix<double, Eigen::Dynamic, 1>& lambda,
        const ObjectiveT& objective, const ConstraintT& constraints) {

        vector_t res = objective.gradient(x);

        for(int i = 0; i < constraints.size(); ++i) {
           res += lambda[i] * constraints[i].gradient(x);
        }

        return res;
    }

    template<typename ConstraintT>
    vector_t compute_feasible_point(const vector_t& x0, const ConstraintT& constraints) {
        // We solve an unconstrained problem in order to find a feasible starting point
        // for our local QP

        // Construct the objective function
        ScalarField<N> obj;
        obj = [&constraints] (const vector_t& x) -> double {
            double res = 0;
            for(int i = 0; i < constraints.size(); ++i) {
                if(constraints[i].is_inequality_) {                // Inequality constraints
                    res += std::max(0.0, constraints[i](x));
                } else {                                          // Equality constraints
                    res += constraints[i](x)*constraints[i](x);
                }
            }
            return res;
        };

        // And now minimize it
        GradientDescent<N> optimizer;
        optimizer.set_tol(1e-10);
        return optimizer.optimize(obj, x0, BacktrackingLineSearch());
    }
    
    vector_t solve_problem(Eigen::Matrix<double, N, N>& B_k, vector_t& grad_f_k,
        Eigen::Matrix<double, Eigen::Dynamic, 1> &c_k, Eigen::Matrix<double, Eigen::Dynamic , N> A_k, std::vector<bool>& inequality_flag,
        Eigen::Matrix<double, Eigen::Dynamic, 1>& lambda, const vector_t& x0) {

        // Copy x0 to a local variable since x0 is passed by const reference
        vector_t x_k = vector_t::Zero(N, 1);

        // Create solution vectors
        Eigen::Matrix<double, Eigen::Dynamic, 1> solution;      
        vector_t p_k;
        Eigen::Matrix<double, Eigen::Dynamic, 1> lambda_w;              // working set lagrange multipliers

        // Other useful variables
        double alpha_k = 1;
        double temp;
        int blocking_constraint = -1;
        double current_inf_w;
        int to_erase_w;
        int iteration_counter = 0;

        // Create and set-up the working set
        std::vector<int> working_set{};
        // We initialize it with equality constraints and active inequality constraints
        for(int i = 0; i < inequality_flag.size(); ++i) {
            if(!inequality_flag[i]) {                                   // Equality constraints
                working_set.push_back(i);
            } else {                                                    
                if(c_k(i) >= -1e-6) {                                   // Inequality ACTIVE constraints
                    working_set.push_back(i);
                }
            }
        }
        
        while(iteration_counter <= max_iter_subproblem_) {
            // We initialize the blocking constraint to -1, meaning that there is no blocking constraint for the moment
            blocking_constraint = -1;
            // Update iteration counter
            iteration_counter++;
            // Find p_k. We directly solve the KKT system
            // We first assemble the matrices
            // We begin by extracting the submatrices and subvectors associated to the working set
            int m_w = working_set.size();                                 // working set dimension
            int counter = 0;
            Eigen::Matrix<double, Eigen::Dynamic, N> A_k_w(m_w, N);       // Jacobian restriction
            Eigen::Matrix<double, Eigen::Dynamic, 1> c_k_w(m_w);          // Constraints restriction
            for(auto value : working_set) {
                A_k_w.row(counter) = A_k.row(value);
                c_k_w(counter) = c_k(value);
                ++counter;
            }
            // Now we assign the blocks to their positions
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> KKT(N + m_w, N + m_w);
            Eigen::Matrix<double, Eigen::Dynamic, 1> rhs(N + m_w);
            // Matrix
            KKT.topLeftCorner(N, N) = B_k;
            KKT.topRightCorner(N, m_w) = A_k_w.transpose();
            KKT.bottomLeftCorner(m_w, N) = A_k_w;
            KKT.bottomRightCorner(m_w, m_w).setZero();
            // Rhs
            rhs.head(N) = -grad_f_k - B_k*x_k;
            rhs.tail(m_w) = -c_k_w - A_k_w*x_k;
            // Solve the system
            solution = KKT.ldlt().solve(rhs);
            // Extract result
            p_k = solution.head(N);
            lambda_w = solution.tail(m_w);
            // DEBUG
            #ifdef DEBUG
                std::cout << "Sub-problem number " << iteration_counter << std::endl;
                std::cout << "p_k = [" ;
                for(int l = 0; l < p_k.size(); l++){
                    std::cout << p_k[l] << ", ";
                }
                std::cout << "], lambda_k = [";
                for(int l = 0; l < lambda_w.size(); l++){
                    std::cout << lambda_w[l] << ", ";
                }
                std::cout << "]" << std::endl;
                std::cout << "||p_k|| = " << p_k.norm() << std::endl;
            #endif
            if(p_k.norm() <= 1e-6) {    // p_k == 0
                // Check sign of inequality constraints' multipliers in the working set
                int flag = 0;
                for(int i = 0; i < m_w; ++i) {
                    if(inequality_flag[working_set[i]] == true && lambda_w[i] < 0) {
                        flag = 1;
                    }
                }
                if(!flag) {                             // Optimum found
                    // DEBUG
                    #ifdef DEBUG
                        std::cout << "Problem terminated" << std::endl;
                        #endif
                    num_iter_.push_back(iteration_counter);
                    break;
                }                  
                // Otherwise we need to find out which constraint to remove from the working set
                current_inf_w = std::numeric_limits<double>::max();
                to_erase_w = -1;
                for(int i = 0; i < m_w; ++i) {
                    if(inequality_flag[working_set[i]] == true && lambda_w[i] < current_inf_w) {
                        current_inf_w = lambda_w[i];
                        to_erase_w = i;
                    }
                }
                if(to_erase_w != -1) {working_set.erase(std::next(working_set.begin(), to_erase_w));}
                // x_k+1 = x_k
            } else {                     // p_k != 0
                // Compute alpha_k
                alpha_k = 1;
                for(int i = 0; i < inequality_flag.size(); ++i) {
                    if(std::find(working_set.begin(), working_set.end(), i) == working_set.end() && A_k.row(i) * p_k > 0) {
                        temp = - (c_k(i) + A_k.row(i) * x_k) / (A_k.row(i) * p_k + 1e-12);
                        if(temp < alpha_k) { 
                            alpha_k = temp;
                            blocking_constraint = i;
                        }
                    }
                }
                // update x
                x_k = x_k + alpha_k * p_k;
                if(alpha_k != 1) {                                      // if there was a blocking constraint
                    working_set.push_back(blocking_constraint);         // we add it to the working set
                } // else, the working set remains unchanged
            }
        }

        // Before exiting we need to restore the full vector of Lagrange multipliers, setting to
        // zero the ones associated to non-active constraints
        lambda.setZero();
        for(int i = 0; i < working_set.size(); ++i) {
            lambda[working_set[i]] = lambda_w(i);
        }

        return x_k;
    }

public:
    // Constructor (default)
    SQP() = default;
    // Constructor
    SQP(double mu, int max_iter, double feasibility_tol, double stationarity_tol, double tau, double eta) :
        mu_(mu), max_iter_(max_iter), feasibility_tol_(feasibility_tol), stationarity_tol_(stationarity_tol), tau_(tau), eta_(eta) {}

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

        // Clear Data
        this->clearData();

        // Copy x0 to a local variable since x0 is passed by const reference
        // and declare also x_new
        vector_t x_old = x0;
        // Find a feasible starting point
        x_old = compute_feasible_point(x0, constraints);
        // DEBUG
        #ifdef DEBUG
            std::cout << "Feasible Starting Point : [" << x_old.transpose() << "]" << std::endl;
        #endif
        vector_t x_new = x_old;

        // Declare the variable for the step
        vector_t p_k;

        // Declare the L1 merit function for the problem
        MeritFunction<N, ObjectiveT, ConstraintT> phi(objective, constraints);

        // Other necessary variables for the algorithm
        double gamma = 0;                                               // Used in mu update
        double alpha_k = 1;                                             // Used for step length computation
        double theta_k = 1;                                             // Used in Algorithm 18.2
        double temp1;                                                   // Temporary support double variable 1
        double temp2;                                                   // Temporary support double variable 2
        Eigen::Matrix<double, Eigen::Dynamic, 1> rhs;                   // Useful for lambda update computation
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> M;        // Useful for lambda update computation
        vector_t s_k;                                                   // Useful for Hessian approximation (BFGS update)
        vector_t y_k;                                                   // Useful for Hessian approximation (BFGS update)
        vector_t r_k;                                                   // Useful for Hessian approximation (BFGS update)
        std::vector<bool> inequality_flag(constraints.size(), false);   // Useful for active-set method
        double current_step_length;                                     // Useful for accessory termination criterion
        double short_step_counter = 0;                                  // Accessory termination criterion counter

        // Extract constraints types to pass to the SQP active-set method problem
        for(int i = 0; i < constraints.size(); ++i) {
            if(constraints[i].is_inequality_) { inequality_flag[i] = true; }
        }

        // Create a vector of Lagrange multipliers, we initialize it to zero for all constraints
        // This is a common choice in many libraries, but other initializations could be performed
        Eigen::Matrix<double, Eigen::Dynamic, 1> lambda(constraints.size());
        lambda.setZero();

        // Create the approximation of the Hessian of the Lagrangian, we initialize it to the identity matrix for the first iteration
        Eigen::Matrix<double, N, N> B_k = Eigen::Matrix<double, N, N>::Identity();

        // Evaluate f(x0), grad(f(x0)), c_i(x0) and A(x0)
        double f_k = objective(x_old);
        vector_t grad_f_k = objective.gradient(x_old);
        Eigen::Matrix<double, Eigen::Dynamic, 1> c_k(constraints.size());
        for(std::size_t i = 0; i < constraints.size(); ++i) { c_k[i] = constraints[i](x_old); }
        Eigen::Matrix<double, Eigen::Dynamic, N> A_k(constraints.size(), N);
        for(std::size_t i = 0; i < constraints.size(); ++i) { A_k.row(i) = constraints[i].gradient(x_old).transpose(); }

        // Main loop of the SQP method
        for (int k = 0; k < max_iter_; ++k) {
            // Solve the current quadratic problem
            #ifdef DEBUG
                std::cout << "Solving problem " << k +1 <<std::endl;
            #endif
            p_k = solve_problem(B_k, grad_f_k, c_k, A_k, inequality_flag, lambda, x_old);

            // Choose mu such that p_k is a descent direction for the merit function at x_k
            // We begin by computing gamma
            gamma = 0;
            for(int i = 0; i < constraints.size(); ++i) {
                if(std::abs(lambda[i]) > gamma) {gamma = std::abs(lambda[i]); }
            }

            // Now we can compute the update for mu
            if(1/mu_ < gamma + 1e-2) { mu_ = 1/(gamma + 2e-2); }

            // Compute step length
            alpha_k = 1;
            phi.set_mu(mu_);    // Update the merit function if necessary
            while(phi(x_old + alpha_k * p_k) > phi(x_old) + eta_ * alpha_k * phi.derivative(x_old, p_k)) { alpha_k = alpha_k * tau_;}

            // Compute next point
            x_new = x_old + alpha_k * p_k;
            // Also add it to the optimum data structure
            optimum_.push_back(x_new);

            // Evaluate f(x_k), grad(f(x_k)), c_i(x_k) and A(x_k)
            f_k = objective(x_new);
            values_.push_back(f_k);             // Also add it to the values data structure
            grad_f_k = objective.gradient(x_new);
            for(std::size_t i = 0; i < constraints.size(); ++i) { c_k[i] = constraints[i](x_new); }
            for(std::size_t i = 0; i < constraints.size(); ++i) { A_k.row(i) = constraints[i].gradient(x_new).transpose(); }

            // Check termination condition
            // We both check feasibility and stationarity of the problem
            // Feasibility
            double feasibility = 0;
            for(int i = 0; i < constraints.size(); ++i){
                double c_i = constraints[i](x_new);
                if(constraints[i].is_inequality_ == true) {              // Inequality constraints
                feasibility += std::max(0.0, c_i) * std::max(0.0, c_i);
            }
                else{                                                   // Equality constraints
                    feasibility += c_i * c_i;
                }
            }
            feasibility = std::sqrt(feasibility);
            // Stationarity
            auto lagrangian_grad = lagrangian_gradient(x_new, lambda, objective, constraints);
            double stationarity = lagrangian_grad.norm();
            // DEBUG
            #ifdef DEBUG
                std::cout << "Feasibility = " << feasibility << ", Stationarity = " << stationarity << std::endl;
            #endif
            // Check condition
            if(feasibility < feasibility_tol_ && stationarity < stationarity_tol_) {break;}
            // Extra condition on the step length, useful for problems in which there are
            // no active constraints at the minima (for problems with only inequality constraints)
            current_step_length = (x_new - x_old).norm();
            if(current_step_length <= 1e-10) { short_step_counter++;}
            else { short_step_counter = 0;}
            if(short_step_counter == 2) { break;}

            // Hessian approximation (BFGS update)
            s_k = x_new - x_old;

            // We perform the update only if s_k is non-zero
            if(s_k.norm() > 1e-12) {
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
                B_k = B_k - (B_k * (s_k * s_k.transpose()) * B_k)/(temp2 + 1e-12) + (r_k * r_k.transpose())/(s_k.transpose() * r_k);
            }

            // Update x_old
            x_old = x_new;

        }

        return x_new;
    }

    // Observers
    std::vector<int> num_iter() const { return num_iter_; }             // Number of iterations
    const std::vector<vector_t>& optimum() const { return optimum_; }   // Optimal solutions
    const std::vector<double>& values() const { return values_; }       // Objective function values at the optimal solutions
};

} // namespace fdapde

#endif