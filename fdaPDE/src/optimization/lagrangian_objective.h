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

#ifndef __FDAPDE_Lagrangian_Objective_H__
#define __FDAPDE_Lagrangian_Objective_H__

#include <vector>
#include <type_traits>
#include <Eigen/Core>

namespace fdapde {

template<int N, typename ObjectiveT>
class LagrangianObjective {
private:
    using vector_t = std::conditional_t<N == Eigen::Dynamic, Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::Matrix<double, N, 1>>;

    ObjectiveT objective_;                          // Objective function f(x)
    std::vector<double> lambda_;                    // Lagrange multipliers
    std::vector<ObjectiveT> constraints_;           // Constraint functions c_k(x)
    std::vector<bool> ineq_constr_;                 // Vector to store information on constraints type (true = inequality, false = equality)
    double mu_;                                     // Penalty parameter

public:
    // Constructor
    LagrangianObjective(const ObjectiveT& objective, const std::vector<double>& lambda,
        const std::vector<ObjectiveT>& constraints, const std::vector<bool> ineq_constr, double mu) 
        : objective_(objective), lambda_(lambda), constraints_(constraints), ineq_constr_(ineq_constr), mu_(mu) {}
    // Call operator 
    double operator()(const vector_t& x) const {
        double value = objective_(x);
        for (std::size_t k = 0; k < constraints_.size(); ++k) {
            const double c_k = constraints_[k](x);
            if (ineq_constr_[k] == true) {  // Inequality constraint
                value += (1.0 / (2.0 * mu_)) * (std::max(0.0, lambda_[k] + c_k * mu_) * std::max(0.0, lambda_[k] + c_k * mu_) - lambda_[k] * lambda_[k]);     
            } else {                        // Equality constraint
                value += -lambda_[k] * c_k + (1.0 / (2.0 * mu_)) * c_k * c_k;
            } 
        }
        return value;
    }
    // Gradient method, created as a functor in order to comply with the optimizers interface
    // We return a callable object instead of the vector with the gradient evaluation directly
    // Optimizers will perform : (1) auto grad = lagrangian_objective.gradient(); (2) vector_t g = grad(x);
    struct GradientFunctor {
        const LagrangianObjective& obj_;

        vector_t operator()(const vector_t& x) const {
            vector_t value = obj_.objective_.gradient(x);

            for (std::size_t k = 0; k < obj_.constraints_.size(); ++k) {
                const double c_k = obj_.constraints_[k](x);
                if (obj_.ineq_constr_[k] == true ) {    // Inequality constraint
                    value += std::max(0.0, obj_.lambda_[k] + c_k * obj_.mu_) * obj_.constraints_[k].gradient(x);
                } else {                                // Equality constraint
                    value += (-obj_.lambda_[k] + c_k / obj_.mu_) * obj_.constraints_[k].gradient(x);
                }    
            }
            return value;
        }
    };

    GradientFunctor gradient() const {
        return GradientFunctor{*this};
    }
};

} // namespace fdapde

#endif