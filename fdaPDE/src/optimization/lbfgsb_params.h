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

#ifndef __FDAPDE_LBFGSB_PARAMS_H__
#define __FDAPDE_LBFGSB_PARAMS_H__

#include "header_check.h"

namespace fdapde {

    class LBFGSBParams {
        public :
        // Parameters
            int max_iter_ = 1000;                                       // maximum number of iterations
            double epsilon_ = 1e-6;                                     // tolerance
            double step_ = 1.0;                                         // step-size
            int mem_size_ = 6;                                          // memory size for Hessian approximation
            double c1_ = 1e-4;                                          // Armijo condition constant                  
            double c2_ = 0.9;                                           // Wolfe condition constant
            double backtrack_ = 0.5;                                    // kept for fallback
            double alpha_min_ = 1e-20;                                  // minimum value for alpha
            double f_tol_ = 1e-12;                                      // function value change tolerance

        // Options checker
        inline void check_params() const {
            if (max_iter_ < 0)
                throw std::invalid_argument("'max_iter' must be non-negative");
            if (epsilon_ < 0)
                throw std::invalid_argument("'epsilon' must be non-negative");
            if (step_ < 0)
                throw std::invalid_argument("'step' must be non-negative");
            if (mem_size_ <= 0)
                throw std::invalid_argument("'mem_size' must be positive");
            if (c1_ < 0)
                throw std::invalid_argument("'c1' must be non-negative");
            if (c2_ <= 0 || c2_ >= 1)
                throw std::invalid_argument("'c2' must satisfy 0 < c2 < 1");
            if (backtrack_ < 0)
                throw std::invalid_argument("'backtrack' must be non-negative");
            if (alpha_min_ < 0)
                throw std::invalid_argument("'alpha_min' must be non-negative");
            if (f_tol_ <= 0)
                throw std::invalid_argument("'f_tol' must be non-negative");
            
        }
    };

}   // namespace fdapde

#endif // __FDAPDE_LBFGSB_PARAMS_H__