#ifndef PYTHONIC_NUMPY_RANDOM_NONCENTRAL_CHISQUARE_HPP
#define PYTHONIC_NUMPY_RANDOM_NONCENTRAL_CHISQUARE_HPP

#include "pythonic/include/numpy/random/noncentral_chisquare.hpp"
#include "pythonic/include/numpy/random/generator.hpp"

#include "pythonic/types/ndarray.hpp"
#include "pythonic/types/NoneType.hpp"
#include "pythonic/types/tuple.hpp"
#include "pythonic/utils/functor.hpp"
#include "pythonic/numpy/random/chisquare.hpp"
#include "pythonic/numpy/random/normal.hpp"
#include "pythonic/numpy/random/poisson.hpp"
#include <math.h>

#include <random>
#include <algorithm>

PYTHONIC_NS_BEGIN
namespace numpy
{
  namespace random
  {

    template <class pS>
    types::ndarray<double, pS> noncentral_chisquare(double df, double nonc,
                                                    pS const &shape)
    {
      types::ndarray<double, pS> result{shape, types::none_type()};
      std::generate(result.fbegin(), result.fend(),
                    [&]() { return noncentral_chisquare(df, nonc); });
      return result;
    }

    auto noncentral_chisquare(double df, double nonc, long size)
        -> decltype(noncentral_chisquare(df, nonc,
                                         types::array<long, 1>{{size}}))
    {
      return noncentral_chisquare(df, nonc, types::array<long, 1>{{size}});
    }

    double noncentral_chisquare(double df, double nonc, types::none_type d)
    {
      double U =
          std::uniform_real_distribution<double>{0., 1.}(details::generator);
      if (npy_isnan(nonc)) {
        return NPY_NAN;
      }
      if (nonc == 0) {
        return chisquare(df);
      }
      if (1 < df) {
        const double Chi2 = chisquare(df - 1);
        const double n = normal() + sqrt(nonc);
        return Chi2 + n * n;
      } else {
        const RAND_INT_TYPE i = poisson(nonc / 2.0);
        return chisquare(bitgen_state, df + 2 * i);
      }
    }
  }
}
PYTHONIC_NS_END

#endif
