#ifndef PYTHONIC_NUMPY_RANDOM_GUMBEL_HPP
#define PYTHONIC_NUMPY_RANDOM_GUMBEL_HPP

#include "pythonic/include/numpy/random/gumbel.hpp"
#include "pythonic/include/numpy/random/generator.hpp"

#include "pythonic/types/ndarray.hpp"
#include "pythonic/types/NoneType.hpp"
#include "pythonic/types/tuple.hpp"
#include "pythonic/utils/functor.hpp"
#include <math.h>

#include <random>
#include <algorithm>

PYTHONIC_NS_BEGIN
namespace numpy
{
  namespace random
  {

    template <class pS>
    types::ndarray<double, pS> gumbel(double loc, double scale, pS const &shape)
    {
      types::ndarray<double, pS> result{shape, types::none_type()};
      std::generate(result.fbegin(), result.fend(),
                    [&]() { return gumbel(loc, scale); });
      return result;
    }

    auto gumbel(double loc, double scale, long size)
        -> decltype(gumbel(loc, scale, types::array<long, 1>{{size}}))
    {
      return gumbel(loc, scale, types::array<long, 1>{{size}});
    }

    double gumbel(double loc, double scale, types::none_type d)
    {
      double U =
          std::uniform_real_distribution<double>{0., 1.}(details::generator);
      if (U < 1.0) {
        return loc - scale * log(-log(U));
      }
      return gumbel(loc, scale);
    }
  }
}
PYTHONIC_NS_END

#endif
