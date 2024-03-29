#ifndef PYTHONIC_NUMPY_RANDOM_LOGSERIES_HPP
#define PYTHONIC_NUMPY_RANDOM_LOGSERIES_HPP

#include "pythonic/include/numpy/random/logseries.hpp"
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
    types::ndarray<double, pS> logseries(double p, pS const &shape)
    {
      types::ndarray<double, pS> result{shape, types::none_type()};
      std::generate(result.fbegin(), result.fend(),
                    [&]() { return logseries(p); });
      return result;
    }

    auto logseries(double p, long size)
        -> decltype(logseries(p, types::array<long, 1>{{size}}))
    {
      return logseries(p, types::array<long, 1>{{size}});
    }

    double logseries(double p, types::none_type d)
    {
      double q, r, U, V;
      double result;

      r = log(1.0 - p);

      while (1) {
        V = std::uniform_real_distribution<double>{0., 1.}(details::generator);
        if (V >= p) {
          return 1;
        }
        U = std::uniform_real_distribution<double>{0., 1.}(details::generator);
        q = 1.0 - exp(r * U);
        if (V <= q * q) {
          result = (double)floor(1 + log(V) / log(q));
          if ((result < 1) || (V == 0.0)) {
            continue;
          } else {
            return result;
          }
        }
        if (V >= q) {
          return 1;
        }
        return 2;
      }
    }
  }
}
PYTHONIC_NS_END

#endif
