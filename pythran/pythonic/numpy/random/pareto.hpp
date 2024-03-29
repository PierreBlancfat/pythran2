#ifndef PYTHONIC_NUMPY_RANDOM_PARETO_HPP
#define PYTHONIC_NUMPY_RANDOM_PARETO_HPP

#include "pythonic/include/numpy/random/generator.hpp"
#include "pythonic/include/numpy/random/pareto.hpp"

#include "pythonic/types/NoneType.hpp"
#include "pythonic/types/ndarray.hpp"
#include "pythonic/types/tuple.hpp"
#include "pythonic/utils/functor.hpp"
#include <math.h>

#include <algorithm>
#include <random>

PYTHONIC_NS_BEGIN
namespace numpy
{
  namespace random
  {

    template <class pS>
    types::ndarray<double, pS> pareto(double a, pS const &shape)
    {
      types::ndarray<double, pS> result{shape, types::none_type()};
      std::exponential_distribution<float> distribution{};
      std::generate(result.fbegin(), result.fend(), [&]() {
        return exp(distribution(details::generator) / a) - 1;
      });
      return result;
    }

    auto pareto(double a, long size)
        -> decltype(pareto(a, types::array<long, 1>{{size}}))
    {

      return pareto(a, types::array<long, 1>{{size}});
    }

    double pareto(double a, types::none_type d)
    {
      return exp(std::exponential_distribution<float>{}(details::generator) /
                 a) -
             1;
    }
  }
}
PYTHONIC_NS_END

#endif
