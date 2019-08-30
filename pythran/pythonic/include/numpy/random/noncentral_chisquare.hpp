#ifndef PYTHONIC_INCLUDE_NUMPY_RANDOM_NONCENTRAL_CHISQUARE_HPP
#define PYTHONIC_INCLUDE_NUMPY_RANDOM_NONCENTRAL_CHISQUARE_HPP

#include "pythonic/include/utils/functor.hpp"
#include "pythonic/include/types/ndarray.hpp"
#include "pythonic/include/types/NoneType.hpp"
#include "pythonic/include/types/tuple.hpp"

PYTHONIC_NS_BEGIN
namespace numpy
{
  namespace random
  {
    template <class pS>
    types::ndarray<double, pS> noncentral_chisquare(double df, double nonc,
                                                    pS const &shape);

    auto noncentral_chisquare(double df, double nonc, long size)
        -> decltype(noncentral_chisquare(df, nonc,
                                         types::array<long, 1>{{size}}));

    double noncentral_chisquare(double df = 0.0, double nonc = 1.0,
                                types::none_type size = {});

    DEFINE_FUNCTOR(pythonic::numpy::random, noncentral_chisquare);
  }
}
PYTHONIC_NS_END

#endif
