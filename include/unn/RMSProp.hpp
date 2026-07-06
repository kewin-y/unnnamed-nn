#ifndef RMSPROP_HPP
#define RMSPROP_HPP

#include "unn/Layer_Dense.hpp"
#include <cstdint>

namespace unn
{
class RMSProp
{
private:
  const double lr;
  const double epsilon;
  const double rho;

  uint32_t iterations;

public:
  RMSProp(double lr = 1.0f, double epsilon = 1e-8, double rho = 0.9f);

  void update_params(Layer_Dense &layer);
  void post_update();
};
} // namespace unn

#endif
