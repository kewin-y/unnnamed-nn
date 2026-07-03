#ifndef ADAM_HPP
#define ADAM_HPP

#include "unn/Layer_Dense.hpp"
#include <cstdint>

namespace unn
{
class Adam
{
private:
  const double learning_rate;
  const double epsilon;
  const double beta1;
  const double beta2;

  uint32_t iterations;

public:
  Adam(double learning_rate = 1.0f, double epsilon = 1e-8, double beta1 = 0.9f, double beta2 = 0.999f);

  void update_params(Layer_Dense &layer);
  void post_update();
};
} // namespace unn

#endif
