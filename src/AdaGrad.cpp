#include "unn/AdaGrad.hpp"

namespace unn
{
AdaGrad::AdaGrad(double lr, double epsilon)
    : lr{lr}, epsilon{epsilon}, iterations{0}
{
}

void AdaGrad::update_params(Layer_Dense &layer)
{
  if (!layer.moment2_initialized()) {
    layer.init_moment2();
  }

  layer.weights_moment2.array() += (layer.get_d_weights().array() * layer.get_d_weights().array());
  layer.biases_moment2.array() += (layer.get_d_biases().array() * layer.get_d_biases().array());

  Eigen::MatrixXd wcof = -lr * (layer.weights_moment2.array() + epsilon).cwiseSqrt().cwiseInverse();

  Eigen::MatrixXd bcof = -lr * (layer.biases_moment2.array() + epsilon).cwiseSqrt().cwiseInverse();

  layer.weights.array() += wcof.array() * layer.get_d_weights().array();
  layer.biases.array() += bcof.array() * layer.get_d_biases().array();
}

void AdaGrad::post_update() { iterations++; }
} // namespace unn
