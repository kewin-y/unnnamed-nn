#include "unn/Adam.hpp"
#include <cmath>

namespace unn
{
Adam::Adam(double lr, double epsilon, double beta1, double beta2)
    : lr{lr}, epsilon{epsilon}, beta1{beta1}, beta2{beta2}, iterations{0}
{
}

void Adam::update_params(Layer_Dense &layer)
{
  if (!layer.moment2_initialized() || !layer.moment1_initialized()) {
    layer.init_moment1();
    layer.init_moment2();
  }

  layer.weights_moment1 = beta1 * layer.weights_moment1.array() + (1 - beta1) * layer.weights_moment1.array();
  layer.weights_moment2 = beta2 * layer.weights_moment2.array() +
                          (1 - beta2) * (layer.weights_moment2.array() * layer.weights_moment2.array());

  layer.biases_moment1 = beta1 * layer.biases_moment1.array() + (1 - beta1) * layer.biases_moment1.array();
  layer.biases_moment2 = beta2 * layer.biases_moment2.array() +
                         (1 - beta2) * (layer.biases_moment2.array() * layer.biases_moment2.array());

  Eigen::MatrixXd weights_m1_corrected = layer.weights_moment1.array() / (1 - std::pow(beta1, iterations));
  Eigen::MatrixXd weights_m2_corrected = layer.weights_moment2.array() / (1 - std::pow(beta2, iterations));

  Eigen::MatrixXd biases_m1_corrected = layer.biases_moment1.array() / (1 - std::pow(beta1, iterations));
  Eigen::MatrixXd biases_m2_corrected = layer.biases_moment2.array() / (1 - std::pow(beta2, iterations));

  layer.weights.array() += -lr * (weights_m1_corrected.array() *
                                             (weights_m2_corrected.array() + epsilon).cwiseSqrt().cwiseInverse());

  layer.biases.array() += -lr * (biases_m1_corrected.array() *
                                            (biases_m2_corrected.array() + epsilon).cwiseSqrt().cwiseInverse());
}

void Adam::post_update() { iterations++; }
} // namespace unn
