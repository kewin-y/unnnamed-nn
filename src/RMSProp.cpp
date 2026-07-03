#include "unn/RMSProp.hpp"
namespace unn
{
RMSProp::RMSProp(double learning_rate, double epsilon, double rho)
    : learning_rate{learning_rate}, epsilon{epsilon}, rho{rho}, iterations{0}
{
}

void RMSProp::update_params(Layer_Dense &layer)
{
  if (!layer.moment2_initialized()) {
    layer.init_moment2();
  }

  layer.weights_moment2.array() = rho * (layer.weights_moment2.array()) +
                                  (1 - rho) * (layer.get_d_weights().array() * layer.get_d_weights().array());
  layer.biases_moment2.array() = rho * (layer.biases_moment2.array()) +
                                 (1 - rho) * (layer.get_d_biases().array() * layer.get_d_biases().array());

  Eigen::MatrixXd wcof = -learning_rate * (layer.weights_moment2.array() + epsilon).cwiseSqrt().cwiseInverse();

  Eigen::MatrixXd bcof = -learning_rate * (layer.biases_moment2.array() + epsilon).cwiseSqrt().cwiseInverse();

  layer.weights.array() += wcof.array() * layer.get_d_weights().array();
  layer.biases.array() += bcof.array() * layer.get_d_biases().array();
}

void RMSProp::post_update() { iterations++; }
} // namespace unn
