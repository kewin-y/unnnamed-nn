#include "unn/SGD.hpp"

namespace unn
{
SGD::SGD(double learning_rate, double decay, double momentum)
    : starting_learning_rate{learning_rate}, decay{decay}, momentum{momentum}, learning_rate{learning_rate},
      iterations{0}
{
}

void SGD::update_params(Layer_Dense &layer)
{
  if (momentum != 0.0) {
    if (!layer.moment1_initialized()) {
      layer.init_moment1();
    }

    // clang-format off
    // https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    Eigen::MatrixXd weights_velocity =
        momentum * layer.weights_moment1.array()
        - learning_rate * layer.get_d_weights().array();

    Eigen::VectorXd biases_velocity =
       momentum * layer.biases_moment1.array()
       - learning_rate * layer.get_d_biases().array();
    // clang-format on

    layer.weights.array() += weights_velocity.array();
    layer.biases.array() += biases_velocity.array();

    layer.weights_moment1 = std::move(weights_velocity);
    layer.biases_moment1 = std::move(biases_velocity);

  } else {
    layer.biases.array() += -learning_rate * layer.get_d_biases().array();
    layer.weights.array() += -learning_rate * layer.get_d_weights().array();
  }
}
void SGD::pre_update() { learning_rate = starting_learning_rate / (1 + decay * iterations); }
void SGD::post_update() { iterations++; }
} // namespace unn
