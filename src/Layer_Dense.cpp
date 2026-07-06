#include <unn/Layer_Dense.hpp>

#include <cassert>

namespace unn
{
Layer_Dense::Layer_Dense(Eigen::Index n_inputs, Eigen::Index n_outputs)
{
  // Intialize with random weights ( Uniform[0, 1] )
  weights = (Eigen::ArrayXXd::Random(n_outputs, n_inputs) * 0.5 + 0.5);
  biases = (Eigen::ArrayXXd::Random(n_outputs, 1) * 0.5 + 0.5);
}

Layer_Dense::Layer_Dense(const Eigen::MatrixXd &weights, const Eigen::VectorXd &biases)
    : weights(weights), biases(biases)
{
}

Eigen::MatrixXd Layer_Dense::operator()(const Eigen::MatrixXd &inputs)
{
  this->input = inputs;

  return (weights * inputs).colwise() + biases;
}

// Note shape(d_next) = shape(output)
void Layer_Dense::backward(const Eigen::MatrixXd &d_next)
{
  // This sums d_next * w across each output derivative
  this->d_input = weights.transpose() * d_next;

  // This sums d_next * x across each input
  this->d_weights = d_next * input.transpose();

  this->d_biases = d_next.rowwise().sum();
}

const Eigen::MatrixXd &Layer_Dense::get_d_weights() const { return this->d_weights; }
const Eigen::MatrixXd &Layer_Dense::get_d_biases() const { return this->d_biases; }

bool Layer_Dense::moment1_initialized() const
{
  // You could do .size() != 0
  return biases_moment1.size() == biases.size() && weights_moment1.rows() == weights.rows() &&
         weights_moment1.cols() == weights.cols();
}
void Layer_Dense::init_moment1()
{
  const auto biases_size = this->biases.size();
  const auto weights_rows = this->weights.rows();
  const auto weights_cols = this->weights.cols();

  this->weights_moment1 = Eigen::MatrixXd::Zero(weights_rows, weights_cols);
  this->biases_moment1 = Eigen::VectorXd::Zero(biases_size);
}

bool Layer_Dense::moment2_initialized() const
{
  return biases_moment2.size() == biases.size() && weights_moment2.rows() == weights.rows() &&
         weights_moment2.cols() == weights.cols();
}

void Layer_Dense::init_moment2()
{
  const auto biases_size = this->biases.size();
  const auto weights_rows = this->weights.rows();
  const auto weights_cols = this->weights.cols();

  this->weights_moment2 = Eigen::MatrixXd::Zero(weights_rows, weights_cols);
  this->biases_moment2 = Eigen::VectorXd::Zero(biases_size);
}
} // namespace unn
