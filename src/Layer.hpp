#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

namespace unn
{
class Layer
{
public:
        Layer(unsigned int nin, unsigned int nout, unsigned int batches);

        void forward(const Eigen::MatrixXd &inputs);

        Eigen::MatrixXd inputs;
        Eigen::MatrixXd weights;
        Eigen::MatrixXd outputs;
        Eigen::VectorXd biases;

private:
        const unsigned int nin;
        const unsigned int nout;
        const unsigned int batches;
};
} // namespace unn

#endif
