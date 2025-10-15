#include "Layer.hpp"
#include "Eigen/Core"
#include <cassert>

namespace unn
{
Layer::Layer(unsigned int nin, unsigned int nout, unsigned int batches)
    : nin{nin}, nout{nout}, batches{batches}
{
        weights = Eigen::MatrixXd::Random(nout, nin);
        biases = Eigen::VectorXd::Random(nout);
}
void Layer::forward(const Eigen::MatrixXd &inputs)
{
        bool valid_in_nrows = inputs.rows() == nin;
        bool valid_in_ncols = inputs.cols() == batches;

        bool valid_out_nrows;
        bool valid_out_ncols;

        assert(((valid_in_nrows && valid_in_ncols) && "inputs has invalid size"));
        outputs = (weights * inputs).rowwise() + biases.transpose();

        valid_out_nrows = outputs.rows() == nout;
        valid_out_ncols = outputs.cols() == batches;

        assert(((valid_out_nrows && valid_out_ncols) && "outputs has invalid size"));
}
} // namespace unn
