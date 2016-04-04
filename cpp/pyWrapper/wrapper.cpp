//
// Created by prassanna on 4/04/16.
//


#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <stdexcept>

#include "NumPyArrayData.h"

namespace bp = boost::python;
namespace np = boost::numpy;

#define ASSERT_THROW(a,msg) if (!(a)) throw std::runtime_error(msg);

np::ndarray divideByTwo(const np::ndarray& arr)
{
    ASSERT_THROW( (arr.get_nd() == 2), "Expected two-dimensional array");
    ASSERT_THROW( (arr.get_dtype() == np::dtype::get_builtin<double>()), "Expected array of type double (np.float64)");

    np::ndarray result = np::zeros(bp::make_tuple(arr.shape(0),arr.shape(1)), np::dtype::get_builtin<double>());

    NumPyArrayData<double> arr_data(arr);
    NumPyArrayData<double> result_data(result);

    for (int i=0; i<arr.shape(0); i++) {
        for (int j=0; j<arr.shape(1); j++) {
            result_data(i,j) = arr_data(i,j) / 2.0;
        }
    }

    return result;
}
bp::tuple createGridArray(int rows, int cols)
{
    np::ndarray xgrid = np::zeros(bp::make_tuple(rows, cols), np::dtype::get_builtin<int>());
    np::ndarray ygrid = np::zeros(bp::make_tuple(rows, cols), np::dtype::get_builtin<int>());

    NumPyArrayData<int> xgrid_data(xgrid);
    NumPyArrayData<int> ygrid_data(ygrid);

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            xgrid_data(i,j) = i;
            ygrid_data(i,j) = j;
        }
    }

    return bp::make_tuple(xgrid,ygrid);
}

BOOST_PYTHON_MODULE(Rfsvm)
{
    np::initialize();

    bp::def("divideByTwo", divideByTwo);
    bp::def("createGridArray", createGridArray);
}
