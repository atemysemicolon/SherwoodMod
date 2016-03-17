//
// Created by prassanna on 11/03/16.
//
#include <boost/python.hpp>
#include "Interfaces.h"

namespace msrcrf = MicrosoftResearch::Cambridge::Sherwood;


struct IDataPointCollectionWrap : msrcrf::IDataPointCollection, boost::python::wrapper<msrcrf::IDataPointCollection>
{
    IDataPointCollectionWrap(){}

    IDataPointCollectionWrap(msrcrf::IDataPointCollection &rhs) :  msrcrf::IDataPointCollection(rhs)
    { }
    unsigned int Count() const
    {
        return this->get_override("Count")();
    }

};

class Dummymsrcrf{};


using namespace boost::python;

BOOST_PYTHON_MODULE(pyIDataPointCollection)
{

    scope ms = class_<Dummymsrcrf>("msrcrf");


    class_<IDataPointCollectionWrap>("IDataPointCollection")
            .def("Count", pure_virtual(&IDataPointCollectionWrap::Count))
                    ;

}