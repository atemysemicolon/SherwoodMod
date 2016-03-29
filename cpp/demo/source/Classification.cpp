#include "Classification.h"

#include "FeatureResponseFunctions.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponseFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, bool root_node)
  {
    return AxisAlignedFeatureResponse::CreateRandom(random, data, dataIndices,i0,i1,root_node);
  }

  /*
  LinearFeatureResponse2d LinearFeatureFactory::CreateRandom(Random& random)
  {
    return LinearFeatureResponse2d::CreateRandom(random);
  }*/

  LinearFeatureResponse LinearFeatureFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, bool root_node)
  {
    return LinearFeatureResponse::CreateRandom(random, data, dataIndices,i0,i1,root_node);
  }

  LinearFeatureResponseSVM LinearFeatureSVMFactory::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, bool root_node)
  {
    return LinearFeatureResponseSVM::CreateRandom(random, data, dataIndices,i0,i1,root_node);
  }
} } }

