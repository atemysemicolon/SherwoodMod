#pragma once

// This file defines some IFeatureResponse implementations used by the example code in
// Classification.h, DensityEstimation.h, etc. Note we represent IFeatureResponse
// instances using simple structs so that all tree data can be stored
// contiguously in a linear array.

#include <string>

#include "Sherwood.h"

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  class Random;

    #define WITH_FISHER true
    #define FISHER_CLUSTERS 16
    #define HYPER_MACHINE_DIM 64
    #define HYPER_LBP_DIM 59
    #define HYPER_LOCATION_DIM 2
    #define HYPER_MACHINE_PAIRS (HYPER_MACHINE_DIM*FISHER_CLUSTERS)
    #define HYPER_LBP_PAIRS (HYPER_LBP_DIM*FISHER_CLUSTERS)
    #define HYPERFISHER_MACHINE_DIM (2*HYPER_MACHINE_PAIRS)
    #define HYPERFISHER_LBP_DIM (2*HYPER_LBP_PAIRS)
    #define HYPER_FISHER_DIM (HYPER_MACHINE_DIM+HYPER_LBP_DIM+HYPER_LOCATION_DIM)

  /// <summary>
  /// A feature that orders data points using one of their coordinates,
  /// i.e. by projecting them onto a coordinate axis.
  /// </summary>
  class AxisAlignedFeatureResponse
  {
    int axis_;

  public:
    AxisAlignedFeatureResponse()
    {
      axis_ = -1;
    }

    /// <summary>
    /// Create an AxisAlignedFeatureResponse instance for the specified axis.
    /// </summary>
    /// <param name="axis">The zero-based index of the axis.</param>
    AxisAlignedFeatureResponse(int axis)
    {
      axis_ = axis;
    }

    /// <summary>
    /// Create an AxisAlignedFeatureResponse instance with a random choice of axis.
    /// </summary>
    /// <param name="randomNumberGenerator">A random number generator.</param>
    /// <returns>A new AxisAlignedFeatureResponse instance.</returns>
    static AxisAlignedFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node);

    int Axis() const
    {
      return axis_;
    }

    // IFeatureResponse implementation
    float GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const;

    std::string ToString() const;
  };

  /// <summary>
  /// A feature that orders data points using a linear combination of their
  /// coordinates, i.e. by projecting them onto a given direction vector.
  /// </summary>
  class LinearFeatureResponse2d
  {
    float dx_, dy_;

  public:
    LinearFeatureResponse2d()
    {
      dx_ = 0.0;
      dy_ = 0.0;
    }

    /// <summary>
    /// Create a LinearFeatureResponse2d instance for the specified direction vector.
    /// </summary>
    /// <param name="dx">The first element of the direction vector.</param>
    /// <param name="dx">The second element of the direction vector.</param> 
    LinearFeatureResponse2d(float dx, float dy)
    {
      dx_ = dx; dy_ = dy;
    }

    /// <summary>
    /// Create a LinearFeatureResponse2d instance with a random direction vector.
    /// </summary>
    /// <param name="randomNumberGenerator">A random number generator.</param>
    /// <returns>A new LinearFeatureResponse2d instance.</returns>
    static LinearFeatureResponse2d CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node);

    // IFeatureResponse implementation
    float GetResponse(const IDataPointCollection& data, unsigned int index) const;

    std::string ToString()  const;
  };

  class LinearFeatureResponse
  {
  protected:
      std::vector<float> vWeights_;
      int		dimensions_;
      float	bias_;
      int		nIndex_;


  public:
      //std::string ;
      LinearFeatureResponse():
              dimensions_(-1),
              bias_(0.0f)
      {
        //m_param_filename = "/home/prassanna/Development/Code3/Parameters/parametersTaskManager2.ini";
      }

      /// <summary>
      /// Create a LinearFeatureResponse instance for the specified direction vector.
      /// </summary>
      /// <param name="dx">The first element of the direction vector.</param>
      /// <param name="dx">The second element of the direction vector.</param>
      LinearFeatureResponse(float* pWeights, const int dimensions)
      {

        vWeights_ = std::vector<float>(pWeights, pWeights+sizeof pWeights/sizeof pWeights[0]);
        dimensions_ = dimensions;
      }

      /// <summary>
      /// Create a LinearFeatureResponse2d instance with a random direction vector.
      /// </summary>
      /// <param name="randomNumberGenerator">A random number generator.</param>
      /// <returns>A new LinearFeatureResponse2d instance.</returns>
      static LinearFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node);

      // IFeatureResponse implementation
      float GetResponse(const IDataPointCollection& data, unsigned int index) const;

      std::string ToString()  const;
  };


    class LinearFeatureResponseSVM
    {
    protected:
        std::vector<int> vIndex_;
        std::vector<float> vWeights_;
        int		dimensions_;
        float	bias_;
        int		nIndex_;

    public:
        LinearFeatureResponseSVM():
                dimensions_(-1),
                bias_(0.0f)
        {
            //m_param_filename = "/home/prassanna/Development/Code3/Parameters/parametersTaskManager2.ini";
        }

        /// <summary>
        /// Create a LinearFeatureResponse instance for the specified direction vector.
        /// </summary>
        /// <param name="dx">The first element of the direction vector.</param>
        /// <param name="dx">The second element of the direction vector.</param>
        LinearFeatureResponseSVM(float* pWeights, const int dimensions)
        {

            vWeights_ = std::vector<float>(pWeights, pWeights+sizeof pWeights/sizeof pWeights[0]);
            dimensions_ = dimensions;
        }
        static LinearFeatureResponseSVM CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node=false);
        static void GenerateMask(Random& random, std::vector<int>& vIndex, int dims , bool root_node);

        float GetResponse(const IDataPointCollection &data, unsigned int index) const;
        std::string ToString()  const;
    };


} } }
