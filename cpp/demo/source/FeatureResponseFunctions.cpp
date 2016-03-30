#include "FeatureResponseFunctions.h"

#include <cmath>

#include <sstream>

#include "DataPointCollection.h"
#include "Random.h"

#include <eigen3/Eigen/Eigen>
#include "svm_utils.h"
#include "eigen_extensions.h"


namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  AxisAlignedFeatureResponse AxisAlignedFeatureResponse ::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node)
  {
    return AxisAlignedFeatureResponse(random.Next(0, 2));
  }

  float AxisAlignedFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int sampleIndex) const
  {
    const DataPointCollection& concreteData = (DataPointCollection&)(data);
    return concreteData.GetDataPoint((int)sampleIndex)[axis_];
  }

  std::string AxisAlignedFeatureResponse::ToString() const
  {
    std::stringstream s;
    s << "AxisAlignedFeatureResponse(" << axis_ << ")";

    return s.str();
  }

  /// <returns>A new LinearFeatureResponse2d instance.</returns>
  LinearFeatureResponse2d LinearFeatureResponse2d::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node)
  {
    double dx = 2.0 * random.NextDouble() - 1.0;
    double dy = 2.0 * random.NextDouble() - 1.0;

    double magnitude = sqrt(dx * dx + dy * dy);

    return LinearFeatureResponse2d((float)(dx / magnitude), (float)(dy / magnitude));
  }

  float LinearFeatureResponse2d::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    return dx_ * concreteData.GetDataPoint((int)index)[0] + dy_ * concreteData.GetDataPoint((int)index)[1];
  }

  std::string LinearFeatureResponse2d::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse(" << dx_ << "," << dy_ << ")";

    return s.str();
  }


  /// <returns>A new LinearFeatureResponse instance.</returns>
  LinearFeatureResponse LinearFeatureResponse::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node=false)
  {
    LinearFeatureResponse lr;
    //lr.dimensions_ = data.GetDimension();
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    lr.dimensions_ = concreteData.Dimensions();
    lr.vWeights_.resize(lr.dimensions_,-1);

    double magnitude = 0.0f;
    for (int i=0; i<lr.dimensions_; i++)
    {
      double rnd = 2.0 * random.NextDouble() - 1.0;
      magnitude += rnd*rnd;
      lr.vWeights_[i] = (float)rnd;
    }
    magnitude = sqrt(magnitude);

    for (int i=0; i<lr.dimensions_; i++)
      lr.vWeights_[i] /= (float)magnitude;

    lr.dimensions_ = concreteData.Dimensions();
    return lr;
  }

  float LinearFeatureResponse::GetResponse(const IDataPointCollection& data, unsigned int index) const
  {
    // Multiply the weights by the vector to classify and sum
    //return vec4::Dot(&vWeights_[0], ((const DataPointCollection&)(data)).GetDataPoint((int)index), dimensions_) + bias_;
    const DataPointCollection& concreteData = (const DataPointCollection&)(data);
    std::vector<float> rowData = concreteData.GetDataPointRange(index);
    float response = std::inner_product(rowData.begin(),rowData.end(), vWeights_.begin(), bias_);
    return response;

  }

  std::string LinearFeatureResponse::ToString() const
  {
    std::stringstream s;
    s << "LinearFeatureResponse(";
    s << vWeights_[0];
    for (int i=1; i<dimensions_; i++)
      s << "," << vWeights_[i];
    s << ")";

    return s.str();
  }




//MAIN STUFF
    void LinearFeatureResponseSVM::GenerateMask(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
    {

        int numBloks = random.Next(1, dims+1);

        for(int i=0;i<numBloks;i++)
        {
            int indx = random.Next(0,dims);
            vIndex.push_back(indx);
        }

    }

    void LinearFeatureResponseSVM::GenerateMaskFisher(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
    {

        bool machine_lbp_choice = (random.NextDouble()>0.5);
        bool loc_choice = random.NextDouble()>0.5;
        int numBloks = 0;
        int maxBloks = 1;

        if(machine_lbp_choice)
        {
            numBloks = random.Next(1,HYPER_MACHINE_PAIRS);
            maxBloks = HYPER_MACHINE_PAIRS;
        }
        else
        {
            numBloks = random.Next(1, HYPER_LBP_PAIRS);
            maxBloks = HYPER_LBP_PAIRS;
        }



        /*

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

            //Decides either LBP or Machine at each stage

            //Randomly choose machine(true) or lbp(false)
            bool machine_lbp_choice = (random.NextDouble()>0.5);

            //Decide whether to include location
            bool loc_choice = random.NextDouble()>0.5;

            //Choosing number of fisher pairs to take in
            int numBloks=0;
            int maxBloks =1;
            if(machine_lbp_choice)
            {
                numBloks = random.Next(1, HYPER_MACHINE_PAIRS);
                maxBloks = HYPER_MACHINE_PAIRS;
            }
            else
            {
                numBloks = random.Next(1,HYPER_LBP_PAIRS);
                maxBloks = HYPER_LBP_PAIRS;
            }


            bool lbp_choice = !(machine_lbp_choice); //Easy choosing arithmetic
            int index=0;
            //Iterate over the entire FV part
            for (int i=0; i<numBloks; i++)
            {

                //Choosing a pair-index randomly
                int indx = random.Next(0, maxBloks);//*numBloks;

                //Selecting MEAN
                //First Term = LBP offset
                vIndex[index++] = (HYPERFISHER_MACHINE_DIM*lbp_choice) + indx;

                //Selecting STD_DEV
                //First term is the LBP offset
                //Second term is the std_dev offset
                //2nd term automatically scales to LBP or Machine when required
                vIndex[index++] = (HYPERFISHER_MACHINE_DIM*lbp_choice) + (indx + (HYPER_MACHINE_PAIRS*machine_lbp_choice + HYPER_LBP_PAIRS*lbp_choice));


                //if(vIndex[index]<=0 || vIndex[index]>=FEATURE_LENGTH_SUPERPIXEL)
                //continue;
            }



            if(loc_choice)
            {
                    vIndex[index++] = HYPERFISHER_MACHINE_DIM+HYPER_LBP_DIM; //normalized x
                    vIndex[index++] = FEATURE_LENGTH_HYPERCOLUMN*BLOK_SIZE_SUPERPIXEL+FEATURE_LENGTH_LBP+1; //normalized y

            }
            nIndex = index;
            // May be commented to speed up
            for (index; index<numBloks; index++)
                vIndex[index] = -1;

                //FISHERS!

         */

    }



    LinearFeatureResponseSVM LinearFeatureResponseSVM::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node)
    {
        //HACK - Modifying this
        using namespace esvm;
        LinearFeatureResponseSVM lr;
        const DataPointCollection& concreteData = (const DataPointCollection&)(data);
        //this->dimensions_ =  concreteData.Dimensions();
        lr.dimensions_ = concreteData.Dimensions();


        GenerateMask(random, lr.vIndex_, lr.dimensions_, root_node);
        int nWeights = lr.vIndex_.size();
        //std::cout<<"[DEBUG - printing weights] : "<<nWeights<<std::endl;

        // Copy the samples for training this classifier
        int nSamples = i1-i0+1;
        //std::vector<std::vector<float> > vFeatures(nSamples, std::vector<float>(nWeights));
        Eigen::MatrixXf vFeatures(nSamples, nWeights);
        std::vector<int> vLabels(nSamples);
        int indx=0;
        for (unsigned int i=i0; i<i1; i++, indx++)
        {
            //memcpy(&vFeatures[indx][0], ((DataPointCollection&)data).GetDataPoint(dataIndices[i]), lr.dimensions_*sizeof(float));
            std::vector<float> rowData = concreteData.GetDataPointRange(dataIndices[i]);
            for(int j=0;j<nWeights;j++) {
                vFeatures(indx, j) = rowData[lr.vIndex_[j]];
                //std::cout<<"[DEBUG - printing features] : "<<vFeatures(indx,j)<<", ";
            }
            //std::cout<<std::endl;
            vLabels[indx] = (int)((DataPointCollection&)data).GetIntegerLabel(dataIndices[i]);
            //std::cout<<"[DEBUG - printing labels] : "<<vLabels[indx]<<","<<i<<std::endl;

        }

        //SVM TRAINING PART
        SVMClassifier svm;
        svm.setDisplay(true);
        svm.setC(svm_c);
        svm.train(vFeatures, vLabels);
        Eigen::MatrixXf w;
        float b;
        svm.getw(w, lr.bias_);


        //Hacky way
        for(int k=0;k<w.rows();k++)
            lr.vWeights_.push_back(w(k,0));








        return lr;
    }

    float LinearFeatureResponseSVM::GetResponse(const IDataPointCollection& data, unsigned int index) const
    {
        const DataPointCollection& concreteData = (const DataPointCollection&)(data);
        std::vector<float> rowData = concreteData.GetDataPointRange(index);
        float response = std::inner_product(rowData.begin(),rowData.end(), vWeights_.begin(), bias_);
        return response;
    }

    std::string LinearFeatureResponseSVM::ToString() const
    {
        std::stringstream s;
        s << "LinearFeatureResponse()";

        return s.str();
    }





        } } }
