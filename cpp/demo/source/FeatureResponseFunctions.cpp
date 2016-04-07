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

    void LinearFeatureResponseSVM::GenerateMaskLBP(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
    {
        int maxBloks = 12;
        int feature_length = 59;

        int numBloks = random.Next(1,maxBloks+1);


        for (int i = 0; i< numBloks;i ++ )
        {
            int idx = random.Next(0,maxBloks);
            int feature_start = idx * feature_length;
            for(int j = feature_start; j<feature_start+feature_length;j++)
            {
                vIndex.push_back(j);
            }
        }

    }

    void LinearFeatureResponseSVM::GenerateMaskFisher(Random &random, std::vector<int>& vIndex, int dims, bool root_node)
    {

        bool machine_choice = (random.NextDouble()>0.5);
        bool loc_choice = random.NextDouble()>0.5;
        int numBloks = 0;
        int maxBloks = 1;

        if(machine_choice)
        {
            numBloks = random.Next(1,HYPER_MACHINE_PAIRS);
            maxBloks = HYPER_MACHINE_PAIRS;
        }
        else
        {
            numBloks = random.Next(1, HYPER_LBP_PAIRS);
            maxBloks = HYPER_LBP_PAIRS;
        }

        bool lbp_choice = !machine_choice;

        for(int i=0;i<numBloks;i++)
        {
            int indx = random.Next(0,maxBloks);

            //Selecting mean
            vIndex.push_back( (HYPERFISHER_MACHINE_DIM*lbp_choice) + indx );

            //Selecting std_dev
            vIndex.push_back( (HYPERFISHER_MACHINE_DIM*lbp_choice) + (indx + (HYPER_MACHINE_PAIRS*machine_choice + HYPER_LBP_PAIRS*lbp_choice)) );
        }

        if(loc_choice)
        {
            vIndex.push_back(HYPERFISHER_MACHINE_DIM+HYPER_LBP_DIM);
            vIndex.push_back(HYPERFISHER_MACHINE_DIM+HYPER_LBP_DIM+1);
        }

    }


    void LinearFeatureResponseSVM::GenerateMaskHypercolumn(Random &random, std::vector<int> &vIndex, int dims,
                                                           bool root_node)
    {

        //Discarding LBP and position to check
        int numBloks = random.Next (5, 15);
        for(int i=0;i<numBloks;i++)
        {
            int idx = random.Next (0,NN_DIM);
            vIndex.push_back (idx);
            vIndex.push_back (idx+NN_DIM);
        }

    }


            LinearFeatureResponseSVM LinearFeatureResponseSVM::CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node)
    {
        //HACK - Modifying this
        using namespace esvm;
        LinearFeatureResponseSVM lr;
        const DataPointCollection& concreteData = (const DataPointCollection&)(data);
        //this->dimensions_ =  concreteData.Dimensions();
        lr.dimensions_ = concreteData.Dimensions();


        GenerateMaskHypercolumn (random, lr.vIndex_, lr.dimensions_, root_node); //CHANGE THIS DEPENDING ON OPERATION
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


        }

        //SVM TRAINING PART
        //std::cout<<"[DEBUG - printing sizes - Weights | Features] : "<<nWeights<<"|"<<vFeatures.rows()<<","<<vFeatures.cols()<<std::endl;
        SVMClassifier svm;
        svm.setDisplay(false);
        svm.setC(svm_c);
        svm.train(vFeatures, vLabels);
        Eigen::MatrixXf w;
        float b;
        svm.getw(w, lr.bias_);
        lr.bias_=-1*lr.bias_;

        lr.vWeights_.resize(lr.dimensions_, 0);
        //Hacky way
        for(int k=0;k<w.rows();k++)
            lr.vWeights_[lr.vIndex_[k]] = w(k,0);
            //lr.vWeights_.push_back(w(k,0));

        //std::cout<<" [Debug - at feature response weight size] "<<lr.vWeights_.size ()<<std::endl;

        /*std::cout<<"Features -> (rows, columns)"<<vFeatures.rows()<<" "<<vFeatures.cols()<<std::endl;
        std::cout<<"Dims -> Weights(eigen) vs weights(lr)"<<w.rows()<<"\t"<<lr.vWeights_.size()<<std::endl;
       for(int i=0;i<10;i++)
            std::cout<<lr.vWeights_[i]<<" ";

        std::cout<<std::endl;*/








        return lr;
    }

    float LinearFeatureResponseSVM::GetResponse(const IDataPointCollection& data, unsigned int index) const
    {

        const DataPointCollection& concreteData = (const DataPointCollection&)(data);
        std::vector<float> rowData = concreteData.GetDataPointRange(index);


        /*//MANUAL WAY
          //std::vector<float> vFeatures;
          //std::cout<<"[Debug At getresponse weight size]"<<vWeights_.size ()<<std::endl;
          float response = bias_;
        for(int j=0;j<vWeights_.size();j++) {
            response+=rowData[vIndex_[j]]*vWeights_[j];
            //std::cout<<"[DEBUG FEATURERESPONSE - printing online response | step] : "<<response<<"   "<<j<<std::endl;
        }*/

        //aUTOMATIC WAY
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
