#include <stdio.h>

#include <string>
#include <iostream>
#include <fstream>

#include "Platform.h"

#include "Graphics.h"
#include "dibCodec.h"

#include "Sherwood.h"

#include "CumulativeNormalDistribution.h"

#include "DataPointCollection.h"

#include "Classification.h"
#include "DensityEstimation.h"
#include "SemiSupervisedClassification.h"
#include "Regression.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;


void DisplayTextFiles(const std::string& relativePath);

std::auto_ptr<DataPointCollection> LoadTrainingData(
  const std::string& filename,
  const std::string& alternativePath,
  int dimension,
  DataDescriptor::e descriptor);

// Store (Linux-friendly) relative paths to training data
const std::string CLAS_DATA_PATH = "/data/supervised classification";
const std::string SSCLAS_DATA_PATH = "/data/semi-supervised classification";
const std::string REGRESSION_DATA_PATH = "/data/regression";
const std::string DENSITY_DATA_PATH = "/data/density estimation";







int main(int argc, char* argv[])
{




  int data_dimensions = 3;


  //HARDCODING DEFAULTS - get from boost argparse
  TrainingParameters trainingParameters;
  trainingParameters.MaxDecisionLevels = 10;
  trainingParameters.NumberOfCandidateFeatures = 10;
  trainingParameters.NumberOfCandidateThresholdsPerFeature = 10;
  trainingParameters.NumberOfTrees = 10;
  trainingParameters.Verbose = true;

  std::string dummy = "";


  std::string train_filename = "../../demo/data/sclf/sample_train.txt";
  std::auto_ptr<DataPointCollection> trainingData
          = std::auto_ptr<DataPointCollection> ( LoadTrainingData(train_filename,
                                                                  dummy,
                                                                  data_dimensions,
                                                                  DataDescriptor::HasClassLabels ) );


  std::string test_filename = "../../demo/data/sclf/sample_test.txt";
  std::auto_ptr<DataPointCollection> testdata
          = std::auto_ptr<DataPointCollection> ( LoadTrainingData(test_filename,
                                                                  dummy,
                                                                  data_dimensions,
                                                                  DataDescriptor::HasClassLabels ) );



  if (trainingData.get()==0)
    return 0; // LoadTrainingData() generates its own progress/error messages


  LinearFeatureSVMFactory linearFeatureFactory;

  std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest
          = ClassificationDemo<LinearFeatureResponseSVM>::Train(*trainingData,
                                                                &linearFeatureFactory,
                                                                trainingParameters);

  forest->Serialize("out.forest");
  std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > trained_forest
          = Forest<LinearFeatureResponseSVM, HistogramAggregator>::Deserialize("out.forest");


  std::vector<HistogramAggregator> distbns;
  ClassificationDemo<LinearFeatureResponseSVM>::Test(*trained_forest.get(),
                                                     *testdata.get(),
                                                     distbns);

  forest.release();
  trained_forest.release();
  distbns.clear();




  return 0;
}

std::auto_ptr<DataPointCollection> LoadTrainingData(
  const std::string& filename,
  const std::string& alternativePath,
  int dimension,
  DataDescriptor::e descriptor)
{
  std::ifstream r;

  r.open(filename.c_str());

  if(r.fail())
  {
    std::string path;

    try
    {
      path = GetExecutablePath();
    }
    catch(std::runtime_error& e)
    {
      std::cout<< "Failed to determine executable path. " << e.what();
      return std::auto_ptr<DataPointCollection>(0);
    }

    path = path + alternativePath;

    r.open(path.c_str());

    if(r.fail())
    {
      std::cout << "Failed to open either \"" << filename << "\" or \"" << path.c_str() << "\"." << std::endl;
      return std::auto_ptr<DataPointCollection>(0);
    }
  }

  std::auto_ptr<DataPointCollection> trainingData;
  try
  {
    trainingData = DataPointCollection::Load (
      r,
      dimension,
      descriptor );
  }
  catch (std::runtime_error& e)
  {
    std::cout << "Failed to read training data. " << e.what() << std::endl;
    return std::auto_ptr<DataPointCollection>(0);
  }

  if (trainingData->Count() < 1)
  {
    std::cout << "Insufficient training data." << std::endl;
    return std::auto_ptr<DataPointCollection>(0);
  }

  return trainingData;
}

void DisplayTextFiles(const std::string& relativePath)
{
  std::string path;

  try
  {
    path = GetExecutablePath();
  }
  catch(std::runtime_error& e)
  {
    std::cout<< "Failed to find demo data files. " << e.what();
    return;
  }

  path = path + relativePath;

  std::vector<std::string> filenames;

  try
  {
    GetDirectoryListing(path, filenames, ".txt");
  }
  catch(std::runtime_error& e)
  {
    std::cout<< "Failed to list demo data files. " << e.what();
    return;
  }

  if (filenames.size() > 0)
  {
    std::cout << "The following demo data files can be specified as if they were on your current path:-" << std::endl;

    for(std::vector<std::string>::size_type i=0; i<filenames.size(); i++)
      std::cout << "  " << filenames[i].c_str() << std::endl;
  }
}

