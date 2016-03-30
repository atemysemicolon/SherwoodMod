#include <stdio.h>

#include <string>
#include <iostream>
#include <fstream>

#include "Platform.h"

#include "Graphics.h"
#include "dibCodec.h"

#include "Sherwood.h"

#include "CumulativeNormalDistribution.h"

#include "CommandLineParser.h"
#include "DataPointCollection.h"

#include "Classification.h"
#include "DensityEstimation.h"
#include "SemiSupervisedClassification.h"
#include "Regression.h"

using namespace MicrosoftResearch::Cambridge::Sherwood;

void DisplayHelp();

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
  if(argc<2 || std::string(argv[1])=="/?" || toLower(argv[1])=="help")
  {
    DisplayHelp();
    return 0;
  }

  std::string mode = toLower(argv[1]); //  argv[0] is name of exe, argv[1] defines command line mode

  // These command line parameters are reused over several command line modes...
  StringParameter trainingDataPath("path", "Path of file containing training data.");
  StringParameter forestOutputPath("forest", "Path of file containing forest.");
  StringParameter forestPath("forest", "Path of file containing forest.");
  StringParameter testDataPath("data", "Path of file containing test data.");
  StringParameter outputPath("output", "Path of file containing output.");
  NaturalParameter T("t", "No. of trees in the forest (default = {0}).", 10);
  NaturalParameter D("d", "Maximum tree levels (default = {0}).", 10, 20);
  NaturalParameter F("f", "No. of candidate feature response functions per split node (default = {0}).", 10);
  NaturalParameter L("l", "No. of candidate thresholds per feature response function (default = {0}).", 1);
  SingleParameter a("a", "The number of 'effective' prior observations (default = {0}).", true, false, 10.0f);
  SingleParameter b("b", "The variance of the effective observations (default = {0}).", true, true, 400.0f);
  SimpleSwitchParameter verboseSwitch("Enables verbose progress indication.");
  SingleParameter plotPaddingX("padx", "Pad plot horizontally (default = {0}).", true, false, 0.1f);
  SingleParameter plotPaddingY("pady", "Pad plot vertically (default = {0}).", true, false, 0.1f);

  EnumParameter split(
    "s",
    "Specify what kind of split function to use (default = {0}).",
    "axis;linear",
    "axis-aligned split;linear split",
    "axis");

  // Behaviour depends on command line mode...
  if (mode == "clas" || mode == "class")
  {
    // Supervised classification
    CommandLineParser parser;
    parser.SetCommand("SW CLAS");

    parser.AddArgument(trainingDataPath);
    parser.AddSwitch("T", T);
    parser.AddSwitch("D", D);
    parser.AddSwitch("F", F);
    parser.AddSwitch("L", L);

    parser.AddSwitch("split", split);

    parser.AddSwitch("PADX", plotPaddingX);
    parser.AddSwitch("PADY",  plotPaddingY);
    parser.AddSwitch("VERBOSE", verboseSwitch);

    if (argc == 2)
    {
      parser.PrintHelp();
      DisplayTextFiles(CLAS_DATA_PATH);
      return 0;
    }

    if (parser.Parse(argc, argv, 2) == false)
      return 0;

    TrainingParameters trainingParameters;
    trainingParameters.MaxDecisionLevels = D.Value;
    trainingParameters.NumberOfCandidateFeatures = F.Value;
    trainingParameters.NumberOfCandidateThresholdsPerFeature = L.Value;
    trainingParameters.NumberOfTrees = T.Value;
    trainingParameters.Verbose = verboseSwitch.Used();

    PointF plotDilation(plotPaddingX.Value, plotPaddingY.Value);

    // Load training data for a 2D density estimation problem.
    std::auto_ptr<DataPointCollection> trainingData = std::auto_ptr<DataPointCollection> ( LoadTrainingData(
      trainingDataPath.Value,
      CLAS_DATA_PATH + "/" + trainingDataPath.Value,
      3,
      DataDescriptor::HasClassLabels ) );

    std::string test_filename = "../../demo/data/sclf/sample_test.txt";
    std::auto_ptr<DataPointCollection> testdata = std::auto_ptr<DataPointCollection> ( LoadTrainingData(test_filename,
            CLAS_DATA_PATH + "/" + trainingDataPath.Value,
            3,
            DataDescriptor::HasClassLabels ) );



    if (trainingData.get()==0)
      return 0; // LoadTrainingData() generates its own progress/error messages

    if (split.Value == "linear")
    {
      LinearFeatureSVMFactory linearFeatureFactory;
      //LinearFeatureFactory linearFeatureFactory;

      std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > forest = ClassificationDemo<LinearFeatureResponseSVM>::Train(
        *trainingData,
        &linearFeatureFactory,
        trainingParameters);

      forest->Serialize("out.forest");
      std::auto_ptr<Forest<LinearFeatureResponseSVM, HistogramAggregator> > trained_forest = Forest<LinearFeatureResponseSVM, HistogramAggregator>::Deserialize("out.forest");

      /*std::auto_ptr<Bitmap<PixelBgr> > result = std::auto_ptr<Bitmap<PixelBgr> >(ClassificationDemo<LinearFeatureResponseSVM>::Visualize(*forest, *trainingData, Size(300, 300), plotDilation));
      std::cout << "\nSaving output image to result.dib" << std::endl;
      result->Save("result.dib");
       */

      std::vector<HistogramAggregator> distbns;
      ClassificationDemo<LinearFeatureResponseSVM>::Test(*trained_forest.get(), *testdata.get(), distbns);

      forest.release();
      trained_forest.release();
      distbns.clear();
    }

  }
  else
  {
    std::cout << "Unrecognized command line argument, try SW HELP." << std::endl;
    return 0;
  }

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

void DisplayHelp()
{
  // Create a dummy command line parser so we can display command line
  // help in the usual format.
  EnumParameter mode(
    "mode",
    "Select mode of operation.",
    "clas;density;regression;ssclas",
    "Supervised 2D classfication;2D density estimation;1D to 1D regression;Semi-supervised 2D classification");

  StringParameter args("args...", "Other mode-specific arguments");

  CommandLineParser parser;
  parser.SetCommand("SW");
  parser.AddArgument(mode);
  parser.AddArgument(args);

  std::cout << "Sherwood decision forest library demos." << std::endl << std::endl;
  parser.PrintHelp();

  std::cout << "To get more help on a particular mode of operation, omit the arguments, e.g.\nsw density" << std::endl;
}
