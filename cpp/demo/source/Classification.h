#pragma once

// This file defines types used to illustrate the use of the decision forest
// library in simple multi-class classification task (2D data points).

#include <stdexcept>
#include <algorithm>

#include "Graphics.h"

#include "Sherwood.h"

#include "StatisticsAggregators.h"
#include "FeatureResponseFunctions.h"
#include "DataPointCollection.h"
#include "Classification.h"
#include "PlotCanvas.h"
#include "ParallelForestTrainer.h"
#include <fstream>
#include <iostream>
#include <iterator>

namespace MicrosoftResearch { namespace Cambridge { namespace Sherwood
{
  template<class F>
  class IFeatureResponseFactory
  {
  public:
    virtual F CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node)=0;
  };

/*  class LinearFeatureFactory: public IFeatureResponseFactory<LinearFeatureResponse2d>
  {
  public:
    LinearFeatureResponse2d CreateRandom(Random& random);
  };
  */

  class LinearFeatureFactory: public IFeatureResponseFactory<LinearFeatureResponse>
  {
  public:
      LinearFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node);
  };


  class AxisAlignedFeatureResponseFactory : public IFeatureResponseFactory<AxisAlignedFeatureResponse>
  {
  public:
    AxisAlignedFeatureResponse CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node);
  };

  class LinearFeatureSVMFactory: public IFeatureResponseFactory<LinearFeatureResponseSVM>
  {
  public:
      LinearFeatureResponseSVM CreateRandom(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1, float svm_c, bool root_node);
  };

  template<class F>
  class ClassificationTrainingContext : public ITrainingContext<F,HistogramAggregator> // where F:IFeatureResponse
  {
  private:
    int nClasses_;
    float minIG_;
    unsigned int minSamples_;
      IGType igType;

    IFeatureResponseFactory<F>* featureFactory_;

  public:
    ClassificationTrainingContext(int nClasses, IFeatureResponseFactory<F>* featureFactory, IGType information_gain_type)
    {
      nClasses_ = nClasses;
      featureFactory_ = featureFactory;
      igType = information_gain_type;
    }

  private:
    // Implementation of ITrainingContext
    F GetRandomFeature(Random& random, const IDataPointCollection& data, unsigned int* dataIndices, const unsigned int i0, const unsigned int i1,float svm_c, bool root_node)
    {
      return featureFactory_->CreateRandom(random, data, dataIndices,i0,i1,svm_c,root_node);
    }

    HistogramAggregator GetStatisticsAggregator()
    {
      return HistogramAggregator(nClasses_);
    }

    double ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
    {
      switch(igType)
      {
        case ig_shannon:
          return ComputeInformationGainShannon(allStatistics, leftStatistics, rightStatistics);
              break;
        case ig_gini:
          return ComputeInformationGainGINI(allStatistics, leftStatistics, rightStatistics);
              break;
        default:
          printf("ERROR: Unknown IG type\n");
              exit(0);
      };
    }

      double ComputeInformationGainShannon(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
        if (nTotalSamples <= 1)
          return 0.0;

        double entropyBefore = allStatistics.Entropy();
        double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.Entropy() + rightStatistics.SampleCount() * rightStatistics.Entropy()) / nTotalSamples;
        return entropyBefore - entropyAfter;
      }

      double ComputeInformationGainGINI(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
        if (nTotalSamples <= 1)
          return 0.0;

        double entropyBefore = allStatistics.EntropyGINI();
        double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.EntropyGINI() + rightStatistics.SampleCount() * rightStatistics.EntropyGINI()) / nTotalSamples;
        return entropyBefore - entropyAfter;
      }

      double ComputeInformationGainReweighted(const HistogramAggregator& global, const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics)
      {
        // TODO: Bad!!!!
        unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
        if (nTotalSamples <= 1)
          return 0.0;

        double entropyBefore = allStatistics.Entropy(global.bins_, global.SampleCount());
        double entropyAfter = (leftStatistics.SampleCount() * leftStatistics.Entropy(global.bins_, global.SampleCount()) + rightStatistics.SampleCount() * rightStatistics.Entropy(global.bins_, global.SampleCount())) / nTotalSamples;
        return entropyBefore - entropyAfter;
      }

    bool ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain)
    {
      return gain < 0.01;
    }
  };

  template<class F>
  class ClassificationDemo
  {
      static const PixelBgr UnlabelledDataPointColor;

  public:
      static std::auto_ptr<Forest<F, HistogramAggregator> > Train (
              const DataPointCollection& trainingData,
              IFeatureResponseFactory<F>* featureFactory,
              const TrainingParameters& trainingParameters ) // where F : IFeatureResponse
      {
        if (trainingData.HasLabels() == false)
          throw std::runtime_error("Training data points must be labelled.");
        if (trainingData.HasTargetValues() == true)
          throw std::runtime_error("Training data points should not have target values.");

        std::cout << "Running training..." << std::endl;

        Random random;


        ClassificationTrainingContext<F> classificationContext(trainingData.CountClasses(), featureFactory, trainingParameters.igType);
        std::auto_ptr<Forest<F, HistogramAggregator> > forest;
        forest = ForestTrainer<F, HistogramAggregator>::TrainForest (random, trainingParameters,
                                                                             classificationContext, trainingData );

        //forest = ForestTrainer<F, HistogramAggregator>::TrainForest (random, trainingParameters,
          //                                                           classificationContext, trainingData );



              //std::auto_ptr<Forest<F,HistogramAggregator> >forest2 = ParallelForestTrainer<F,HistogramAggregator>::TrainForest(random, TrainingParameters, classificationContext, trainingData);

        return forest;
      }



      /// <summary>
      /// Apply a trained forest to some test data.
      /// </summary>
      /// <typeparam name="F">Type of split function</typeparam>
      /// <param name="forest">Trained forest</param>
      /// <param name="testData">Test data</param>
      /// <returns>An array of class distributions, one per test data point</returns>
      static void Test(const Forest<F, HistogramAggregator>& forest, const DataPointCollection& testData, std::vector<HistogramAggregator>& distributions) // where F : IFeatureResponse
      {
        //To save output
        //std::ofstream FILE(filename_predict);
        //std::ostream_iterator<float> output_iterator(FILE,"\t");


        int correct = 0 ;
        int nClasses = forest.GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();

        std::vector<std::vector<int> > leafIndicesPerTree;
        forest.Apply(testData, leafIndicesPerTree);

        std::vector<HistogramAggregator> result(testData.Count());

        for (int i = 0; i < testData.Count(); i++)
        {
          // Aggregate statistics for this sample over all leaf nodes reached
          result[i] = HistogramAggregator(nClasses);
          for (int t = 0; t < forest.TreeCount(); t++)
          {
            int leafIndex = leafIndicesPerTree[t][i];
            result[i].Aggregate(forest.GetTree(t).GetNode(leafIndex).TrainingDataStatistics);

          }
          int GT = testData.GetIntegerLabel(i);
          int pred_class = result[i].FindTallestBinIndex();
          float prob = result[i].GetProbability(pred_class);
          if(GT == pred_class)
            correct++;

          std::cout<<"[DEBUG : GT, Class, probablility]  - "<<GT<<" "<<pred_class<<" "<<prob<<"\t"<<bool(GT==pred_class)<<std::endl;
        }

        distributions  = result;
        std::cout<<"[DEBUG : score]  - "<<correct<<" / "<<testData.Count()<<std::endl;

        //return result;
      }
  };

  template<class F>
  const PixelBgr ClassificationDemo<F>::UnlabelledDataPointColor = PixelBgr::FromArgb(192, 192, 192);

/*  template<class F>
  class ClassificationDemo
  {
    static const PixelBgr UnlabelledDataPointColor;

  public:
    static std::auto_ptr<Forest<F, HistogramAggregator> > Train (
      const DataPointCollection& trainingData,
      IFeatureResponseFactory<F>* featureFactory,
      const TrainingParameters& TrainingParameters ) // where F : IFeatureResponse
    {
      if (trainingData.Dimensions() != 2)
        throw std::runtime_error("Training data points must be 2D.");
      if (trainingData.HasLabels() == false)
        throw std::runtime_error("Training data points must be labelled.");
      if (trainingData.HasTargetValues() == true)
        throw std::runtime_error("Training data points should not have target values.");

      std::cout << "Running training..." << std::endl;

      Random random;


      ClassificationTrainingContext<F> classificationContext(trainingData.CountClasses(), featureFactory);

      std::auto_ptr<Forest<F, HistogramAggregator> > forest
        = ForestTrainer<F, HistogramAggregator>::TrainForest (
        random, TrainingParameters, classificationContext, trainingData );



      //      std::auto_ptr<Forest<F,HistogramAggregator> >forest = ParallelForestTrainer<F,HistogramAggregator>::TrainForest(random, TrainingParameters, classificationContext, trainingData);

      return forest;
    }

    static std::auto_ptr<Bitmap<PixelBgr> > Visualize(
      Forest<F, HistogramAggregator>& forest,
      DataPointCollection& trainingData,
      Size PlotSize,
      PointF PlotDilation) // where F: IFeatureResponse
    {
      // Size PlotSize = new Size(300, 300), PointF PlotDilation = new PointF(0.1f, 0.1f)
      // Generate some test samples in a grid pattern (a useful basis for creating visualization images)
      PlotCanvas plotCanvas(trainingData.GetRange(0), trainingData.GetRange(1), PlotSize, PlotDilation);

      std::auto_ptr<DataPointCollection> testData = std::auto_ptr<DataPointCollection>(
        DataPointCollection::Generate2dGrid(plotCanvas.plotRangeX, PlotSize.Width, plotCanvas.plotRangeY, PlotSize.Height) );

      std::cout << "\nApplying the forest to test data..." << std::endl;

      std::vector<std::vector<int> > leafNodeIndices;
      forest.Apply(*testData, leafNodeIndices);

      // Same colours as those used in the book
      assert(trainingData.CountClasses()<=4);
      PixelBgr colors[4];
      colors[0] = PixelBgr::FromArgb(183, 170, 8);
      colors[1] = PixelBgr::FromArgb(194, 32, 14);
      colors[2] = PixelBgr::FromArgb(4, 154, 10);
      colors[3] = PixelBgr::FromArgb(13, 26, 188);

      PixelBgr grey = PixelBgr::FromArgb(127, 127, 127);

      // Create a visualization image
      std::auto_ptr<Bitmap<PixelBgr> > result = std::auto_ptr<Bitmap<PixelBgr> >(
        new Bitmap<PixelBgr>(PlotSize.Width, PlotSize.Height) );

      // For each pixel...
      int index = 0;
      for (int j = 0; j < PlotSize.Height; j++)
      {
        for (int i = 0; i < PlotSize.Width; i++)
        {
          // Aggregate statistics for this sample over all leaf nodes reached
          HistogramAggregator h(trainingData.CountClasses());
          for (int t = 0; t < forest.TreeCount(); t++)
          {
            int leafIndex = leafNodeIndices[t][index];
            h.Aggregate(forest.GetTree((t)).GetNode(leafIndex).TrainingDataStatistics);
          }

          // Let's muddy the colors with grey where the entropy is high.
          float mudiness = 0.5f*(float)(h.Entropy());

          float R = 0.0f, G = 0.0f, B = 0.0f;

          for (int b = 0; b < trainingData.CountClasses(); b++)
          {
            float p = (1.0f-mudiness)*h.GetProbability(b); // NB probabilities sum to 1.0 over the classes

            R += colors[b].R * p;
            G += colors[b].G * p;
            B += colors[b].B * p;
          }

          R += grey.R * mudiness;
          G += grey.G * mudiness;
          B += grey.B * mudiness;

          PixelBgr c = PixelBgr::FromArgb((unsigned char)(R), (unsigned char)(G), (unsigned char)(B));

          result->SetPixel(i, j, c); // painfully slow but safe

          index++;
        }
      }

      Graphics<PixelBgr> g(result->GetBuffer(), result->GetWidth(), result->GetHeight(), result->GetStride());

      for (unsigned int s = 0; s < trainingData.Count(); s++)
      {
        PointF x(
          (trainingData.GetDataPoint(s)[0] - plotCanvas.plotRangeX.first) / plotCanvas.stepX,
          (trainingData.GetDataPoint(s)[1] - plotCanvas.plotRangeY.first) / plotCanvas.stepY);

        RectangleF rectangle(x.X - 3.0f, x.Y - 3.0f, 6.0f, 6.0f);
        g.FillRectangle(colors[trainingData.GetIntegerLabel(s)], rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height);
        g.DrawRectangle(PixelBgr::FromArgb(0,0,0), rectangle.X, rectangle.Y, rectangle.Width, rectangle.Height);
      }

      return result;
    }

    /// <summary>
    /// Apply a trained forest to some test data.
    /// </summary>
    /// <typeparam name="F">Type of split function</typeparam>
    /// <param name="forest">Trained forest</param>
    /// <param name="testData">Test data</param>
    /// <returns>An array of class distributions, one per test data point</returns>
    static void Test(const Forest<F, HistogramAggregator>& forest, const DataPointCollection& testData, std::vector<HistogramAggregator>& distributions) // where F : IFeatureResponse
    {
      int nClasses = forest.GetTree(0).GetNode(0).TrainingDataStatistics.BinCount();

      std::vector<std::vector<int> > leafIndicesPerTree;
      forest.Apply(testData, leafIndicesPerTree);

      std::vector<HistogramAggregator> result(testData.Count());

      for (int i = 0; i < testData.Count(); i++)
      {
        // Aggregate statistics for this sample over all leaf nodes reached
        result[i] = HistogramAggregator(nClasses);
        for (int t = 0; t < forest.TreeCount(); t++)
        {
          int leafIndex = leafIndicesPerTree[t][i];
          result[i].Aggregate(forest.GetTree(t).GetNode(leafIndex).TrainingDataStatistics);
        }

        std::cout<<"[DEBUG : Class, probablility]  - "<<result[i].FindTallestBinIndex()<<" "<<result[i].GetProbability(result[i].FindTallestBinIndex())<<std::endl;
      }

      distributions  = result;

      //return result;
    }
  };

  template<class F>
  const PixelBgr ClassificationDemo<F>::UnlabelledDataPointColor = PixelBgr::FromArgb(192, 192, 192);
  */
} } }
