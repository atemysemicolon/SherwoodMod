cmake_minimum_required(VERSION 3.2)
project(cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/linux")

set(SOURCE_FILES
    demo/source/Bitmap.h
    demo/source/Classification.cpp
    demo/source/Classification.h
    demo/source/ClassificationExample.h
    demo/source/CommandLineParser.cpp
    demo/source/CommandLineParser.h
    demo/source/CumulativeNormalDistribution.cpp
    demo/source/CumulativeNormalDistribution.h
    demo/source/DataPointCollection.cpp
    demo/source/DataPointCollection.h
    demo/source/DensityEstimation.cpp
    demo/source/DensityEstimation.h
    demo/source/DensityEstimationExample.h
    demo/source/dibCodec.cpp
    demo/source/dibCodec.h
    demo/source/FeatureResponseFunctions.cpp
    demo/source/FeatureResponseFunctions.h
    demo/source/FloydWarshall.cpp
    demo/source/FloydWarshall.h
    demo/source/Graphics.cpp
    demo/source/Graphics.h
    demo/source/main.cpp
    demo/source/Platform.cpp
    demo/source/Platform.h
    demo/source/PlotCanvas.cpp
    demo/source/PlotCanvas.h
    demo/source/Regression.cpp
    demo/source/Regression.h
    demo/source/RegressionExample.h
    demo/source/SemiSupervisedClassification.cpp
    demo/source/SemiSupervisedClassification.h
    demo/source/StatisticsAggregators.cpp
    demo/source/StatisticsAggregators.h
    demo/source/stdafx.h
    demo/source/stringUtilities.h
    lib/Forest.h
    lib/ForestTrainer.h
    lib/Interfaces.h
    lib/Node.h
    lib/ParallelForestTrainer.h
    lib/ProgressStream.h
    lib/Random.h
    lib/Sherwood.h
    lib/TrainingParameters.h
    lib/Tree.h)

    include_directories("${CMAKE_SOURCE_DIR}/lib")
    include_directories("${CMAKE_SOURCE_DIR}/demo/source")


add_executable(sw ${SOURCE_FILES})