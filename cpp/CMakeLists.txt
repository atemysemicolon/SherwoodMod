cmake_minimum_required(VERSION 2.8)
project(cpp)

#Defaults
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin/linux")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w") #No Warnings while compiling


#Openmp flags
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#Boost-Py
FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)
FIND_PACKAGE(Boost COMPONENTS python)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
#LINK_LIBRARIES(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!



set(SOURCE_FILES
    demo/source/Classification.cpp
    demo/source/CommandLineParser.cpp
    demo/source/CumulativeNormalDistribution.cpp
    demo/source/DataPointCollection.cpp
    #demo/source/DensityEstimation.cpp
    demo/source/dibCodec.cpp
    demo/source/FeatureResponseFunctions.cpp
    demo/source/FloydWarshall.cpp
    demo/source/Graphics.cpp
    demo/source/main.cpp
    demo/source/Platform.cpp
    demo/source/PlotCanvas.cpp
    #demo/source/Regression.cpp
    #demo/source/SemiSupervisedClassification.cpp
    demo/source/StatisticsAggregators.cpp)

include_directories("${CMAKE_SOURCE_DIR}/lib")
include_directories("${CMAKE_SOURCE_DIR}/demo/source")


#SVM TEST
add_library(svmlib demo/source/svm_utils.cpp)
add_executable(svm_test demo/source/svm_test.cpp)
target_link_libraries(svm_test svmlib svm)
#add_library(classification demo/source/Classification.cpp)
#target_link_libraries(classification svmlib svm)


#Main Stuff
add_executable(sw ${SOURCE_FILES})
target_link_libraries(sw ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} svmlib svm)
