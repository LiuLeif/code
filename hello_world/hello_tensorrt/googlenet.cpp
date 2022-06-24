#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "convolution_plugin.h"
#include "inner_product_plugin.h"
#include "lrn_plugin.h"
#include "opencv2/imgproc.hpp"
#include "pooling_plugin.h"
#include "power_plugin.h"
#include "relu_plugin.h"
#include "softmax_plugin.h"

using namespace nvinfer1;
#define WIDTH 224
#define HEIGHT 224

void readBMP(const char* filename, float* data) {
    cv::Mat image;
    image = cv::imread(filename, 1);
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(WIDTH, WIDTH), cv::INTER_LINEAR);
    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        data[i] = (float)rgb_image.data[i * 3] - 104.0;
        data[i + WIDTH * HEIGHT] = (float)rgb_image.data[i * 3 + 1] - 117.0;
        data[i + WIDTH * HEIGHT * 2] = (float)rgb_image.data[i * 3 + 2] - 123.0;
    }
}

class Logger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* msg) noexcept override {
        std::cout << msg << std::endl;
    }
};

class SampleGoogleNet {
   public:
    SampleGoogleNet() {}
    bool build();
    bool infer();
    bool teardown();

   private:
    bool constructNetwork(
        std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
        std::unique_ptr<nvinfer1::INetworkDefinition>& network);

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};

    nvinfer1::Dims mInputDims;
    nvinfer1::Dims mOutputDims;

    std::unique_ptr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob;
};

Logger logger;

bool SampleGoogleNet::build() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(0));
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    auto parser = std::unique_ptr<nvcaffeparser1::ICaffeParser>(
        nvcaffeparser1::createCaffeParser());
    constructNetwork(parser, network);

    builder->setMaxBatchSize(1);
    config->setMaxWorkspaceSize(1 << 20);
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

    std::unique_ptr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};

    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();
    return true;
}

bool SampleGoogleNet::constructNetwork(
    std::unique_ptr<nvcaffeparser1::ICaffeParser>& parser,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network) {
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(
        "model/googlenet.prototxt", "model/googlenet.caffemodel", *network,
        nvinfer1::DataType::kFLOAT);

    // network->markOutput(*blobNameToTensor->find("pool1/3x3_s2"));
    // network->markOutput(*blobNameToTensor->find("pool2/3x3_s2"));
    // network->markOutput(*blobNameToTensor->find("inception_3a/pool"));
    // network->markOutput(*blobNameToTensor->find("inception_3a/pool"));
    network->markOutput(*blobNameToTensor->find("prob"));

    return true;
}

bool SampleGoogleNet::teardown() {
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

bool SampleGoogleNet::infer() {
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());

    int inputSize = std::accumulate(
        mInputDims.d, mInputDims.d + mInputDims.nbDims, 1,
        std::multiplies<int>());
    int outputSize = std::accumulate(
        mOutputDims.d, mOutputDims.d + mOutputDims.nbDims, 1,
        std::multiplies<int>());
    std::cout << "input size: " << inputSize << std::endl;
    std::cout << "output size: " << outputSize << std::endl;
    void* hostInputBuffer = malloc(inputSize * sizeof(float));
    void* hostOutputBuffer = malloc(outputSize * sizeof(float));
    void* deviceInputBuffer;
    void* deviceOutputBuffer;
    cudaMalloc(&deviceInputBuffer, inputSize * sizeof(float));
    cudaMalloc(&deviceOutputBuffer, outputSize * sizeof(float));

    readBMP(std::string("data/tench.bmp").c_str(), (float*)hostInputBuffer);
    cudaMemcpy(
        deviceInputBuffer, hostInputBuffer, inputSize * sizeof(float),
        cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* bindings[2] = {
        deviceInputBuffer,
        deviceOutputBuffer,
    };
    context->enqueue(1, bindings, stream, nullptr);
    cudaError_t error = cudaMemcpy(
        hostOutputBuffer, deviceOutputBuffer, outputSize * sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    printf("output:\n");
    for (int i = 0; i < std::min<int>(outputSize, 16); i++) {
        std::cout << ((float*)hostOutputBuffer)[i] << " ";
    }
    std::cout << std::endl;
    return true;
}

int main(int argc, char** argv) {
    REGISTER_TENSORRT_PLUGIN(SoftmaxPluginCreator);
    REGISTER_TENSORRT_PLUGIN(PowerPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ReluPluginCreator);
    REGISTER_TENSORRT_PLUGIN(PoolingPluginCreator);
    REGISTER_TENSORRT_PLUGIN(InnerProductPluginCreator);
    REGISTER_TENSORRT_PLUGIN(ConvolutionPluginCreator);
    REGISTER_TENSORRT_PLUGIN(LRNPluginCreator);

    SampleGoogleNet sample;
    sample.build();
    sample.infer();
    sample.teardown();
    return 0;
}
