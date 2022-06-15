// 2022-06-14 10:53
#include <iostream>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void Softmax(float*, float*, int);

using namespace nvinfer1;

class MnistSoftmaxPluginV2 : public IPluginV2IOExt {
   public:
    MnistSoftmaxPluginV2() { std::cout << __FUNCTION__ << std::endl; }

    MnistSoftmaxPluginV2(const PluginFieldCollection fc) {}

    MnistSoftmaxPluginV2(const void* data, size_t length) {}

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        return Dims4(1, 10, 1, 1);
    }

    int initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        float* dst = reinterpret_cast<float*>(outputs[0]);
        const float* src = reinterpret_cast<const float*>(inputs[0]);

        Softmax(dst, const_cast<float*>(src), 10);
        // cudaError_t error = cudaMemcpy(dst, tmp, 40, cudaMemcpyHostToDevice);
        return 0;
    }

    size_t getSerializationSize() const noexcept override { return 0; }

    void serialize(void* buffer) const noexcept override {}

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {}

    //! The combination of kLINEAR + kINT8/kHALF/kFLOAT is supported.
    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        // bool condition = inOut[pos].format == TensorFormat::kLINEAR;
        // condition &= inOut[pos].type != DataType::kINT32;
        // condition &= inOut[pos].type == inOut[0].type;
        return inOut[pos].type == DataType::kFLOAT;
    }

    DataType getOutputDataType(
        int index, const DataType* inputTypes,
        int nbInputs) const noexcept override {
        (void)index;
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override { return "SOFTMAX"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    void destroy() noexcept override { delete this; }

    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new MnistSoftmaxPluginV2(*this);
        return plugin;
    }

    void setPluginNamespace(const char* libNamespace) noexcept override {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

    bool isOutputBroadcastAcrossBatch(
        int outputIndex, const bool* inputIsBroadcasted,
        int nbInputs) const noexcept override {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override {
        return false;
    }

   private:
    Dims mInputDims;
    Dims mOutputDims;
    std::string mNamespace;
};

class MnistSoftmaxPluginV2Creator : public IPluginCreator {
   public:
    const char* getPluginName() const noexcept override { return "SOFTMAX"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override {
        return &mFieldCollection;
    }
    IPluginV2* createPlugin(
        const char* name, const PluginFieldCollection* fc) noexcept override {
        auto* plugin = new MnistSoftmaxPluginV2(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }
    IPluginV2* deserializePlugin(
        const char* name, const void* serialData,
        size_t serialLength) noexcept override {
        auto* plugin = new MnistSoftmaxPluginV2(serialData, serialLength);
        mPluginName = name;
        return plugin;
    }
    void setPluginNamespace(const char* libNamespace) noexcept override {
        mNamespace = libNamespace;
    }
    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

   private:
    std::string mNamespace;
    std::string mPluginName;
    PluginFieldCollection mFieldCollection{0, nullptr};
};
