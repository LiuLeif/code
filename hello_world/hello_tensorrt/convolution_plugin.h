// 2022-06-14 10:53
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"

extern void Convolution(
    float*, const float*, int, int, int, int, int, int, int, int, int, int,
    float*, float*, cudaStream_t);

extern void ConvolutionInt8(
    int8_t*, const int8_t*, int, int, int, int, int, int, int, int, int, int,
    int, int, float*, float*, cudaStream_t);

using namespace nvinfer1;

class ConvolutionPlugin : public IPluginV2IOExt {
   public:
    ConvolutionPlugin(const PluginFieldCollection fc) {
        for (int i = 0; i < fc.nbFields; i++) {
            auto field = fc.fields[i];
            if (std::string(field.name) == "num_output") {
                this->mOutputChannel = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_weights") {
                this->mKernelWeights = *(Weights*)field.data;
            }
            if (std::string(field.name) == "bias_weights") {
                this->mBiasWeights = *(Weights*)field.data;
            }
            if (std::string(field.name) == "kernel_h") {
                this->mKernelH = *((int*)field.data);
            }
            if (std::string(field.name) == "kernel_w") {
                this->mKernelW = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_h") {
                this->mStrideH = *((int*)field.data);
            }
            if (std::string(field.name) == "stride_w") {
                this->mStrideW = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_h") {
                this->mPadH = *((int*)field.data);
            }
            if (std::string(field.name) == "pad_w") {
                this->mPadW = *((int*)field.data);
            }
        }
    }

    ConvolutionPlugin(const void* data, size_t length) {
        mInputChannel = ((int*)data)[0];
        mOutputChannel = ((int*)data)[1];
        mH = ((int*)data)[2];
        mW = ((int*)data)[3];
        mKernelH = ((int*)data)[4];
        mKernelW = ((int*)data)[5];
        mStrideH = ((int*)data)[6];
        mStrideW = ((int*)data)[7];
        mPadH = ((int*)data)[8];
        mPadW = ((int*)data)[9];

        int kc = ((int*)data)[10];
        int bc = ((int*)data)[11];
        mType = ((int*)data)[12];
        mInputScale = ((int*)data)[13];
        mOutputScale = ((int*)data)[14];
        float* kernel = (float*)malloc(kc * 4);
        float* bias = (float*)malloc(bc * 4);
        memcpy(kernel, ((int*)data) + 15, kc * 4);
        memcpy(bias, ((int*)data) + 15 + kc, bc * 4);
        mKernelWeights = Weights{
            .type = DataType::kFLOAT,
            .values = kernel,
            .count = kc,
        };
        mBiasWeights = Weights{
            .type = DataType::kFLOAT,
            .values = bias,
            .count = bc,
        };
    }

   public:
    int getNbOutputs() const noexcept override { return 1; }

    Dims getOutputDimensions(
        int index, const Dims* inputs, int nbInputDims) noexcept override {
        int channel = inputs->d[0];
        int h = inputs->d[1];
        int w = inputs->d[2];

        Dims3 outputDims;
        outputDims.nbDims = 3;
        outputDims.d[0] = mOutputChannel;
        outputDims.d[1] = (h + 2 * mPadH - mKernelH) / mStrideH + 1;
        outputDims.d[2] = (w + 2 * mPadW - mKernelW) / mStrideW + 1;
        return outputDims;
    }

    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override {
        return 0;
    }

    int enqueue(
        int batchSize, const void* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override {
        // std::cout << *this;
        if (mType == (int)DataType::kFLOAT) {
            float* dst = reinterpret_cast<float*>(outputs[0]);
            const float* src = reinterpret_cast<const float*>(inputs[0]);
            Convolution(
                dst, src, mInputChannel, mOutputChannel, mH, mW, mKernelH,
                mKernelW, mStrideH, mStrideW, mPadH, mPadW,
                (float*)mKernelWeights.values, (float*)mBiasWeights.values,
                stream);
        } else {
            int8_t* dst = reinterpret_cast<int8_t*>(outputs[0]);
            const int8_t* src = reinterpret_cast<const int8_t*>(inputs[0]);
            ConvolutionInt8(
                dst, src, mInputScale, mOutputScale, mInputChannel,
                mOutputChannel, mH, mW, mKernelH, mKernelW, mStrideH, mStrideW,
                mPadH, mPadW, (float*)mKernelWeights.values,
                (float*)mBiasWeights.values, stream);
        }

        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return (12 + 3 + mKernelWeights.count + mBiasWeights.count) * 4;
    }

    void serialize(void* buffer) const noexcept override {
        ((int*)buffer)[0] = mInputChannel;
        ((int*)buffer)[1] = mOutputChannel;
        ((int*)buffer)[2] = mH;
        ((int*)buffer)[3] = mW;
        ((int*)buffer)[4] = mKernelH;
        ((int*)buffer)[5] = mKernelW;
        ((int*)buffer)[6] = mStrideH;
        ((int*)buffer)[7] = mStrideW;
        ((int*)buffer)[8] = mPadH;
        ((int*)buffer)[9] = mPadW;
        ((int*)buffer)[10] = mKernelWeights.count;
        ((int*)buffer)[11] = mBiasWeights.count;
        ((int*)buffer)[12] = mType;
        ((int*)buffer)[13] = mInputScale;
        ((int*)buffer)[14] = mOutputScale;
        memcpy(
            ((int*)buffer) + 15, mKernelWeights.values,
            mKernelWeights.count * 4);
        memcpy(
            ((int*)buffer) + 15 + mKernelWeights.count, mBiasWeights.values,
            mBiasWeights.count * 4);
    }

    void configurePlugin(
        const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out,
        int nbOutput) noexcept override {
        mType = (int)in[0].type;
        mInputScale = in[0].scale;
        mOutputScale = out[0].scale;
        auto dims = in[0].dims;
        mInputChannel = dims.d[0];
        mH = dims.d[1];
        mW = dims.d[2];
    }

    bool supportsFormatCombination(
        int pos, const PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) const noexcept override {
        // std::cout << (int)inOut[pos].format << " " << (int)inOut[pos].type
        //           << " " << (int)inOut[0].type << std::endl;
        return inOut[pos].format == TensorFormat::kLINEAR &&
               (inOut[pos].type == DataType::kFLOAT ||
                inOut[pos].type == DataType::kINT8) &&
               inOut[pos].type == inOut[0].type;
    }
    DataType getOutputDataType(
        int index, const DataType* inputTypes,
        int nbInputs) const noexcept override {
        (void)index;
        return inputTypes[0];
    }

    const char* getPluginType() const noexcept override {
        return "CONVOLUTION";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    void destroy() noexcept override { delete this; }
    IPluginV2Ext* clone() const noexcept override {
        auto* plugin = new ConvolutionPlugin(*this);
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

    friend std::ostream& operator<<(
        std::ostream& os, const ConvolutionPlugin& c) {
        // clang-format off
        return (os
                << " input channel: " << c.mInputChannel
                << " output channel: " << c.mOutputChannel
                << " h: " << c.mH
                << " w: " << c.mW
                << " kernel: " << c.mKernelH << " " << c.mKernelW
                << " stride: " << c.mStrideH << " " << c.mStrideW
                << " pad: " << c.mPadH << " " << c.mPadW
                << " type: " << c.mType << " scale: " << c.mInputScale << " " << c.mOutputScale
                << std::endl
        );
        // clang-format on
    }

   private:
    int mOutputChannel;
    int mInputChannel;
    int mH;
    int mW;
    Weights mKernelWeights;
    Weights mBiasWeights;
    int mKernelH;
    int mKernelW;
    int mStrideH;
    int mStrideW;
    int mPadH;
    int mPadW;
    int mType;
    int mInputScale;
    int mOutputScale;
    std::string mNamespace;
};

class ConvolutionPluginCreator : public IPluginCreator {
   public:
    const char* getPluginName() const noexcept override {
        return "CONVOLUTION";
    }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override {
        return &mFieldCollection;
    }
    IPluginV2* createPlugin(
        const char* name, const PluginFieldCollection* fc) noexcept override {
        auto* plugin = new ConvolutionPlugin(*fc);
        mFieldCollection = *fc;
        mPluginName = name;
        return plugin;
    }
    IPluginV2* deserializePlugin(
        const char* name, const void* serialData,
        size_t serialLength) noexcept override {
        auto* plugin = new ConvolutionPlugin(serialData, serialLength);
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
