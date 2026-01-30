# RDKS100P 模型部署全流程

## 内容：

量化：pytorch训练的浮点型模型转板端bpu推理的int8的hbm模型，模型中间格式onnx。

部署：使用转化后的hbm模型在板端部署

模型量化工具链：OE，天工开物

统一计算平台（UCP）：板端部署库文件

## 模型部署流程：

1. 得到模型的onnx文件
2. 借助OE工具链，将其转化为BPU推理使用的hbm文件
3. 根据模板和文档，使用模型推理
4. BPU 要求输入数据的每一行起始地址（Stride）必须是 32 字节的倍数。

## 以deeplabv3+为例，记录一次模型部署流程。

1. hb_compile: 将onnx格式模型转化为hbm模型

   检查onnx模型中有那些算子是BPU不支持的。

   删除特定的节点名称，参考官方给的例程

   转化的验证集，特别注意，验证集的图片分辨率要和模型的输入大小一致，npy格式数据

   配置文件 参考官方给出的例子进行修改

   转化之后会得到一些日志文件和中间模型

   html文件可以查看模型的静态特性

   动态精度：hrt_model_exec perf --model_file model.hbm

2. hb_model_info：解析hbm和bc编译时的依赖及参数信息、onnx模型基本信息，同时支持对bc可删除节点进行查询

   hb_model_info ${model_file}

   重点关注输入和输出

   ​	input_y  input  [1, 224, 224, 1] UINT8 

   ​	input_uv input  [1, 112, 112, 2] UINT8 

   ​	output   output [1, 1000]        FLOAT32

   deeplabv3+的输出信息为：

   ​	2026-01-19 16:22:17,965 INFO input.1_y  input  [1, 1024, 2048, 1] UINT8
   ​	2026-01-19 16:22:17,966 INFO input.1_uv input  [1, 512, 1024, 2]  UINT8
   ​	2026-01-19 16:22:17,966 INFO 705        output [1, 1, 1024, 2048] INT8

   它的输出为[1, 1, 1024, 2048]

3. 根据官方的其他模型推理案例的源码来进行测试和修改，主要在于数据的对齐很难搞，输出格式不对不能输出正确的结果。

   板端部署：统一计算平台(这个是OE工具链里带的，这个工具链的docker镜像是部署在开发机的wsl ubuntu环境)

   官方示例：RGB输入的ResNet18模型部署

   ```c++
   #include <fstream>
   #include <iostream>
   #include <vector>
   #include <cstring>
   
   #include "hobot/dnn/hb_dnn.h"
   #include "hobot/hb_ucp.h"
   #include "hobot/hb_ucp_sys.h"
   
   const char* hbm_path = "resnet18_224x224_rgb.hbm";
   std::string data_path = "input.bin";
   
   // Read binary input file
   int read_binary_file(std::string file_path, char **bin, int *length) {
       std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
       ifs.seekg(0, std::ios::end);
       *length = ifs.tellg();
       ifs.seekg(0, std::ios::beg);
       *bin = new char[sizeof(char) * (*length)];
       ifs.read(*bin, *length);
       ifs.close();
       return 0;
   }
   
   int main() {
     // Get model handle
     hbDNNPackedHandle_t packed_dnn_handle;
     hbDNNHandle_t dnn_handle;
     hbDNNInitializeFromFiles(&packed_dnn_handle, &hbm_path, 1);
     const char **model_name_list;
     int model_count = 0;
     hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
     hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
   
     // Prepare input and output tensor
     std::vector<hbDNNTensor> input_tensors;
     std::vector<hbDNNTensor> output_tensors;
     input_tensors.resize(1);   // This model has only one input
     output_tensors.resize(1);  // This model has only one output
   
     // Initialize and malloc the input tensor
     hbDNNTensor input = input_tensors[0];
     hbDNNGetInputTensorProperties(&input.properties, dnn_handle, 0);
     int input_memSize = input.properties.alignedByteSize;
     hbUCPMallocCached(&input.sysMem, input_memSize, 0);
   
     // Initialize and malloc the output tensor
     hbDNNTensor output = output_tensors[0];
     hbDNNGetOutputTensorProperties(&output.properties, dnn_handle, 0);
     int output_memSize = output.properties.alignedByteSize;
     hbUCPMallocCached(&output.sysMem, output_memSize, 0);
   
     // Copy binary input data to input tensor
     int32_t data_length = 0;
     char *data = nullptr;
     auto ret = read_binary_file(data_path, &data, &data_length);
     memcpy(reinterpret_cast<char *>(input.sysMem.virAddr),
            data, input.sysMem.memSize);
     hbUCPMemFlush(&(input.sysMem), HB_SYS_MEM_CACHE_CLEAN);
   
     // Submit task and wait till it completed
     hbUCPTaskHandle_t task_handle{nullptr};
     hbDNNInferV2(&task_handle, &output, &input, dnn_handle);
     hbUCPSchedParam ctrl_param;
     HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
     ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
     hbUCPSubmitTask(task_handle, &ctrl_param);
     hbUCPWaitTaskDone(task_handle, 0);
   
     // Parse inference result and calculate TOP1
     hbUCPMemFlush(&output.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
     auto result = reinterpret_cast<float *>(output.sysMem.virAddr);
     float max_score = 0.0;
     int label = -1;
     for (auto i = 0; i < 1000; i++) {
       float score = result[i];
       if (score > max_score) {
         label = i;
         max_score = score;
       }
     }
     std::cout << "label: " << label << std::endl;
   }
   ```

## 其他PTQ转化工具：

- hb_verifier: 一致性验证工具，两模型之间的余弦相似度对比、输出一致性对比、余弦相似度越接近1，说明对比的两个量化模型的输出越接近。一致性对比会打印对比模型的输出一致性信息，包括输出名称、一致性、不一致元素数量、最大绝对误差、最大相对误差
- ![image-20260119211116659](C:\Users\86159\AppData\Roaming\Typora\typora-user-images\image-20260119211116659.png)
- hb_eval_preprocess: 用于对模型精度进行评估时，在x86环境下对图片数据进行预处理。
- hb_config_generator: 用于支持您获取模型编译最简yaml配置文件、包含全部参数默认值的yaml配置文件的工具。
  - hb_config_generator --simple-yaml：生成嘴贱yaml配置文件
  - hb_config_generator --simple-yaml --model model.onnx --march nash-e：生成基于模型信息的最简yaml配置文件
  - hb_config_generator --full-yaml：包含全部参数默认值的yaml配置文件
  - hb_config_generator --full-yaml --model model.onnx --march nash-e：生成基于模型信息的全部参数默认值yaml配置文件

**HBRuntime推理库：**地平线提供的一套x86端模型推理库，支持对常用训练框架直接导出的onnx原始模型、地平线工具链进行PTQ转化过程中产出的各阶段onnx模型以及地平线工具链转化过程中产出的bc模型和hbm模型进行推理

```python
import numpy as np
# 加载地平线依赖库
from horizon_tc_ui.hb_runtime import HBRuntime

# 准备模型运行的输入，此处`input.npy`为处理好的数据
data = np.load("input.npy")    
# 加载模型文件，根据实际模型进行设置
# ONNX模型
sess = HBRuntime("model.onnx")
# HBIR模型
sess = HBRuntime("model.bc")
# HBM模型
sess = HBRuntime("model.hbm")
# 获取输入&输出节点名称
input_names = sess.input_names
output_names = sess.output_names

# 准备输入数据，根据实际输入类型和layout进行准备，配置格式要求为字典形式，输入名称和输入数据组成键值对
# 如模型仅有一个输入
input_feed = {input_names[0]: data}
# 如模型有多个输入
input_feed = {input_names[0]: data1, input_names[1]: data2}
# 进行模型推理，推理的返回值是一个list，依次与output_names指定名称一一对应
output = sess.run(output_names, input_feed)
```

ONNX模型推理：

```py
import numpy as np
# 加载地平线依赖库
from horizon_tc_ui.hb_runtime import HBRuntime

# 准备模型运行的输入，此处`input.npy`为处理好的数据
data = np.load("input.npy")  
# 加载模型文件
sess = HBRuntime("model.onnx")
# 获取模型输入节点信息
input_names = sess.input_names

# 假设此模型只有一个输入，开始模型推理
output = sess.run(None, {input_names[0]: data})
```

HBIR（.bc）模型推理

```py
import numpy as np
# 加载地平线依赖库
from horizon_tc_ui.hb_runtime import HBRuntime

# 准备模型运行的输入，此处`input.npy`为处理好的数据
data = np.load("input.npy")  
# 加载模型文件
sess = HBRuntime("model.bc")
# 获取模型输入节点信息
input_names = sess.input_names

# 假设此模型只有一个输入，开始模型推理
output = sess.run(None, {input_names[0]: data})
```

HBM模型推理

```py
import numpy as np
# 加载地平线依赖库
from horizon_tc_ui.hb_runtime import HBRuntime

# 准备模型运行的输入，此处`input.npy`为处理好的数据
data = np.load("input.npy")  
# 加载模型文件
sess = HBRuntime("model.hbm")
# 获取模型输入节点信息
input_names = sess.input_names

# 假设此模型只有一个输入，开始模型推理
output = sess.run(None, {input_names[0]: data})
```

Debug工具：暂时还没有仔细研究，列为TODO

## 模型部署示例：

### deeplabv3+模型部署

```c++
#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

///////////////////////////////////// 保存与可视化 //////////////////////////////////////////
/**
 * @brief 将分割掩码转化为彩色图像
 * @param mask 预测结果 (单通道灰度，值为类别ID)
 * @param width 图像宽度
 * @param height 图像高度
 * @param filename 保存的路径
 */
void save_segmentation_result(const std::vector<uint8_t>& mask, int width, int height, const std::string& filename) {
    if (mask.empty()) return;
    // 将 mask 向量转为 cv::Mat，将vector 数据映射为OpenCV的单通道矩阵(CV_8UC1)
    cv::Mat mask_mat(height, width, CV_8UC1, (void*)mask.data());
    // 存放最终彩色结果的矩阵
    cv::Mat color_mat;
    // 1. mask_mat * 15: 将较小的类别索引（如 0, 1, 2）放大，以便在伪彩色映射中产生明显的颜色区分
    // 2. applyColorMap: 将灰度值映射为 COLORMAP_JET 调色板颜色（蓝-绿-红渐变）
    cv::applyColorMap(mask_mat * 15, color_mat, cv::COLORMAP_JET); 
    // 将处理后的彩色图像写入文件
    cv::imwrite(filename, color_mat);
    std::cout << "Successfully saved segmentation result to " << filename << std::endl;
}

/**
 * 将分割结果叠加到原图上
 * @param src 原图 (BGR)
 * @param mask 预测结果 (单通道灰度，值为类别ID)
 * @param filename 保存的文件名
 */
void blend_segmentation(cv::Mat& src, const std::vector<uint8_t>& mask, const std::string& filename) {
    int h = src.rows; // 矩阵行数，对应图像的高度
    int w = src.cols; // 矩阵列数，对应图像的宽度

    // 1. 将 mask 向量转为 cv::Mat
    cv::Mat mask_mat(h, w, CV_8UC1, (void*)mask.data());
    // 2. 将 ID 映射为彩色图 (使用 COLORMAP_JET)
    cv::Mat color_mask;
    cv::applyColorMap(mask_mat * 15, color_mask, cv::COLORMAP_JET);
    // 3. 图像融合：dst = src * 0.6 + color_mask * 0.4 + 0
    cv::Mat blended;
    cv::addWeighted(src, 0.6, color_mask, 0.4, 0, blended);
    // 4. 保存并显示
    cv::imwrite(filename, blended);
    std::cout << "Blended result saved to: " << filename << std::endl;
}

///////////////////////////////////// 数据处理 /////////////////////////////////////
/**
 * @brief 准备分量分离的 NV12 输入数据
 * @param image_path 本地图像路径
 * @param inputs     输入张量数组（inputs[0]为Y，inputs[1]为UV）
 * @return int       成功返回 0
 */
int prepare_nv12_input(const std::string& image_path, std::vector<hbDNNTensor>& inputs) {
    if (inputs.size() < 2) return -1;

    // 获取 Y 分量的尺寸
    int h = inputs[0].properties.validShape.dimensionSize[1];
    int w = inputs[0].properties.validShape.dimensionSize[2];

    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) return -1;
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(w, h));
    cv::Mat yuv420p;
    cv::cvtColor(resized, yuv420p, cv::COLOR_BGR2YUV_I420);

    // --- 填充 Input[0] (Y) ---
    uint8_t* y_src = yuv420p.data;
    uint8_t* y_dest = reinterpret_cast<uint8_t*>(inputs[0].sysMem.virAddr);
    int y_stride = static_cast<int>(inputs[0].properties.stride[1]);
    for (int i = 0; i < h; ++i) {
        std::memcpy(y_dest + i * y_stride, y_src + i * w, w);
    }
    hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);

    // --- 填充 Input[1] (UV) ---
    uint8_t* u_src = yuv420p.data + (h * w);
    uint8_t* v_src = u_src + (h * w / 4);
    uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs[1].sysMem.virAddr);
    int uv_stride = static_cast<int>(inputs[1].properties.stride[1]);
    for (int i = 0; i < h / 2; ++i) {
        uint8_t* row_dest = uv_dest + i * uv_stride;
        for (int j = 0; j < w / 2; ++j) {
            row_dest[j * 2] = u_src[i * (w / 2) + j];
            row_dest[j * 2 + 1] = v_src[i * (w / 2) + j];
        }
    }
    hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);

    return 0;
}

////////////////////////////////////// main函数 ///////////////////////////////////////
int main(int argc, char **argv) {
    // hbDNNPackedHandle_t：指向打包的多个模型。
    hbDNNPackedHandle_t packed_dnn_handle;
    // 指定模型文件路径
    const char* model_file_name = "/home/sunrise/Desktop/RDKS100_Drowning/tem/deeplabv3plus_efficientnetm2_1024x2048_nv12.hbm";
    const char* image_file_name = "/home/sunrise/Desktop/test_deeplabv3+/test.png";
    /*
    从文件完成对dnnPackedHandle的创建和初始化。调用方法可以跨函数、跨线程使用返回的dnnPackedHandle
    * hbDNNPackedHandle_t *dnnPackedHandle：指向多个模型
    * char const **modelFileNames：模型文件路径
    * int32_t modelFileCount：模型文件数量
    * return：0 表示API成功
    int32_t hbDNNInitializeFromFiles(hbDNNPackedHandle_t *dnnPackedHandle,
                                 char const **modelFileNames,
                                 int32_t modelFileCount);
    */
    int ret = hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);
    if (ret != 0) return -1;

    const char **model_name_list;
    int model_count = 0;
    /*
    获取dnnPackedHandle中包含的模型名称列表和个数
    * char const ***modelNameList：模型名称列表
    * int32_t *modelNameCount：模型名称数量
    * hbDNNPackedHandle_t dnnPackedHandle：指向多个模型
    * return：0 表示API成功
    int32_t hbDNNGetModelNameList(char const ***modelNameList, 
                              int32_t *modelNameCount,
                              hbDNNPackedHandle_t dnnPackedHandle);
    */
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    // hbDNNHandle_t：指向单一模型
    hbDNNHandle_t dnn_handle;
    /*
    从dnnPackedHandle所指向模型列表中获取一个模型的句柄，调用方可以跨函数、跨线程使用返回的dnnHandle
    * hbDNNHandle_t *dnnHandle：指向一个模型
    * hbDNNPackedHandle_t dnnPackedHandle：指向多个模型
    * char const *modelName：模型名称
    * return：0 表示API成功
    int32_t hbDNNGetModelHandle(hbDNNHandle_t *dnnHandle,
                            hbDNNPackedHandle_t dnnPackedHandle,
                            char const *modelName);
     */
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    int input_count = 0;
    /*
    获取dnnHandle所指向模型的输入tensor数量
    * int32_t *inputCount：输入tensor数量
    * hbDNNHandle_t dnnHandle：指向一个模型
    * return：0 表示API成功
    int32_t hbDNNGetInputCount(int32_t *inputCount, 
                           hbDNNHandle_t dnnHandle);
    */
    hbDNNGetInputCount(&input_count, dnn_handle);
    // hbDNNTensor：输入tensor，用于存放输入输出的信息
    std::vector<hbDNNTensor> input(input_count);
    /*
    获取dnnHandle所指向模型的输入tensor属性
    * hbDNNTensorProperties *properties：输入tensor属性
    * hbDNNHandle_t dnnHandle：指向一个模型
    * int32_t inputIndex：输入tensor索引
    * return：0 表示API成功
    int32_t hbDNNGetInputTensorProperties(hbDNNTensorProperties *properties,
                                      hbDNNHandle_t dnnHandle,
                                      int32_t inputIndex);
    */
    for (int i = 0; i < input_count; i++) {
        hbDNNGetInputTensorProperties(&input[i].properties, dnn_handle, i);
        auto &props = input[i].properties;

        if (props.stride[0] < 0) {
            // std::cout << "Fixing input[" << i << "] stride..." << std::endl;
            if (i == 0) { // Y分量
                // Input[0]: 1x1024x2048x1 (NV12)
                props.stride[0] = 2097152; 
                props.stride[1] = 2048;
                props.stride[2] = 1;
                props.stride[3] = 1;
                props.alignedByteSize = 2097152; // 1024 * 2048 * 1.5
            } else if (i == 1) { // UV分量
                // Input[1]: 1x512x1024x2 (NV12)
                // 报错要求: stride[0] >= 1048576, stride[1] >= 2048
                props.stride[0] = 1048576; 
                props.stride[1] = 2048;
                props.stride[2] = 2; // 因为最后一个维度是 2
                props.stride[3] = 1;
                props.alignedByteSize = 1048576; // 512 * 2048
            }
        }

        hbUCPMallocCached(&input[i].sysMem, props.alignedByteSize, 0); 
    }
    prepare_nv12_input(image_file_name, input);

    int output_count = 0;
    /*
    获取dnnHandle所指向模型的输出tensor数量
    * int32_t *outputCount：输出tensor数量
    * hbDNNHandle_t dnnHandle：指向一个模型
    * return：0 表示API成功
    int32_t hbDNNGetOutputCount(int32_t *outputCount, 
                            hbDNNHandle_t dnnHandle);
    */
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> output(output_count);
    /*
    获取dnnHandle所指向模型的输出tensor属性
    * hbDNNTensorProperties *properties：输出tensor的信息
    * hbDNNHandle_t dnnHandle：指向一个模型
    * int32_t outputIndex：输出tensor索引
    * return：0 表示API成功
    int32_t hbDNNGetOutputTensorProperties(hbDNNTensorProperties *properties,
                                       hbDNNHandle_t dnnHandle, 
                                       int32_t outputIndex);
    */
    for (int i = 0; i < output_count; i++){
        hbDNNGetOutputTensorProperties(&output[i].properties, dnn_handle, i);
        auto &props = output[i].properties;

        if (props.stride[0] < 0) {
            int64_t h = static_cast<int64_t>(props.validShape.dimensionSize[2]);
            int64_t w = static_cast<int64_t>(props.validShape.dimensionSize[3]);
            props.stride[0] = h * w;
            props.stride[1] = h * w;
            props.stride[2] = w;
            props.stride[3] = 1;
            props.alignedByteSize = static_cast<uint64_t>(h * w);
        }
        hbUCPMallocCached(&output[i].sysMem, props.alignedByteSize, 0);
    }

    hbUCPTaskHandle_t task_handle = nullptr;
    ret = hbDNNInferV2(&task_handle, output.data(), input.data(), dnn_handle);
    if (ret != 0 || task_handle == nullptr) return -1;

    hbUCPSchedParam infer_sched_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&infer_sched_param);
    hbUCPSubmitTask(task_handle, &infer_sched_param);
    hbUCPWaitTaskDone(task_handle, 0);

    // 在循环外部先读取一次原图，用于叠加
    cv::Mat original_img = cv::imread(image_file_name);

    for (int i = 0; i < output_count; i++) {
        hbUCPMemFlush(&(output[i].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
        auto &props = output[i].properties;
        
        int h = props.validShape.dimensionSize[2];
        int w = props.validShape.dimensionSize[3];
        int stride = static_cast<int>(props.stride[2]); 

        if (h <= 0 || w <= 0) continue;

        int8_t *raw_ptr = reinterpret_cast<int8_t *>(output[i].sysMem.virAddr);
        std::vector<uint8_t> seg_mask(static_cast<size_t>(h * w));

        for (int row = 0; row < h; ++row) {
            std::memcpy(seg_mask.data() + row * w, raw_ptr + row * stride, static_cast<size_t>(w));
        }

        // 保存纯色分割图
        save_segmentation_result(seg_mask, w, h, "output_mask.png");

        // 保存半透明叠加图
        if (!original_img.empty()) {
            cv::Mat resized_src;
            cv::resize(original_img, resized_src, cv::Size(w, h)); // 确保原图尺寸与输出一致
            blend_segmentation(resized_src, seg_mask, "output_blended.png");
        }
    }
    // 释放资源
    hbUCPReleaseTask(task_handle);
    for (auto &in : input) hbUCPFree(&in.sysMem);
    for (auto &out : output) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);
    return 0;
}
```

```cmake
cmake_minimum_required(VERSION 3.0)
project(deeplabv3+)
# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 设置编译器标志
# libdnn.so depends on system software dynamic link library, use -Wl,-unresolved-symbols=ignore-in-shared-libs to shield during compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,-unresolved-symbols=ignore-in-shared-libs")
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_C_FLAGS_DEBUG " -Wall -Werror -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE " -Wall -Werror -O3 ")
set(CMAKE_C_FLAGS_RELEASE " -Wall -Werror -O3 ")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# 设置OpenCV包
find_package(OpenCV REQUIRED)
# S100 UCP库路径配置
set(HOBOT_INCLUDE_PATH "/usr/include")
set(HOBOT_LIB_PATH "/usr/hobot/lib")
# 包含头文件路径
include_directories(${HOBOT_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
# 链接库路径
link_directories(${HOBOT_LIB_PATH})
# 添加可执行文件
add_executable(main main.cc)
# 链接所需的库
target_link_libraries(main
                      ${OpenCV_LIBS}    # OpenCV库
                      dnn               # S100 DNN推理库
                      hbucp             # S100 UCP统一计算平台库
                      pthread           # 多线程支持
                      rt                # 实时库
                      dl                # 动态链接库支持
                      m                 # 数学库
                      )
```

### rough_waternet(改deeplabv3+模型)模型部署

闸室墙的类别索引为 0，水体的类别索引为 1，船舶的类别索引为 2

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

// 宏定义：检查函数返回值，如果出错则打印错误码并退出
#define CHECK_RET(ret, msg) \
    if ((ret) != 0) { \
        std::cerr << msg << " Error code: " << (ret) << std::endl; \
        return -1; \
    }

int main() {
    const char* model_file = "rough_waternet.hbm"; 
    const char* image_file = "water.jpg";

    // --- 1. 初始化并加载模型 (Init DNN) ---
    hbDNNPackedHandle_t packed_dnn_handle;
    CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN failed");

    // 获取模型名称列表并根据名称获取模型句柄
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    // --- 2. 准备输入 Tensor (Input Tensor Preparation) ---
    int input_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    std::vector<hbDNNTensor> inputs(input_count);

    for (int i = 0; i < input_count; i++) {
        // 获取输入 Tensor 的属性（形状、类型、步长等）
        hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);
        auto &props = inputs[i].properties;
        
        /**
         * 针对 512x512 NV12 模型的输入对齐修复 (Memory Alignment)
         * BPU 硬件要求每一行数据的起始地址必须符合对齐要求（通常是16或32字节）。
         * 这里手动覆盖 Stride 信息以确保与硬件预期的 512 宽度完全匹配。
         */
        if (i == 0) { // Y 分量 (Brightness/Luma)
            props.stride[0] = 262144; // 512 * 512
            props.stride[1] = 512;    // 行步长 (Row Stride)
            props.stride[2] = 1; 
            props.stride[3] = 1;
            props.alignedByteSize = 262144;
        } else { // UV 分量 (Chroma/Color)
            props.stride[0] = 131072; // 512 * 256
            props.stride[1] = 512;    // UV交替存储，一行依然是512字节
            props.stride[2] = 2; 
            props.stride[3] = 1;
            props.alignedByteSize = 131072;
        }
        // 在 BPU Cached 内存中申请空间
        hbUCPMallocCached(&inputs[i].sysMem, props.alignedByteSize, 0);
        inputs[i].sysMem.memSize = props.alignedByteSize;
    }

    // --- 3. 图像预处理 (Preprocessing: BGR to NV12) ---
    cv::Mat bgr = cv::imread(image_file);
    if (!bgr.empty()) {
        cv::Mat resized, yuv420p;
        // 缩放到模型要求的 512x512 尺寸
        cv::resize(bgr, resized, cv::Size(512, 512));
        // 将 BGR 转换为 YUV420P (I420: YYYY...UU...VV) 格式
        cv::cvtColor(resized, yuv420p, cv::COLOR_RGB2YUV_I420);
        
        // 拷贝 Y 数据到第一个输入 Tensor
        std::memcpy(inputs[0].sysMem.virAddr, yuv420p.data, 512 * 512);
        
        // 计算 U 和 V 数据在 I420 内存中的起始位置
        uint8_t* u_src = yuv420p.data + (512 * 512);
        uint8_t* v_src = u_src + (512 * 512 / 4);
        uint8_t* uv_dest = reinterpret_cast<uint8_t*>(inputs[1].sysMem.virAddr);
        
        // 手动将 I420 转换为 NV12 (UV交错存储: UVUVUV...)
        for (int i = 0; i < 256; ++i) { // 高度减半
            for (int j = 0; j < 256; ++j) { // 宽度减半，但每个点存一对 UV
                uv_dest[i * 512 + j * 2] = u_src[i * 256 + j];     // U
                uv_dest[i * 512 + j * 2 + 1] = v_src[i * 256 + j]; // V
            }
        }
        // 刷新 Cache，确保数据从 CPU 同步到 BPU 硬件可见的内存中
        hbUCPMemFlush(&inputs[0].sysMem, HB_SYS_MEM_CACHE_CLEAN);
        hbUCPMemFlush(&inputs[1].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // --- 4. 输出 Tensor 准备 (Output Tensor Preparation) ---
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> outputs(output_count);
    hbDNNGetOutputTensorProperties(&outputs[0].properties, dnn_handle, 0);
    // 申请输出内存：3个类别 * 512 * 512 * 4字节 (Float32 类型)
    outputs[0].properties.alignedByteSize = 3145728; 
    hbUCPMallocCached(&outputs[0].sysMem, 3145728, 0);
    outputs[0].sysMem.memSize = 3145728;

    // --- 5. 执行推理 (Inference) ---
    hbUCPTaskHandle_t task_handle = nullptr;
    // 发起推理任务
    CHECK_RET(hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle), "Infer failed");
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    // 提交任务到 BPU 调度并等待执行结束
    hbUCPSubmitTask(task_handle, &sched);
    hbUCPWaitTaskDone(task_handle, 0);

    // --- 6. 后处理：NCHW 寻址与结果解析 (Post-processing) ---
    // 无效化 Cache，确保读取到的是 BPU 写入的最新的物理内存数据
    hbUCPMemFlush(&(outputs[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);
    float* raw_ptr = (float*)outputs[0].sysMem.virAddr;
    auto &prop = outputs[0].properties;

    int h = prop.validShape.dimensionSize[2]; // 高度: 512
    int w = prop.validShape.dimensionSize[3]; // 宽度: 512
    int c_out = prop.validShape.dimensionSize[1]; // 通道数/类别数: 3

    cv::Mat result(h, w, CV_8UC3);
    // 定义类别对应的颜色调色板 (Class 0, 1, 2)
    std::vector<cv::Vec3b> palette = {{0,0,0}, {0,255,0}, {0,0,255}}; // 黑, 绿, 红
    std::vector<int> class_counts(3, 0);
    float min_val = 1e10f, max_val = -1e10f;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int max_id = 0;
            float current_max = -1e10f;

            // 针对每个像素，在不同通道（类别）中寻找最大概率
            for (int c = 0; c < c_out; c++) {
                /**
                 * 连续 NCHW 寻址公式 (Continuous Memory Access)
                 * 计算公式：通道偏移 + 行偏移 + 列偏移
                 */
                int offset = c * (h * w) + y * w + x; 
                float val = raw_ptr[offset];

                // 记录数值范围以便调试
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;

                // Argmax 逻辑
                if (val > current_max) {
                    current_max = val;
                    max_id = c;
                }
            }
            class_counts[max_id % 3]++;
            // 将预测出的类别 ID 映射到调色板颜色
            result.at<cv::Vec3b>(y, x) = palette[max_id % 3];
        }
    }

    // --- 7. 打印统计信息与结果保存 ---
    std::cout << "--- Inference Debug Info ---" << std::endl;
    std::cout << "Output Value Range: [" << min_val << ", " << max_val << "]" << std::endl;
    for (int k = 0; k < 3; k++) {
        std::cout << "Class " << k << " pixels: " << class_counts[k] 
                  << " (" << (class_counts[k] * 100.0 / (h * w)) << "%)" << std::endl;
    }

    cv::imwrite("output_water.png", result);

    // --- 8. 释放资源 (Resource Cleanup) ---
    hbUCPReleaseTask(task_handle);
    for (auto &in : inputs) hbUCPFree(&in.sysMem);
    for (auto &out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);

    std::cout << "Done! Result saved to output_water.png" << std::endl;
    return 0;
}
```

#### 注意

今天解决了一个模型部署相关的问题，OE工具链提供的deeplabv3+的转化yaml文件是以yuv444格式的训练的，这样会导致一个问题，在后期做stride对齐的时候，会出现各种问题，即使你输入给模型的格式是正确的，但是还是会输出不对，这个问题困扰了我三天，今天偶然间把yuv444修改为rgb，模型的输出就正确了。

后期要把这些都封装成两个类，提供一些必须的API接口。

### mmpose

#### 流程：

1. 环境：mmDeploy

2. 下载原始模型和转化配置文件：

   注意：这里的 mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 可以修改为所需的在openmmlab下提供的模型

   ```
   mim download mmpose --config td-hm_hrnet-w32_8xb64-210e_coco-256x192 --dest .
   ```

3. 模型转化，根据实际的模型地址修改

   ```
   python tools/deploy.py configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py rtmo-m_16xb16-600e_coco-640x640.py rtmo-m_16xb16-600e_coco-640x640-6f4e0306_20231211.pth demo/resources/human-pose.jpg --work-dir mmdeploy_models/mmpose/rtmo --device cpu --show
   ```

4. 将转化好的onnx资源文件放到工具链中，一定要修改验证集图片的分辨率，将其修改为onnx模型的输入尺寸，再生成对应的npy文件，不然会报错，模型转化的配置文件根据不同模型之间需要做不同的修改。实际上工具链对模型转化提供了很高的自由度，只是说在处理模型后处理时不同。

5. 得到板端运行的hbm模型之后，就是模型部署，BPU对模型的部署前置流程基本相同，只是在处理内存对齐和模型输出后处理上有所不同，这就给了很大的模型部署的空间，测试完成之后要将其封装成一个模块。

#### 源代码：

```c++
#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"

#define CHECK_RET(ret, msg) \
    if ((ret) != 0) { \
        std::cerr << "[ERROR] " << msg << " | Code: " << (ret) << std::endl; \
        return -1; \
    }

struct Point { float x, y, score; };

int main() {
    // 1. 设置路径
    const char* model_file = "/home/sunrise/Desktop/test_mmpose/Hrnet.hbm"; 
    const char* image_file = "/home/sunrise/Desktop/test_mmpose/pose.png";

    // 2. 初始化模型
    hbDNNPackedHandle_t packed_dnn_handle = nullptr;
    CHECK_RET(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file, 1), "Init DNN");

    const char **model_name_list = nullptr;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    
    hbDNNHandle_t dnn_handle = nullptr;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    // 3. 准备输入 Tensor
    int input_count = 0;
    hbDNNGetInputCount(&input_count, dnn_handle);
    std::vector<hbDNNTensor> inputs(input_count);

    for (int i = 0; i < input_count; i++) {
        std::memset(&inputs[i], 0, sizeof(hbDNNTensor));
        hbDNNGetInputTensorProperties(&inputs[i].properties, dnn_handle, i);

        uint32_t mem_size = 0;
        if (i == 0) { // input_y: 256x192
            mem_size = 49152;
            // 强制手动填充 Stride 结构体
            inputs[i].properties.stride[0] = 49152; // Total size
            inputs[i].properties.stride[1] = 192;   // Row stride
            inputs[i].properties.stride[2] = 1;     // Pixel stride
            inputs[i].properties.stride[3] = 1;
        } else { // input_uv: 128x96x2
            mem_size = 24576;
            inputs[i].properties.stride[0] = 24576;
            inputs[i].properties.stride[1] = 192;   // UV行步长也是 192
            inputs[i].properties.stride[2] = 2;     // UV交错
            inputs[i].properties.stride[3] = 1;
        }

        CHECK_RET(hbUCPMallocCached(&inputs[i].sysMem, mem_size, 0), "Malloc Input");
        inputs[i].sysMem.memSize = mem_size;
    }

    // 4. 准备输出 Tensor (手动填充 Stride)
    int output_count = 0;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> outputs(output_count);
    for (int i = 0; i < output_count; i++) {
        std::memset(&outputs[i], 0, sizeof(hbDNNTensor));
        hbDNNGetOutputTensorProperties(&outputs[i].properties, dnn_handle, i);

        // HRNet 输出是 [1, 17, 64, 48] FLOAT32
        uint32_t channel_size = 64 * 48 * sizeof(float); // 12288
        uint32_t total_out_size = 17 * channel_size;    // 208896

        outputs[i].properties.stride[0] = channel_size;       // Channel stride
        outputs[i].properties.stride[1] = 48 * sizeof(float); // Row stride (192 bytes)
        outputs[i].properties.stride[2] = sizeof(float);      // 4
        outputs[i].properties.stride[3] = sizeof(float);      // 4

        CHECK_RET(hbUCPMallocCached(&outputs[i].sysMem, total_out_size, 0), "Malloc Output");
        outputs[i].sysMem.memSize = total_out_size;
    }

    // 5. 图像预处理 (192x256 NV12)
    cv::Mat ori_img = cv::imread(image_file);
    if (ori_img.empty()) { std::cerr << "Image load failed!" << std::endl; return -1; }

    cv::Mat resized, yuv420p;
    cv::resize(ori_img, resized, cv::Size(192, 256));
    cv::cvtColor(resized, yuv420p, cv::COLOR_BGR2YUV_I420);

    // 拷贝 Y
    std::memcpy(inputs[0].sysMem.virAddr, yuv420p.data, 49152);

    // 拷贝 UV (I420 转 NV12 交错格式)
    uint8_t* u_src = yuv420p.data + 49152;
    uint8_t* v_src = u_src + 12288;
    uint8_t* uv_dest = (uint8_t*)inputs[1].sysMem.virAddr;
    for (int i = 0; i < 12288; i++) {
        uv_dest[i*2] = u_src[i];
        uv_dest[i*2+1] = v_src[i];
    }

    // 刷新 CPU 缓存到 BPU 内存
    for (int i = 0; i < input_count; i++) hbUCPMemFlush(&inputs[i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    
    // // --- 验证逻辑：存下 BPU 真正看到的图像 ---
    // {
    //     // 1. 创建一个容器存放导出的 YUV 数据 (256 * 192 * 1.5)
    //     cv::Mat check_yuv(256 + 128, 192, CV_8UC1); 
    //     uint8_t* y_ptr = (uint8_t*)inputs[0].sysMem.virAddr;
    //     uint8_t* uv_ptr = (uint8_t*)inputs[1].sysMem.virAddr;
    //     int y_stride = 192;  // 之前我们手动设定的 stride
    //     int uv_stride = 192;

    //     // 拷贝 Y 区域
    //     for (int r = 0; r < 256; r++) {
    //         std::memcpy(check_yuv.data + r * 192, y_ptr + r * y_stride, 192);
    //     }
    //     // 拷贝 UV 区域
    //     for (int r = 0; r < 128; r++) {
    //         std::memcpy(check_yuv.data + (256 + r) * 192, uv_ptr + r * uv_stride, 192);
    //     }

    //     // 2. 转换为 BGR 并保存
    //     cv::Mat bpu_view_bgr;
    //     cv::cvtColor(check_yuv, bpu_view_bgr, cv::COLOR_YUV2BGR_NV12);
    //     cv::imwrite("bpu_input_view.png", bpu_view_bgr);
    //     std::cout << "Debug image saved: bpu_input_view.png. Check if it is distorted!" << std::endl;
    // }

    // 6. 执行推理
    hbUCPTaskHandle_t task_handle = nullptr;
    CHECK_RET(hbDNNInferV2(&task_handle, outputs.data(), inputs.data(), dnn_handle), "Inference");
    
    hbUCPSchedParam sched; HB_UCP_INITIALIZE_SCHED_PARAM(&sched);
    hbUCPSubmitTask(task_handle, &sched);
    hbUCPWaitTaskDone(task_handle, 0);

    // 7. 后处理 (直接寻找最大值，不使用 Heatmap)
    hbUCPMemFlush(&outputs[0].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    float* raw_ptr = (float*)outputs[0].sysMem.virAddr;
    
    int out_c = 17, out_h = 64, out_w = 48;
    std::vector<Point> pts(out_c);
    for (int c = 0; c < out_c; c++) {
        float max_val = -100.0f;
        int max_x = 0, max_y = 0;
        // 使用手动定义的 Stride 进行寻址
        float* ch_ptr = raw_ptr + c * (out_h * out_w);
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float val = ch_ptr[h * out_w + w];
                if (val > max_val) {
                    max_val = val;
                    max_x = w;
                    max_y = h;
                }
            }
        }
        // 坐标映射：48x64 映射回原图尺寸
        pts[c].x = max_x * (ori_img.cols / 48.0f);
        pts[c].y = max_y * (ori_img.rows / 64.0f);
        pts[c].score = max_val;
    }

    // 8. 绘制并保存结果
    for (int i = 0; i < out_c; i++) {
        // 置信度阈值根据量化模型实际表现调整，通常 0.0 以上即为有效
        if (pts[i].score > 0.0f) {
            cv::circle(ori_img, cv::Point(pts[i].x, pts[i].y), 4, cv::Scalar(0, 255, 0), -1);
            cv::putText(ori_img, std::to_string(i), cv::Point(pts[i].x, pts[i].y), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        }
    }

    cv::imwrite("final_result.png", ori_img);
    std::cout << "Success! Process finished. Check final_result.png" << std::endl;

    // 9. 释放资源
    hbUCPReleaseTask(task_handle);
    for (auto &in : inputs) hbUCPFree(&in.sysMem);
    for (auto &out : outputs) hbUCPFree(&out.sysMem);
    hbDNNRelease(packed_dnn_handle);

    return 0;
}
```

