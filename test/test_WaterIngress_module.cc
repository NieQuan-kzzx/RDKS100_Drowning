#include <opencv2/opencv.hpp>
#include "gflags/gflags.h"
#include "PlogInitializer.h"
#include "WaterIngress.h"
#include "common_utils.hpp"

DEFINE_string(patchcore_model, "/home/sunrise/Desktop/RDKS100_Drowning/models/patchcore.hbm", "Patchcore模型路径");
DEFINE_string(seg_model, "/home/sunrise/Desktop/RDKS100_Drowning/models/Water_seg.hbm", "实例分割模型路径");
DEFINE_string(input, "/home/sunrise/Desktop/RDKS100_Drowning/tem/ori_water.png", "输入图像路径");
DEFINE_double(anomaly_threshold, 50.0, "Patchcore异常分数阈值");
DEFINE_string(label_file, "/home/sunrise/Desktop/RDKS100_Drowning/tem/classes_water_seg.names", "分割模型标签文件");
DEFINE_int32(water_class_id, 0, "Water类别ID");
DEFINE_string(output_dir, ".", "输出目录");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    PlogInitializer::getInstance().init(plog::verbose);

    int passed = 0, failed = 0;
    auto test = [&](const char* name, bool ok) {
        PLOGI << (ok ? "[PASS]" : "[FAIL]") << " " << name;
        if (ok) passed++; else failed++;
    };

    // 1) construction
    { WaterIngress wi;
      wi.setPatchcoreThreshold(42.5f);
      wi.setWaterClassId(1);
      wi.setSegLabels({"bg", "water", "pool"});
      test("construction and setters", true); }

    // 2) empty frame
    { WaterIngress wi;
      cv::Mat empty;
      std::vector<Inf::Detection> empty_res;
      wi.process(empty, empty_res);
      test("empty frame handling", true); }

    // 3) valid frame + empty results (no models)
    { WaterIngress wi;
      wi.setPatchcoreThreshold(50.0f);
      cv::Mat frame = cv::imread(FLAGS_input);
      if (frame.empty()) { test("valid frame + empty results", false); }
      else { wi.process(frame, {}); cv::imwrite(FLAGS_output_dir + "/test3_empty_results.jpg", frame);
             test("valid frame + empty results", true); } }

    // 4) full pipeline with actual models
    { WaterIngress wi;
      wi.setPatchcoreThreshold(FLAGS_anomaly_threshold);
      wi.setWaterClassId(FLAGS_water_class_id);
      if (!wi.initPatchcore(FLAGS_patchcore_model)) {
          test("full pipeline (patchcore model unavailable)", true);
      } else if (!wi.initSeg(FLAGS_seg_model)) {
          test("full pipeline (seg model unavailable)", true);
      } else {
          std::vector<std::string> labels;
          try { labels = load_linewise_labels(FLAGS_label_file); } catch (...) {}
          if (!labels.empty()) {
              wi.setSegLabels(labels);
              for (size_t i = 0; i < labels.size(); ++i)
                  if (labels[i] == "water" || labels[i] == "Water") { wi.setWaterClassId(i); break; }
          }
          cv::Mat frame = cv::imread(FLAGS_input);
          if (frame.empty()) { test("full pipeline", false); }
          else {
              wi.process(frame, {});
              cv::imwrite(FLAGS_output_dir + "/test_full_pipeline.jpg", frame);
              test("full pipeline", true);
          }
      } }

    PLOGI << "=== " << passed << " passed, " << failed << " failed ===";
    return failed > 0 ? 1 : 0;
}
