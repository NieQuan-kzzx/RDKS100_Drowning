#include "WaterIngress.h"
#include "PlogInitializer.h"

WaterIngress::WaterIngress()
    : m_patchcore(std::make_unique<Inf::Patchcore>())
    , m_yolo_seg(std::make_unique<Inf::YoloSeg>()) {
    PlogInitializer::getInstance().init(plog::verbose);
}

WaterIngress::~WaterIngress() = default;

bool WaterIngress::initPatchcore(const std::string& model_path) {
    m_patchcore_initialized = m_patchcore->init(model_path);
    return m_patchcore_initialized;
}

bool WaterIngress::initSeg(const std::string& model_path) {
    m_seg_initialized = m_yolo_seg->init(model_path);
    return m_seg_initialized;
}

void WaterIngress::setSegLabels(const std::vector<std::string>& labels) {
    m_yolo_seg->setLabels(labels);
}

void WaterIngress::process(cv::Mat& frame, const std::vector<Inf::Detection>& results) {
    if (frame.empty()) return;

    // 1. 运行 Patchcore 异常检测（仅在模型已初始化时执行）
    float patchcore_score = 0.0f;
    std::vector<Inf::Detection> patchcore_results;
    if (m_patchcore_initialized) {
        patchcore_results = m_patchcore->run(frame);
        if (!patchcore_results.empty()) {
            patchcore_score = patchcore_results[0].score;
        }
    }

    // 2. 运行实例分割检测 "water"（仅在模型已初始化时执行）
    bool has_water = false;
    std::vector<Inf::Detection> seg_results;
    if (m_seg_initialized) {
        seg_results = m_yolo_seg->run(frame);
        for (const auto& det : seg_results) {
            if (det.class_id == m_water_class_id) {
                has_water = true;
            }
        }
    }

    // 3. 绘制 seg 遮罩
    if (m_seg_initialized && !seg_results.empty()) {
        m_yolo_seg->draw(frame, seg_results);
    }

    // 4. 绘制 Patchcore 热力图（置于最上层）
    if (m_patchcore_initialized && !patchcore_results.empty()) {
        m_patchcore->draw(frame, patchcore_results);
    }

    // 5. 双重判定：Patchcore 异常分超阈值 且 检测到 water
    bool water_ingress = (patchcore_score > m_patchcore_threshold) && has_water;

    // 6. 绘制报警
    if (water_ingress) {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Rect(0, 0, frame.cols, 80), cv::Scalar(0, 0, 255), -1);
        cv::addWeighted(overlay, 0.4, frame, 0.6, 0, frame);

        std::string warn_text = "ALARM: WATER INGRESS DROWNING!";
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(warn_text, cv::FONT_HERSHEY_DUPLEX, 1.5, 3, &baseline);
        cv::Point text_org((frame.cols - text_size.width) / 2, 55);

        cv::putText(frame, warn_text, text_org + cv::Point(2, 2),
                    cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 0), 3);
        cv::putText(frame, warn_text, text_org,
                    cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
    }

    // 6. 显示 Patchcore 分数和 water 检测状态
    std::string info = "Patchcore Score: " + std::to_string(patchcore_score).substr(0, 6)
                     + " | Water: " + (has_water ? "YES" : "NO")
                     + " | Alarm: " + (water_ingress ? "TRIGGERED" : "NORMAL");
    cv::putText(frame, info, cv::Point(30, frame.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
}
