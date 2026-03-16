#pragma once

#include "plog/Log.h"
#include "plog/Init.h"
#include "plog/Appenders/ColorConsoleAppender.h"
#include "plog/Formatters/TxtFormatter.h"
#include <mutex>
#include <atomic>

// 自定义日志输出
// namespace plog
// {
//     class MyFormatter
//     {
//     public:
//         static util::String header() { return util::String(); }
//         static util::String format(const Record& record)
//         {
//             util::nostringstream ss;
//             // 格式示例：[级别] [函数名] 消息内容
//             ss << "[" << severityToString(record.getSeverity()) << "] ";
//             ss << "[" << record.getFunc() << "] ";
//             ss << record.getMessage() << "\n";
//             return ss.str();
//         }
//     };
// }

/**
 * @brief 全局 plog 初始化管理器
 * @details 确保 plog 仅初始化一次，避免多库间的初始化冲突
 * 
 * 使用示例（主程序中）：
 *   PlogInitializer::getInstance().init(plog::verbose);
 * 
 * 库代码中无需初始化，直接使用 PLOG* 宏
 */
class PlogInitializer
{
public:
    static PlogInitializer& getInstance()
    {
        static PlogInitializer instance;
        return instance;
    }

    /**
     * @brief 初始化 plog
     * @param severity 日志级别（默认为 plog::info）
     */
    void init(plog::Severity severity = plog::info)
    {
        std::lock_guard<std::mutex> lock(init_mutex);
        
        if (is_initialized.load())
        {
            return; // 已初始化，直接返回
        }

        static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
        plog::init(severity, &consoleAppender);
        is_initialized.store(true);
    }

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const
    {
        return is_initialized.load();
    }

    // 禁用复制和移动
    PlogInitializer(const PlogInitializer&) = delete;
    PlogInitializer& operator=(const PlogInitializer&) = delete;
    PlogInitializer(PlogInitializer&&) = delete;
    PlogInitializer& operator=(PlogInitializer&&) = delete;

private:
    PlogInitializer() = default;
    
    std::mutex init_mutex;
    std::atomic<bool> is_initialized{false};
};

/**
 * @brief 在库代码中使用，确保 plog 已初始化（如果主程序未初始化则自动初始化）
 * 这个宏应该在库的首个使用 PLOG 的 .cpp 文件中调用一次
 */
#define ENSURE_PLOG_INITIALIZED() \
    do { \
        if (!PlogInitializer::getInstance().isInitialized()) { \
            PlogInitializer::getInstance().init(plog::info); \
        } \
    } while(0)

