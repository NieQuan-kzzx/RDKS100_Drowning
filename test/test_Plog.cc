// #include <plog/Log.h> // Step1: include the headers
// #include "plog/Initializers/RollingFileInitializer.h"

// int main()
// {
//     plog::init(plog::debug, "Hello.txt"); // Step2: initialize the logger

//     // Step3: write log messages using a special macro
//     // There are several log macros, use the macro you liked the most

//     PLOGD << "Hello log!"; // short macro
//     PLOG_DEBUG << "Hello log!"; // long macro
//     PLOG(plog::debug) << "Hello log!"; // function-style macro
    
//     // Also you can use LOG_XXX macro but it may clash with other logging libraries
//     LOGD << "Hello log!"; // short macro
//     LOG_DEBUG << "Hello log!"; // long macro
//     LOG(plog::debug) << "Hello log!"; // function-style macro

//     return 0;
// }


#include "PlogInitializer.h"
#include <thread>
#include <vector>

void test_thread_safety() {
    // 模拟多线程同时尝试初始化
    ENSURE_PLOG_INITIALIZED();
    PLOGI << "Log from thread: " << std::this_thread::get_id();
}

int main() {
    // --- 情况 1: 正常初始化 ---
    // 我们手动设置级别为 VERBOSE，以便看到所有级别的日志
    PlogInitializer::getInstance().init(plog::verbose);
    
    PLOGV << "This is a VERBOSE message (Level 1)";
    PLOGD << "This is a DEBUG message (Level 2)";
    PLOGI << "This is an INFO message (Level 3)";

    // --- 情况 2: 重复初始化测试 ---
    // 尝试再次以不同级别初始化，应该被单例的原子锁挡住，不会生效
    PlogInitializer::getInstance().init(plog::error);
    PLOGI << "If you see this, the second init(error) didn't overwrite the first one (Good!).";

    // --- 情况 3: 宏安全性测试 ---
    // 在代码深处使用宏，确保即便不确定是否初始化了，也能安全使用
    ENSURE_PLOG_INITIALIZED();
    PLOGW << "This warning log is safe!";

    // --- 情况 4: 多线程压力测试 ---
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back(test_thread_safety);
    }

    for (auto& t : threads) {
        t.join();
    }

    PLOGI << "Test finished successfully!";
    return 0;
}