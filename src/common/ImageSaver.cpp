#include "ImageSaver.h"

namespace fs = std::filesystem;

ImageSaver::ImageSaver(const std::string &folderPath)
{
    // 创建文件夹
    this->folderPath = folderPath;
    if (!fs::exists(folderPath))
        fs::create_directory(folderPath);
    
}

ImageSaver::~ImageSaver()
{
    this->flush();
}

void ImageSaver::addImage(const cv::Mat &mat, const std::string &name)
{
    mats.push_back(mat.clone());
    names.push_back(name);
}

void ImageSaver::flush(void)
{
    if (this->mats.empty())
        return;

    if (this->timeFolder.empty())
    {
        // 获取当前时间并格式化为文件夹名称
        auto now = std::chrono::system_clock::now();
        auto nowTimeT = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&nowTimeT);

        // 格式化时间字符串为文件夹名（例如：2024-12-21_14-30-00）
        std::ostringstream folderNameStream;
        folderNameStream << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        std::string timeFolder = folderPath + "/" + folderNameStream.str();
        this->timeFolder = timeFolder;
    }

    // 创建以当前时间命名的文件夹
    if(!fs::exists(timeFolder))
        fs::create_directory(timeFolder);


    for (size_t i = 0; i < mats.size(); ++i)
    {
        std::string filePath = timeFolder + "/" + names[i] + ".jpg";
        if (!cv::imwrite(filePath, mats[i]))
        {
            std::cerr << "Error saving image: " << filePath << std::endl;
        }
        else
        {
            std::cout << "Image saved: " << filePath << std::endl;
        }
    }
    mats.clear();
    names.clear();
}