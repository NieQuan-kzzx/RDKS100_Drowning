#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>

class ImageSaver
{
public:
    // 构造函数，传入保存路径
    ImageSaver(const std::string &folderPath);
    // 析构函数，保存所有图片到文件夹
    ~ImageSaver();

    // 向容器中添加一个cv::Mat和图片名
    void addImage(const cv::Mat &mat, const std::string &name);
    void flush(void);

private:
    std::vector<cv::Mat> mats;      // 存储所有cv::Mat
    std::vector<std::string> names; // 存储对应的图片名称
    std::string folderPath;         // 保存图片的文件夹路径
    std::string timeFolder;         // 以时间命名的子文件夹
};
