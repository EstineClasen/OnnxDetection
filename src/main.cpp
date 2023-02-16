#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/cmdline.h"
#include "../include/detection_utils.h"
#include "../include/detector.h"


int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

    bool isGPU = false;
    const std::string classNamesPath = "C:\\Users\\estin\\OneDrive\\Documents\\Visual Studio 2019\\TestYoloToOnnxOD\\TestYoloToOnnxOD\\models\\coco.names";
    const std::vector<std::string> classNamesList = detection_utils::loadClassNames(classNamesPath);
    const std::string imagePath = "C:\\Users\\estin\\OneDrive\\Documents\\Visual Studio 2019\\TestYoloToOnnxOD\\TestYoloToOnnxOD\\images\\bus.jpg";
    const std::string modelPath = "C:\\Users\\estin\\OneDrive\\Documents\\Visual Studio 2019\\TestYoloToOnnxOD\\TestYoloToOnnxOD\\models\\yolov5s.onnx";

    if (classNamesList.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    ObjectDetector detector {nullptr};
    cv::Mat image;
    std::vector<Detection> result;

    try
    {
        detector = ObjectDetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;

        image = cv::imread(imagePath);
        result = detector.detect(image, confThreshold, iouThreshold);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    detection_utils::visualizeDetection(image, result, classNamesList);

    cv::imshow("result", image);
    // cv::imwrite("result.jpg", image);
    cv::waitKey(0);

    return 0;
}
