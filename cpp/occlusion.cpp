#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#define MINIMUM(a, b) (((a) < (b)) ? (a) : (b))
#define MAXIMUM(a, b) (((a) > (b)) ? (a) : (b))

double complexityDetector(cv::Mat &src, cv::Size resize_set, cv::Size superpixel_set)
{
    cv::Mat resized_mat, v_channel;
    std::vector<cv::Mat> hsv_channels;
    cv::resize(src, resized_mat, resize_set);
    cv::cvtColor(resized_mat, resized_mat, cv::COLOR_BGR2HSV);
    cv::split(resized_mat, hsv_channels);
    cv::GaussianBlur(hsv_channels[2], v_channel, cv::Size{5, 5}, 0);

    cv::Mat grad_x, grad_y, grad;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Sobel(v_channel, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::Sobel(v_channel, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    cv::threshold(grad, grad, 50, 1, cv::THRESH_BINARY);

    float w_num_pixel = float(resize_set.width) / float(superpixel_set.width);
    float h_num_pixel = float(resize_set.height) / float(superpixel_set.height);
    cv::Mat score_mat(superpixel_set.height, superpixel_set.width, CV_32F, cv::Scalar(0));
    for (int w = 0; w < superpixel_set.width; w++)
    {
        for (int h = 0; h < superpixel_set.height; h++)
        {
            cv::Rect rect(int(w_num_pixel * w), int(h_num_pixel * h), int(w_num_pixel), int(h_num_pixel));
            cv::Scalar score_tmp = cv::mean(grad(rect));
            float *rowptr = score_mat.ptr<float>(h);
            rowptr[w] = score_tmp[0];
        }
    }
    // std::cout << score_mat << std::endl;
    cv::Mat means, stddev;
    cv::meanStdDev(score_mat, means, stddev);

    double divider = means.at<double>(0) * 2;
    double score;
    score = (divider > 0) ? stddev.at<double>(0) / divider : 1.0;

    return MINIMUM(score, 1.0);
}

int main()
{
    cv::Mat test_img = cv::imread("/home/chenzhou/Documents/PythonRepo/AuxiliaryRepo/CamOcclusionDetect/data/Occluded/648.jpg");
    cv::Size resize_set = cv::Size{160, 120};
    cv::Size superpixel_set = cv::Size{12, 9};
    clock_t tic = clock();
    double score;
    int total_iter = 10000;
    for (int iter = 0; iter < total_iter; iter++)
    {
        score = complexityDetector(test_img, resize_set, superpixel_set);
    }
    clock_t toc = clock();
    std::cout << "function consuming: "
              << (double)(toc - tic) * 1000.0 / (CLOCKS_PER_SEC * total_iter)
              << "ms Score: " << score << std::endl;
}