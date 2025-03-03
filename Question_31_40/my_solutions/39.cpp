// Discrete cosine transform and PSNR
// 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

constexpr std::array<float, 3> gray_scale_coef = {0.0722, 0.7152, 0.2126};

constexpr double pi = 3.141593;

std::vector<std::vector<std::vector<float>>> cvt_BGR2YCbCr(cv::Mat img){
  // get height and width
  int width = img.cols;
  int height = img.rows;
  int channels = img.channels();
  assert (channels == 3);

  // output
  std::vector<std::vector<std::vector<float>>> out(height, std::vector<std::vector<float>>(width, std::vector<float>(3, 0.0)));
  // convert to YCbCr
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      out[x][y][0] = (uchar) (0.299 * (float)img.at<cv::Vec3b>(x, y)[2] + 0.587 * (float)img.at<cv::Vec3b>(x, y)[1] + 0.114 * (float)img.at<cv::Vec3b>(x, y)[0]);
      out[x][y][1] = (uchar) (-0.1687 * (float)img.at<cv::Vec3b>(x, y)[2] - 0.3313 * (float)img.at<cv::Vec3b>(x, y)[1] + 0.5 * (float)img.at<cv::Vec3b>(x, y)[0] + 128);
      out[x][y][2] = (uchar) (0.5 * (float)img.at<cv::Vec3b>(x, y)[2] - 0.4187 * (float)img.at<cv::Vec3b>(x, y)[1] - 0.0813 * (float)img.at<cv::Vec3b>(x, y)[0] + 128);
    }
  }
  return out;
}

cv::Mat cvt_YCbCr2BGR(std::vector<std::vector<std::vector<float>>> img){
  // get height and width
  int width = img[0].size();
  int height = img.size();

  // output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  // convert to YCbCr
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      out.at<cv::Vec3b>(x, y)[2] = (uchar) (img[x][y][0] + 1.402 * ((float)img[x][y][2] - 128));
      out.at<cv::Vec3b>(x, y)[1] = (uchar) (img[x][y][0] - 0.3441 * ((float)img[x][y][1] - 128) - 0.7139 * ((float)img[x][y][2] - 128));
      out.at<cv::Vec3b>(x, y)[0] = (uchar) (img[x][y][0] + 1.7718 * ((float)img[x][y][1] - 128));
    }
  }
  return out;
}


int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  int width = img.cols;
  int height = img.rows;
  auto ycbcr = cvt_BGR2YCbCr(img);


  // Yを0.7倍
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      ycbcr[x][y][0] *= 0.7;
    }
  }
  auto out = cvt_YCbCr2BGR(ycbcr);

  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;
  
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}