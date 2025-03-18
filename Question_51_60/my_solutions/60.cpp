#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <chrono>
#include <random>

class timeMeasure{
private:
  std::chrono::high_resolution_clock::time_point start_time_;
public:
  timeMeasure() : start_time_(std::chrono::high_resolution_clock::now()){}
  void elapsed_time() {
    auto end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start_time_).count(); //処理に要した時間をミリ秒に変換
    std::cout << elapsed << std::endl;
  };
};

double clip(double value, double min, double max){
  return fmin(fmax(value, min), max);
}

cv::Mat alpha_blending(cv::Mat I1, double a1, cv::Mat I2, double a2){
  // I1とI2のサイズが合っていることを確認する
  int h1 = I1.rows, w1 = I1.cols;
  int h2 = I2.rows, w2 = I2.cols;
  // std::cout << h1 << ' ' << h2 << ' ' << w1 << ' ' << w2 << std::endl;
  assert (h1 == h2 && w1 == w2);

  cv::Mat out = cv::Mat::zeros(h1, w1, CV_8UC3);
  for (int x = 0; x < h1; x++){
    for (int y = 0; y < w1; y++){
      for (int c = 0; c < 3; c++){
        out.at<cv::Vec3b>(x, y)[c] = (uchar) clip(
          a1 * I1.at<cv::Vec3b>(x, y)[c] + a2 * I2.at<cv::Vec3b>(x, y)[c],
          0, 255
        );
      }
    }
  }
  return out;
}

int main(int argc, const char* argv[]){
  timeMeasure tm = timeMeasure();
  cv::Mat I1 = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  cv::Mat I2 = cv::imread("thorino_resized.jpg", cv::IMREAD_COLOR);
  cv:: Mat out = alpha_blending(I1, 0.6, I2, 0.4);
  
  tm.elapsed_time();

  cv::imwrite("out_60.jpg", out);
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}