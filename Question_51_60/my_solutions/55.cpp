// template matching SAD
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

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

std::pair<int, int> template_matching_SAD(cv::Mat I, cv::Mat T){
  int H = I.rows;
  int W = I.cols;
  int channels = I.channels();
  int h = T.rows;
  int w = T.cols;
  assert (channels == T.channels());

  int xm = 0, ym = 0;
  long long temp_sad = (long long) 1e18;

  for (int i = 0; i < H; i++){
    for (int j = 0; j < W; j++){
      long long temp_value = 0;
      bool valid = true;
      for (int x = 0; x < h; x++){
        for (int y = 0; y < w; y++){
          if (i + x >= H || j + y >= W){
            valid = false;
            break;
          }
          for (int c = 0; c < channels; c++){
            long long v = (long long) I.at<cv::Vec3b>(i + x, j + y)[c] - (long long) T.at<cv::Vec3b>(x, y)[c];
            temp_value += abs(v);
          }
        }
        if (!valid) break;
      }
      if (valid && temp_value < temp_sad){
        xm = i; 
        ym = j;
        temp_sad = temp_value;
      }
    }
  }
  return {xm, ym};
}


int main(int argc, const char* argv[]){
  timeMeasure tm = timeMeasure();
  cv::Mat I = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  cv::Mat T = cv::imread("imori_part.jpg", cv::IMREAD_COLOR);
  int h = T.rows;
  int w = T.cols;
  
  std::pair<int, int> coord = template_matching_SAD(I, T);
  cv::rectangle(I, cv::Point(coord.second, coord.first), cv::Point(coord.second + w, coord.first + h), cv::Scalar(0,0,255));
  tm.elapsed_time();

  cv::imwrite("out_55.jpg", I);
  cv::imshow("answer", I);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}