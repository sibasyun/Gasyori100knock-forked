// template matching ZNCC
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

std::vector<double> calc_channels_mean(cv::Mat I){
  int H = I.rows;
  int W = I.cols;
  int channels = I.channels();

  std::vector<double> mean_I(channels, 0);

  for (int i = 0; i < H; i++){
    for (int j = 0; j < W; j++){
      for (int c = 0; c < channels; c++){
        mean_I[c] += (double) I.at<cv::Vec3b>(i, j)[c];
      }
    }
  }
  for (int c = 0; c < channels; c++) mean_I[c] /= double(H * W);

  return mean_I;
}

std::pair<int, int> template_matching_ZNCC(cv::Mat I, cv::Mat T){
  int H = I.rows;
  int W = I.cols;
  int channels = I.channels();
  int h = T.rows;
  int w = T.cols;
  assert (channels == 3 || channels == 1);
  assert (channels == T.channels());

  // 要素ごとの平均を計算する
  std::vector<double> mean_T = calc_channels_mean(T);

  int xm = -1, ym = -1;
  double max_zncc = -1;

  cv::Mat temp_I = (channels == 3 ? cv::Mat::zeros(h, w, CV_8UC3) : cv::Mat::zeros(h, w, CV_8UC1));

  for (int i = 0; i < H; i++){
    for (int j = 0; j < W; j++){
      double temp_value = 0;
      double denom_I = 0;
      double denom_T = 0;
      bool valid = true;

      for (int x = 0; x < h; x++){
        for (int y = 0; y < w; y++){
          if (i + x >= H || j + y >= W){
            valid = false;
            break;
          }
          for (int c = 0; c < channels; c++){
            if (channels == 1){
              temp_I.at<uchar>(x, y) = I.at<uchar>(i+x, j+w);
            } else {
              temp_I.at<cv::Vec3b>(x, y)[c] = I.at<cv::Vec3b>(i+x, j+w)[c];
            }
          }
        }
      }
      if (!valid) break;

      auto mean_I = calc_channels_mean(temp_I);

      for (int x = 0; x < h; x++){
        for (int y = 0; y < w; y++){
          
          for (int c = 0; c < channels; c++){
            double vI = I.at<cv::Vec3b>(i + x, j + y)[c];
            double vT = T.at<cv::Vec3b>(x, y)[c];
            denom_I += (vI - mean_I[c]) * (vI - mean_I[c]);
            denom_T += (vT - mean_T[c]) * (vT - mean_T[c]);
            temp_value += (vI - mean_I[c]) * (vT - mean_T[c]);
          }
        }
        
      }
      double temp_zncc = (double) temp_value / sqrt(denom_I) / sqrt(denom_T);
      if (valid && temp_zncc > max_zncc){
        xm = i; 
        ym = j;
        max_zncc = temp_zncc;
      }
    }
  }
  // std::cout << max_zncc << std::endl;
  return {xm, ym};
}


int main(int argc, const char* argv[]){
  timeMeasure tm = timeMeasure();
  cv::Mat I = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  cv::Mat T = cv::imread("imori_part.jpg", cv::IMREAD_COLOR);
  int h = T.rows;
  int w = T.cols;
  
  std::pair<int, int> coord = template_matching_ZNCC(I, T);
  cv::rectangle(I, cv::Point(coord.second, coord.first), cv::Point(coord.second + w, coord.first + h), cv::Scalar(0,0,255));
  tm.elapsed_time();

  cv::imwrite("out_57.jpg", I);
  cv::imshow("answer", I);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}