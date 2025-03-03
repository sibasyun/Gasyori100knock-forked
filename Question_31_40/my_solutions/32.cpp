// skew transform
// 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

constexpr std::array<float, 3> gray_scale_coef = {0.0722, 0.7152, 0.2126};

constexpr float pi = 3.141593f;


cv::Mat cvt_gray_scale(cv::Mat img){
  // get height and width
  int width = img.cols;
  int height = img.rows;
  int channels = img.channels();
  assert (channels == 3);

  // output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  // convert to gray scale
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      for (int c = 0; c < channels; c++){
      out.at<uchar>(y, x) += (float)img.at<cv::Vec3b>(y, x)[c] * gray_scale_coef[c];
      }
    }
  }
  return out;
}

cv::Mat DFT(cv::Mat img){
  int width = img.cols;
  int height = img.rows;
  int channels = img.channels();
  auto img_gray = cvt_gray_scale(img);
  
  std::vector<std::vector<std::complex<float>>> G(height, std::vector<std::complex<float>>(width, {0.0f, 0.0f}));
  
  auto g = [&](int l, int k){
    std::complex<float> ret{0.0f, 0.0f};
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        float arg = -2. * pi * (float(l * y / height + k * x / width));
        float val = (float) img_gray.at<uchar>(y, x) / (float) (height * width);
        std::complex<float> temp{cos(arg) * val , sin(arg) * val};
        ret += temp;
      }
    }
    return ret;
  };
  
  for (int l = 0; l < height; l++){
    for (int k = 0; k < width; k++){
      G[l][k] = g(l, k);
    }
  }

  // スケーリング用のvector
  std::vector<std::vector<float>> G_abs(height, std::vector<float>(width, 0.0f));
  float mx = 0.0f;
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      G_abs[y][x] = std::norm(G[y][x]);
      mx = std::max(mx, G_abs[y][x]);
    }
  }
  mx += 0.001; // オーバーフロー対策
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      out.at<uchar>(y, x) = G_abs[y][x] / mx;
    }
  }
  return out;
}

int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("imori.jpg", cv::IMREAD_COLOR);

  cv::Mat out = DFT(img);
  
  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;
  //cv::imwrite("out.jpg", out);

  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}