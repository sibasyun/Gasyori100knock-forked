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

// low pass filter
std::vector<std::vector<std::complex<float>>> LPF(std::vector<std::vector<std::complex<float>>> G, float pass_ratio){
  int height = G.size();
  int width = G[0].size();
  
  int r = height/2;
  
  int filter_d = (int)(float(r) * pass_ratio);
  for (int y = 0; y < G.size() / 2; y++){
    for (int x = 0; x < G[0].size() / 2; x++){
      if (float(y * y + x * x) >= filter_d * filter_d){
        G[y][x] = 0;
        G[y][width-x-1] = 0;
        G[height-y-1][x] = 0;
        G[height-y-1][width-x-1] = 0;
      }
    }
  }
  return G;
}

std::vector<std::vector<std::complex<float>>> DFT(cv::Mat img){
  int width = img.cols;
  int height = img.rows;
  int channels = img.channels();
  auto img_gray = cvt_gray_scale(img);
  
  std::vector<std::vector<std::complex<float>>> G(height, std::vector<std::complex<float>>(width, {0.0f, 0.0f}));
  
  auto g = [&](int l, int k){
    std::complex<float> ret{0.0f, 0.0f};
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        float arg = -2. * pi * ((float)(l * y) / (float) height + float(k * x) / (float) width);
        float val = (float) img_gray.at<uchar>(y, x);
        std::complex<float> temp{cos(arg) * val , sin(arg) * val};
        ret += temp;
      }
    }
    return ret / (float) sqrt(height * width);
  };
  
  for (int l = 0; l < height; l++){
    for (int k = 0; k < width; k++){
      G[l][k] = g(l, k);
    }
  }
  return G;
}

cv::Mat IDFT(std::vector<std::vector<std::complex<float>>> G){
  int height = G.size();
  int width = G[0].size();

  auto f = [&](int l, int k){
    std::complex<float> ret{0.0f, 0.0f};
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        float arg = 2. * pi * ((float)(l * y) / (float) height + float(k * x) / (float) width);
        std::complex<float> temp{cos(arg) , sin(arg)};
        ret += temp * G[y][x];
      }
    }
    return ret / (float) sqrt(height * width);
  };

  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  for (int l = 0; l < height; l++){
    for (int k = 0; k < width; k++){
      out.at<uchar>(l, k) = std::abs(f(l, k));
    }
  }
  return out;
}

int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("imori.jpg", cv::IMREAD_COLOR);

  auto dft = DFT(img);
  auto lpf = LPF(dft, 0.5f);
  auto out = IDFT(lpf);

  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;
  //cv::imwrite("out.jpg", out);

  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}