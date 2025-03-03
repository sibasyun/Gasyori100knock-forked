// Discrete cosine transform and PSNR
// 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

constexpr std::array<float, 3> gray_scale_coef = {0.0722, 0.7152, 0.2126};

constexpr double pi = 3.141593;


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

double w(int x, int y, int u, int v, int T){
  double cu = (u == 0) ? 1.0 / sqrt(2) : 1.0;
  double cv = (v == 0) ? 1.0 / sqrt(2) : 1.0;
  double theta = pi / (2.0 * (double) T);

  return (2.0 * cu * cv / T) * cos((2.0 * (double) x + 1.0) * (double) u * theta) * cos((2.0 * (double) y + 1.0) * (double) v * theta);
}

double MSE(cv::Mat img1, cv::Mat img2){
  int width = img1.cols;
  int height = img1.rows;
  assert (width == img2.cols);
  assert (height == img2.rows);

  double sum = 0;
  for (int c = 0; c < 3; c++){
    for (int x = 0; x < height; x++){
      for (int y = 0; y < width; y++){
        sum += pow(img1.at<cv::Vec3b>(x, y)[c] - img2.at<cv::Vec3b>(x, y)[c] , 2);
      }
    }
  }
  sum /= (double)(width * height * 3);
  return sum;
}

double PSNR(double MSE){
  return 10. * log10(255.*255./MSE);
}


// T x T　の領域に分割したDCT
std::vector<std::vector<std::vector<double>>> DCT(cv::Mat img, int T){
  int width = img.cols;
  int height = img.rows;

  std::vector<std::vector<std::vector<double>>> F(height, std::vector<std::vector<double>>(width, std::vector<double>(3, 0.0)));
  for (int c = 0; c < 3; c++){
    for (int x0 = 0; x0 < height; x0 += T){
      for (int y0 = 0; y0 < width; y0 += T){
        for (int u = 0; u < T; u++){
          for (int v = 0; v < T; v++){
            double sum = 0.0;
            for (int x = 0; x < T; x++){
              for (int y = 0; y < T; y++){
                sum += (double) img.at<cv::Vec3b>(x0 + x, y0 + y)[c] * w(x, y, u, v, T);
              }
            }
            F[x0 + u][y0 + v][c] = sum;
          }
        }
      }
    }
  }
  return F;
}

// inverse DCT
cv::Mat IDCT(std::vector<std::vector<std::vector<double>>> F, int T, int K){
  int height = F.size();
  int width = F[0].size();

  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  for (int c = 0; c < 3; c++){
    for (int x0 = 0; x0 < height; x0 += T){
      for (int y0 = 0; y0 < width; y0 += T){
        for (int x = 0; x < T; x++){
          for (int y = 0; y < T; y++){
            double sum = 0.0;
            for (int u = 0; u < K; u++){
              for (int v = 0; v < K; v++){
                sum += w(x, y, u, v, T) * F[x0 + u][y0 + v][c];
              }
            }
            sum = std::min(std::max(sum, 0.0), 255.0);
            out.at<cv::Vec3b>(x0 + x, y0 + y)[c] = (uchar) sum;
          }
        }
      }
    }
  }
  return out;
}


int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  auto dct = DCT(img, 8);
  // auto lpf = LPF(dft, 0.5f);
  // auto hpf = HPF(dft, 0.1f);
  auto out = IDCT(dct, 8, 4);

  // PSNRの計算

  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;
  //cv::imwrite("out.jpg", out);
  double mse = MSE(img, out);
  std::cout << PSNR(mse) << std::endl;
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}