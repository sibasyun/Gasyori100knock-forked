#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

constexpr std::array<double, 3> gray_scale_coef = {0.0722, 0.7152, 0.2126};

constexpr double pi = 3.141593;

constexpr double Q1[8][8] = {{16, 11, 10, 16, 24, 40, 51, 61},
                            {12, 12, 14, 19, 26, 58, 60, 55},
                            {12, 12, 14, 19, 26, 58, 60, 55},
                            {14, 17, 22, 29, 51, 87, 80, 62},
                            {18, 22, 37, 56, 68, 109, 103, 77},
                            {24, 35, 55, 64, 81, 104, 113, 92},
                            {49, 64, 78, 87, 103, 121, 120, 101},
                            {72, 92, 95, 98, 112, 100, 103, 99}
                          };
                          
constexpr double Q2[8][8] = {{17, 18, 24, 47, 99, 99, 99, 99},
                          {18, 21, 26, 66, 99, 99, 99, 99},
                          {24, 26, 56, 99, 99, 99, 99, 99},
                          {47, 66, 99, 99, 99, 99, 99, 99},
                          {99, 99, 99, 99, 99, 99, 99, 99},
                          {99, 99, 99, 99, 99, 99, 99, 99},
                          {99, 99, 99, 99, 99, 99, 99, 99},
                          {99, 99, 99, 99, 99, 99, 99, 99}
                        };

std::vector<std::vector<std::vector<double>>> cvt_BGR2YCbCr(cv::Mat img){
  // get height and width
  int width = img.cols;
  int height = img.rows;
  int channels = img.channels();
  assert (channels == 3);

  // output
  std::vector<std::vector<std::vector<double>>> out(height, std::vector<std::vector<double>>(width, std::vector<double>(3, 0.0)));
  // convert to YCbCr
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      out[x][y][0] = (uchar) (0.299 * (double)img.at<cv::Vec3b>(x, y)[2] + 0.587 * (double)img.at<cv::Vec3b>(x, y)[1] + 0.114 * (double)img.at<cv::Vec3b>(x, y)[0]);
      out[x][y][1] = (uchar) (-0.1687 * (double)img.at<cv::Vec3b>(x, y)[2] - 0.3313 * (double)img.at<cv::Vec3b>(x, y)[1] + 0.5 * (double)img.at<cv::Vec3b>(x, y)[0] + 128);
      out[x][y][2] = (uchar) (0.5 * (double)img.at<cv::Vec3b>(x, y)[2] - 0.4187 * (double)img.at<cv::Vec3b>(x, y)[1] - 0.0813 * (double)img.at<cv::Vec3b>(x, y)[0] + 128);
    }
  }
  return out;
}

cv::Mat cvt_YCbCr2BGR(std::vector<std::vector<std::vector<double>>> img){
  // get height and width
  int width = img[0].size();
  int height = img.size();

  // output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  // convert to YCbCr
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      out.at<cv::Vec3b>(x, y)[2] = (uchar) (img[x][y][0] + 1.402 * (img[x][y][2] - 128.));
      out.at<cv::Vec3b>(x, y)[1] = (uchar) (img[x][y][0] - 0.3441 * (img[x][y][1] - 128.) - 0.7139 * (img[x][y][2] - 128.));
      out.at<cv::Vec3b>(x, y)[0] = (uchar) (img[x][y][0] + 1.7718 * (img[x][y][1] - 128.));
    }
  }
  return out;
}

cv::Mat cvt_Mat_YCbCr2BGR(cv::Mat img){
  // get height and width
  int width = img.cols;
  int height = img.rows;

  // output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  // convert to YCbCr
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      out.at<cv::Vec3b>(x, y)[2] = (uchar) ((double)img.at<cv::Vec3b>(x, y)[0] + 1.402 * ((double)img.at<cv::Vec3b>(x, y)[2] - 128.));
      out.at<cv::Vec3b>(x, y)[1] = (uchar) ((double)img.at<cv::Vec3b>(x, y)[0] - 0.3441 * ((double)img.at<cv::Vec3b>(x, y)[1] - 128.) - 0.7139 * ((double)img.at<cv::Vec3b>(x, y)[2] - 128.));
      out.at<cv::Vec3b>(x, y)[0] = (uchar) ((double)img.at<cv::Vec3b>(x, y)[0] + 1.7718 * ((double)img.at<cv::Vec3b>(x, y)[1] - 128.));
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

std::vector<std::vector<std::vector<double>>> vec_DCT(std::vector<std::vector<std::vector<double>>> img, int T){
  int width = img[0].size();
  int height = img.size();

  std::vector<std::vector<std::vector<double>>> F(height, std::vector<std::vector<double>>(width, std::vector<double>(3, 0.0)));
  for (int c = 0; c < 3; c++){
    for (int x0 = 0; x0 < height; x0 += T){
      for (int y0 = 0; y0 < width; y0 += T){
        for (int u = 0; u < T; u++){
          for (int v = 0; v < T; v++){
            double sum = 0.0;
            for (int x = 0; x < T; x++){
              for (int y = 0; y < T; y++){
                sum += (double) img[x0+x][y0+y][c] * w(x, y, u, v, T);
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

std::vector<std::vector<std::vector<double>>> quantize(std::vector<std::vector<std::vector<double>>> F){
  int height = F.size();
  int width = F[0].size();
  std::vector<std::vector<std::vector<double>>> Fq(height, std::vector<std::vector<double>>(width, std::vector<double>(3, 0.0)));
  for (int c = 0; c < 3; c++){
    auto Q = (c == 0 ? Q1 : Q2);
    for (int x = 0; x < height; x++){
      for (int y = 0; y < width; y++){
        Fq[x][y][c] = round(F[x][y][c] / Q[x % 8][y % 8]) * Q[x % 8][y % 8];
      }
    }
  }
  return Fq;
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
  int width = img.cols;
  int height = img.rows;

  auto ycbcr = cvt_BGR2YCbCr(img);
  ycbcr = vec_DCT(ycbcr, 8);
  ycbcr = quantize(ycbcr);
  auto idct = IDCT(ycbcr, 8, 8);
  auto out = cvt_Mat_YCbCr2BGR(idct);

  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;
  
  cv::imshow("answer", out);
  cv::imwrite("out.jpg", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}