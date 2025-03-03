// morphology dilate
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

// Gray -> Binary
// from answer 4
cv::Mat Binarize_Otsu(cv::Mat gray){
  int width = gray.cols;
  int height = gray.rows;

  // determine threshold
  double w0 = 0, w1 = 0;
  double m0 = 0, m1 = 0;
  double max_sb = 0, sb = 0;
  int th = 0;
  int val;

  // Get threshold
  for (int t = 0; t < 255; t++){
    w0 = 0;
    w1 = 0;
    m0 = 0;
    m1 = 0;
    for (int y = 0; y < height; y++){
      for (int x = 0; x < width; x++){
        val = (int)(gray.at<uchar>(y, x));

        if (val < t){
          w0++;
          m0 += val;
        } else {
          w1++;
          m1 += val;
        }
      }
    }

    m0 /= w0;
    m1 /= w1;
    w0 /= (height * width);
    w1 /= (height * width);
    sb = w0 * w1 * pow((m0 - m1), 2);
    
    if(sb > max_sb){
      max_sb = sb;
      th = t;
    }
  }

  std::cout << "threshold:" << th << std::endl;

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  // each y, x
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      // Binarize
      if (gray.at<uchar>(y, x) > th){
        out.at<uchar>(y, x) = 255;
      } else {
        out.at<uchar>(y, x) = 0;
      }
    
    }
  }

  return out;
}

// モルフォロジー変換(膨張・収縮)
// type = 0: 膨張, 1: 収縮
cv::Mat Morphology(cv::Mat img, int type){
  // get height and width
  int width = img.cols;
  int height = img.rows;

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  // kernelを作用させた値
  std::vector<std::vector<int>> output_by_kernel(height, std::vector<int>(width, 0));

  // kernel
  int kernel[3][3] = {{0, 1, 0}, {1, 0, 1}, {0, 1, 0}};
  auto f = [&](int value, int x, int y){
    int coeff = 4;
    coeff -= (x == 0 || x == height - 1);
    coeff -= (y == 0 || y == width - 1);
    if (type == 0){
      // 膨張
      return value >= 255;
    } else {
      // 収縮
      return value == 255 * coeff;
    }
  };

  // each y, x
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      for (int dx = -1; dx <= 1; dx++){
        for (int dy = -1; dy <= 1; dy++){
          if (x + dx < 0 || x + dx >= height || y + dy < 0 || y + dy >= width) continue;
          output_by_kernel[x][y] += kernel[1 + dx][1 + dy] * (int) img.at<uchar>(x+dx, y+dy);
        }
      }
    }
  }
  // 値によって処理をする
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      if (f(output_by_kernel[x][y], x, y)){
        out.at<uchar>(x, y) = 255;
      } else out.at<uchar>(x, y) = 0;
    }
  }
  return out;
}

// モルフォロジー処理　膨張×2
int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  cv::Mat gray = cvt_gray_scale(img);
  cv::Mat bin = Binarize_Otsu(gray);
  cv::imwrite("out_47_bin.jpg", bin);
  int type = 1;
  cv::Mat out = Morphology(bin, type);
  out = Morphology(out, type);

  // 計測終了
  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;

  cv::imwrite("out_48.jpg", out);
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}