// morphology dilate
// opening
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

constexpr std::array<float, 3> gray_scale_coef = {0.0722, 0.7152, 0.2126};

constexpr float pi = 3.141593f;

float clip(float value, float min, float max){
  return fmin(fmax(value, 0), 255);
}

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

cv::Mat gaussian_filter(cv::Mat img, double sigma, int kernel_size){
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  if (channel == 1) {
    out = cv::Mat::zeros(height, width, CV_8UC1);
  }

  // prepare kernel
  int pad = floor(kernel_size / 2);
  int _x = 0, _y = 0;
  double kernel_sum = 0;
  
  // get gaussian kernel
  float kernel[kernel_size][kernel_size];

  for (int y = 0; y < kernel_size; y++){
    for (int x = 0; x < kernel_size; x++){
      _y = y - pad;
      _x = x - pad; 
      kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp( - (_x * _x + _y * _y) / (2 * sigma * sigma));
      kernel_sum += kernel[y][x];
    }
  }

  for (int y = 0; y < kernel_size; y++){
    for (int x = 0; x < kernel_size; x++){
      kernel[y][x] /= kernel_sum;
    }
  }

  // filtering
  double v = 0;
  
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      // for BGR
      if (channel == 3){
        for (int c = 0; c < channel; c++){
          v = 0;
          for (int dy = -pad; dy < pad + 1; dy++){
            for (int dx = -pad; dx < pad + 1; dx++){
              if (((x + dx) >= 0) && ((y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)){
                v += (double)img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
              }
            }
          }
          out.at<cv::Vec3b>(y, x)[c] = (uchar)clip(v, 0, 255);
        }
      } else {
        // for Gray
        v = 0;
        for (int dy = -pad; dy < pad + 1; dy++){
          for (int dx = -pad; dx < pad + 1; dx++){
            if (((x + dx) >= 0) && ((y + dy) >= 0) && ((x + dx) < width) && ((y + dy) < height)){
              v += (double)img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
            }
          }
        }
        out.at<uchar>(y, x) = (uchar)clip(v, 0, 255);
      }
    }
  }
  return out;
}


// Sobel filter
cv::Mat sobel_filter(cv::Mat img, int kernel_size, bool horizontal){
  int height = img.rows;
  int width = img.cols;
  int channel = img.channels();

  // prepare output
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  // prepare kernel
  double kernel[kernel_size][kernel_size] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  if (horizontal){
    kernel[0][1] = 0;
    kernel[0][2] = -1;
    kernel[1][0] = 2;
    kernel[1][2] = -2;
    kernel[2][0] = 1;
    kernel[2][1] = 0;
  }

  int pad = floor(kernel_size / 2);

  double v = 0;

  // filtering  
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      v = 0;
      for (int dy = -pad; dy < pad + 1; dy++){
        for (int dx = -pad; dx < pad + 1; dx++){
          if (((y + dy) >= 0) && (( x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)){
            v += img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
          }
        }
      }
      v = fmax(v, 0);
      v = fmin(v, 255);
      out.at<uchar>(y, x) = (uchar)v;
    }
  }
  return out;
}

cv::Mat edge(cv::Mat fx, cv::Mat fy){
  int height = fx.rows;
  int width = fx.cols;
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      float _fx = fx.at<uchar>(x, y);
      float _fy = fy.at<uchar>(x, y);
      out.at<uchar>(x, y) = (uchar)clip(sqrt(_fx * _fx + _fy * _fy), 0, 255);
    }
  }
  return out;
}

uchar quantize(double a){
  if (a <= 22.5){
    return 0;
  } else if (a <= 67.5){
    return 45;
  } else if (a <= 112.5){
    return 90;
  } else if (a <= 157.5){
    return 135;
  } else {
    return 0;
  }
}

cv::Mat angle_quantize(cv::Mat fx, cv::Mat fy){
  int height = fx.rows;
  int width = fx.cols;
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      double temp = atan2(fy.at<uchar>(x, y), fx.at<uchar>(x, y)) * 180 / M_PI;
      if (temp <= -22.5){
        temp += 180.;
      } else if (temp >= 157.5){
        temp -= 180.;
      }
      out.at<uchar>(x, y) = quantize(temp);
    }
  }
  return out;
}

std::vector<std::pair<int, int>> direction(uchar angle){
  if (angle == 0){
    return {{0, 1}, {0, -1}};
  } else if (angle == 45){
    return {{1, -1}, {-1, 1}};
  } else if (angle == 90){
    return {{1, 0}, {-1, 1}};
  } else if (angle == 135){
    return {{1, 1}, {-1, -1}};
  }
}

cv::Mat NMS(cv::Mat edge, cv::Mat angle){
  int height = edge.rows;
  int width = edge.cols;

  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      auto d = direction(angle.at<uchar>(x, y));
      uchar v = edge.at<uchar>(x, y);
      bool is_max = true;
      for (auto [dx, dy] : d){
        if (x + dx >= 0 && x + dx < height && y + dy >= 0 && y + dy < width){
          if (v < edge.at<uchar>(x + dx, y + dy)){
            is_max = false;
            break;
          }
        }
      }
      if (is_max){
        out.at<uchar>(x, y) = v;
      }
    }
  }
  return out;
}

cv::Mat Canny(cv::Mat edge, uchar HT = 50, uchar LT = 20){
  int height = edge.rows;
  int width = edge.cols;

  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      if (edge.at<uchar>(x, y) >= HT){
        out.at<uchar>(x, y) = 255;
      } else if (edge.at<uchar>(x, y) < LT){
        out.at<uchar>(x, y) = 0;
      } else {
        bool is_edge = false;
        for (int dx = -1; dx < 2; dx++){
          for (int dy = -1; dy < 2; dy++){
            if (x + dx >= 0 && x + dx < height && y + dy >= 0 && y + dy < width){
              if (edge.at<uchar>(x + dx, y + dy) >= HT){
                out.at<uchar>(x, y) = 255;
                is_edge = true;
                break;
              }
            }
          }
          if (is_edge){
            break;
          }
        }
      }
    }
  }
  return out;
}

cv::Mat CannyEdgeDetection(cv::Mat img){
  cv::Mat gray = cvt_gray_scale(img);
  cv::Mat gaussian = gaussian_filter(gray, 1.4, 5);
  cv::Mat sobel_x = sobel_filter(gaussian, 3, true);
  cv::Mat sobel_y = sobel_filter(gaussian, 3, false);
  cv::Mat edge_img = edge(sobel_x, sobel_y);
  cv::Mat angle = angle_quantize(sobel_x, sobel_y);
  cv::Mat nms = NMS(edge_img, angle);
  cv::Mat out = Canny(nms);
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

cv::Mat Opening(cv::Mat img, int N){
  for (int i = 0; i < N; i++){
    img = Morphology(img, 1);
  }
  for (int i = 0; i < N; i++){
    img = Morphology(img, 0);
  }
  return img;
}

cv::Mat Closing(cv::Mat img, int N){
  for (int i = 0; i < N; i++){
    img = Morphology(img, 0);
  }
  for (int i = 0; i < N; i++){
    img = Morphology(img, 1);
  }
  return img;
}

cv::Mat MorphologyGradient(cv::Mat img){
  cv::Mat dilate = Morphology(img, 0);
  cv::Mat erode = Morphology(img, 1);
  cv::Mat out = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  for (int y = 0; y < img.rows; y++){
    for (int x = 0; x < img.cols; x++){
      out.at<uchar>(y, x) = dilate.at<uchar>(y, x) - erode.at<uchar>(y, x);
    }
  }
  return out;
}

cv::Mat TopHat(cv::Mat img, int N){
  cv::Mat gray = cvt_gray_scale(img);
  cv::Mat bin = Binarize_Otsu(gray);
  cv::Mat opening = Opening(bin, N);
  cv::Mat out = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  for (int y = 0; y < img.rows; y++){
    for (int x = 0; x < img.cols; x++){
      int val = (int) bin.at<uchar>(y, x) - (int) opening.at<uchar>(y, x);
      out.at<uchar>(y, x) = (uchar) clip(val, 0, 255);
    }
  }
  return out;
}

cv::Mat BlackHat(cv::Mat img, int N){
  cv::Mat gray = cvt_gray_scale(img);
  cv::Mat bin = Binarize_Otsu(gray);
  cv::Mat closing = Closing(bin, N);
  cv::Mat out = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
  for (int y = 0; y < img.rows; y++){
    for (int x = 0; x < img.cols; x++){
      int val = (int) closing.at<uchar>(y, x) - (int) bin.at<uchar>(y, x);
      out.at<uchar>(y, x) = (uchar) clip(val, 0, 255);
    }
  }
  return out;
}

// closing
int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("imori.jpg", cv::IMREAD_COLOR);
  cv::Mat out = TopHat(img, 3);

  // 計測終了
  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;

  cv::imwrite("out_52.jpg", out);
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}