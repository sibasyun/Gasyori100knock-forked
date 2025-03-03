// Discrete cosine transform and PSNR
// 44~46 Hough transform
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <math.h>
#include <chrono>

constexpr std::array<float, 3> gray_scale_coef = {0.0722, 0.7152, 0.2126};

constexpr double pi = 3.141593;

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

std::vector<std::vector<int>> Hough_vote(cv::Mat edge){
  int height = edge.rows;
  int width = edge.cols;
  int rmax = sqrt(height * height + width * width);
  std::vector<std::vector<int>> out(rmax * 2 + 1, std::vector<int>(180, 0));
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      if (edge.at<uchar>(x, y) == 255){
        for (int theta = 0; theta < 180; theta++){
          double rad = theta * M_PI / 180;
          int rho = y * cos(rad) + x * sin(rad);
          out[rho + rmax][theta] += 1;
        }
      }
    }
  }
  return out;
}

cv::Mat convert_vec2Mat(std::vector<std::vector<int>> vote_result){
  int height = vote_result.size();
  int width = vote_result[0].size();
  int max_vote = 0;
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      max_vote = std::max(max_vote, vote_result[x][y]);
    }
  }
  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      out.at<uchar>(x, y) = (uchar)(255 * vote_result[x][y] / max_vote);
    }
  }
  return out;
}

std::vector<std::vector<int>> NMS_Hough(std::vector<std::vector<int>> hough_vote){
  int height = hough_vote.size();
  int width = hough_vote[0].size();
  std::vector<std::vector<int>> out(height, std::vector<int>(width, 0));
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      int v = hough_vote[x][y];
      bool is_max = true;
      for (int dx = -1; dx < 2; dx++){
        for (int dy = -1; dy < 2; dy++){
          if (x + dx >= 0 && x + dx < height && y + dy >= 0 && y + dy < width){
            if (v < hough_vote[x + dx][y + dy]){
              is_max = false;
              break;
            }
          }
        }
        if (!is_max){
          break;
        }
      }
      if (is_max){
        out[x][y] = v;
      }
    }
  }
  return out;
} 

int main(int argc, const char* argv[]){
  // read image
  std::chrono::system_clock::time_point  start, end; // 型は auto で可
  start = std::chrono::system_clock::now(); // 計測開始時間

  cv::Mat img = cv::imread("thorino.jpg", cv::IMREAD_COLOR);
  int width = img.cols;
  int height = img.rows;
  cv::Mat gray = cvt_gray_scale(img);
  cv::Mat gaussian = gaussian_filter(gray, 1.4, 5);
  cv::Mat sobel_x = sobel_filter(gaussian, 3, true);
  cv::Mat sobel_y = sobel_filter(gaussian, 3, false);
  cv::Mat edge_img = edge(sobel_x, sobel_y);
  cv::Mat angle = angle_quantize(sobel_x, sobel_y);
  cv::Mat nms = NMS(edge_img, angle);
  cv::Mat canny = Canny(nms);
  // ここまでcanny法

  auto vote_result = Hough_vote(canny);
  auto nms_hough = NMS_Hough(vote_result);
  auto out = convert_vec2Mat(nms_hough);

  end = std::chrono::system_clock::now();  // 計測終了時間
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換
  std::cout << elapsed << std::endl;
  
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}