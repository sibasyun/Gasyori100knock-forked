#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include <chrono>
#include <random>

uint32_t get_rand() {
  // 乱数生成器（引数にシードを指定可能）
  static std::mt19937 mt32(0);

  // [0, (2^32)-1] の一様分布整数を生成
  return mt32();
}

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

// 58 4 nearest neighbor
// std::vector<std::pair<int, int>> dxdy{{-1, 0}, {0, -1}}; // 左と上

// 59 8 nearest neighbor
std::vector<std::pair<int, int>> dxdy{{-1, 0},{-1, -1}, {0, -1}, {-1, 1}}; // 左と上

std::vector<uchar> create_color(){
  uchar b = get_rand() % 256;
  uchar g = get_rand() % 256;
  uchar r = get_rand() % 256;
  return {b, g, r};
}

cv::Mat labeling(cv::Mat img){
  int height = img.rows;
  int width = img.cols;
  assert (img.channels() == 1); // gray scale image

  std::vector<std::vector<int>> label(height, std::vector<int>(width, -1));
  int now_label = 1;
  std::vector<int> colors(height * width, 0);

  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++)
    {
      // 画素が黒のときはlabel を0にしてスキップ
      if (img.at<uchar>(x, y) == 0) {
        label[x][y] = 0;
        continue;
      };

      // 4近傍のラベルを取得
      std::vector<int> cands;
      for (auto [dx, dy] : dxdy){
        int nx = x + dx;
        int ny = y + dy;

        if ((0 <= nx) && (nx < height) && (0 <= ny) && (ny < width)){
          cands.push_back(label[nx][ny]);
        }
      }
      std::sort(cands.begin(), cands.end());

      if (cands.empty() || cands.back() == 0)
      { // 左と上が0のとき
        label[x][y] = now_label;
        colors[now_label] = now_label;
        now_label++;
      } else {
        int min_label = int(1e9);

        for (int cand : cands){
          if (cand <= 0) continue;
          min_label = std::min(min_label, cand);
          colors[cand] = colors[min_label]; // colors (LUT)の更新
        }
        label[x][y] = colors[min_label];
      }
      
    }
  }

  // integration
  int count = 0;
  for (int l = 2; l <= now_label; l++){
    bool f = true;
    for (int i = 1; i < now_label; i++){
      if (colors[i] == l){
        if (f){
          count++;
          f = false;
        }
        colors[i] = count;
      }
    }
  }

  // 各ラベルの色をランダムに決定
  std::vector<std::vector<uchar>> color_map(now_label);
  for (int i = 0; i < now_label; i++){
    color_map[i] = create_color();
  }

  cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
  for (int x = 0; x < height; x++){
    for (int y = 0; y < width; y++){
      if (label[x][y] == 0) out.at<cv::Vec3b>(x, y) = {0, 0, 0};
      else {
        for (int c = 0; c < 3; c++){
          out.at<cv::Vec3b>(x, y)[c] = color_map[colors[label[x][y]]][c];
        }
      }
    }
  }
  return out;
}

int main(int argc, const char* argv[]){
  timeMeasure tm = timeMeasure();
  cv::Mat I = cv::imread("seg.png", cv::IMREAD_GRAYSCALE);
  cv:: Mat out = labeling(I);
  
  tm.elapsed_time();

  cv::imwrite("out_59.jpg", out);
  cv::imshow("answer", out);
  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}