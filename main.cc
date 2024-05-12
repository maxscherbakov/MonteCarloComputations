#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <utility>
#include <vector>
#define EXPECTED_RESULT M_PI_2
#define EPSILON 0.001
#define POINTS_COUNT 1000000

double square(std::pair<double, double> x_area, std::pair<double, double> y_area) {
  return (y_area.second - y_area.first) * (x_area.second - x_area.first);
}


bool monte_carlo_integration(std::pair<double, double>& x_area, std::pair<double, double>& y_area, int thread_count) {
  std::atomic<unsigned int> total_point_count(0);

  std::vector<std::thread> threads;
  threads.reserve(thread_count);

  for (int thread_num = 0; thread_num < thread_count; ++thread_num) {
    threads.emplace_back([&, thread_num]() {
      std::mt19937 gen(thread_num);
      std::uniform_real_distribution<double> distribution_x(x_area.first, x_area.second);
      std::uniform_real_distribution<double> distribution_y(y_area.first, y_area.second);

      unsigned int point_count = 0;

      for (unsigned int j = 0; j < POINTS_COUNT / thread_count + 1; ++j) {
        double x = distribution_x(gen);
        double y = distribution_y(gen);

        if (x * x + y * y <= 1) {
          point_count++;
        }
      }

      total_point_count += point_count;
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return EXPECTED_RESULT / square(x_area, y_area) - static_cast<double>(total_point_count) / thread_count >= EPSILON;
}


int main(int argc, char const* argv[]) {
  const int thread_count = std::stoi(argv[1]);

  std::pair<double, double> x_area = {-1.0, 1.0};
  std::pair<double, double> y_area = {0.0, 1.0};
  unsigned int bad_results = 0;
  unsigned int launches_count = 100;


  std::chrono::milliseconds total_time{0};  

  for (unsigned int launches = 0; launches < launches_count; launches++) {

    auto start = std::chrono::high_resolution_clock::now();
    bad_results += monte_carlo_integration(x_area, y_area, thread_count);
    auto end = std::chrono::high_resolution_clock::now();
    
    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  }

  std::cout << "Average runtime of monte_carlo run: ";
  std::cout << static_cast<double>(total_time.count()) / launches_count << " ms" << std::endl;

  bool estimation =  static_cast<double>(bad_results) / launches_count 
                                  <=
                    EXPECTED_RESULT * (square(x_area, y_area) - EXPECTED_RESULT) / (POINTS_COUNT * EPSILON * EPSILON * square(x_area, y_area) * square(x_area, y_area));

  if (estimation) {
    std::cout << "The estimation is correct" << std::endl;
  } else {
    std::cout << "Something went wrong..." << std::endl;
  }
}