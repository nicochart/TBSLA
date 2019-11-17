#include <tbsla/cpp/utils/InputParser.hpp>
#include <iostream>

int main(int argc, char **argv) {

  InputParser input(argc, argv);
  std::cout << "-f" << input.get_opt("-f") << std::endl;
  std::cout << "-i" << input.get_opt("-i", "default") << std::endl;
  std::cout << "-f" << input.has_opt("-f") << std::endl;
  std::cout << "-g" << input.has_opt("-g") << std::endl;
}
