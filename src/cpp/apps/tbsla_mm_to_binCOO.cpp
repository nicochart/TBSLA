#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/mm.hpp>

#include <fstream>


int main(int argc, char** argv) {

  if(argc == 3) {
    MatrixCOO m = tbsla::utils::io::readMM(std::string(argv[1]));
    std::ofstream os(std::string(argv[2]), std::ofstream::binary);
    m.print_stats(std::cout);
    m.print_infos(std::cout);
    m.write(os);
    os.close();

    std::cout << "== read m2 ==" << std::endl;
    std::ifstream is(std::string(argv[2]), std::ifstream::binary);
    MatrixCOO m2;
    m2.read(is);
    is.close();
    m2.print_stats(std::cout);
    m2.print_infos(std::cout);
  }
}
