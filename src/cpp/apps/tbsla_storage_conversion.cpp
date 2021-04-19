#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>

#include <fstream>


int main(int argc, char** argv) {

  if(argc == 3) {
    std::ifstream is(std::string(argv[1]), std::ifstream::binary);
    tbsla::cpp::MatrixCOO mcoo;
    mcoo.read(is);
    is.close();
    mcoo.print_stats(std::cout);
    mcoo.print_infos(std::cout);

    tbsla::cpp::MatrixCSR mcsr(mcoo);
    std::ofstream os(std::string(argv[2]), std::ofstream::binary);
    mcsr.write(os);
    os.close();
  }
}
