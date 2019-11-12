#ifndef TBSLA_CPP_MatrixMarket
#define TBSLA_CPP_MatrixMarket

#include <tbsla/cpp/MatrixCOO.hpp>
#include <string>

namespace tbsla { namespace utils { namespace io {

MatrixCOO readMM(std::string fname);

}}}

#endif
