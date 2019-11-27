#ifndef TBSLA_HPX_Matrix
#define TBSLA_HPX_Matrix

#include <tbsla/cpp/Matrix.hpp>

#include <fstream>
#include <vector>

namespace tbsla { namespace hpx {

class Matrix : public virtual tbsla::cpp::Matrix {
  public:
    int const get_gnnz() {return gnnz;};

  protected:
    int gnnz;
};

}}

#endif
