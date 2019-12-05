#ifndef TBSLA_HPX_MatrixELL
#define TBSLA_HPX_MatrixELL

#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/hpx/Matrix.hpp>
#include <tbsla/hpx/Vector.hpp>

#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

namespace tbsla { namespace hpx {

class MatrixELL : public tbsla::cpp::MatrixELL, public tbsla::hpx::Matrix {
  public:
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    void fill_cdiag(int n_row, int n_col, int cdiag, int rp = 0, int RN = 1);
  protected:
    int row_incr = 0; // index of the first value of the local array in the global array

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& columns& values& row_incr& max_col;
    }
};

}}

struct HPX_COMPONENT_EXPORT MatrixELL_server : ::hpx::components::component_base<MatrixELL_server>
{
    // construct new instances
    MatrixELL_server() {}

    MatrixELL_server(tbsla::hpx::MatrixELL const& data)
      : data_(data)
    {
    }

    MatrixELL_server(std::size_t i, std::size_t n, std::string matrix_file)
      : data_()
    {
       std::ifstream is(matrix_file, std::ifstream::binary);
       data_.read(is, i, n);
       is.close();
    }

    MatrixELL_server(std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : data_()
    {
       data_.fill_cdiag(nr, nc, cdiag, pos, nt);
    }

    // Access data.
    tbsla::hpx::MatrixELL get_data() const
    {
        return data_;
    }

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(MatrixELL_server, get_data, get_data_action);

private:
    tbsla::hpx::MatrixELL data_;
};



struct MatrixELL_client : ::hpx::components::client_base<MatrixELL_client, MatrixELL_server>
{
    typedef ::hpx::components::client_base<MatrixELL_client, MatrixELL_server> base_type;

    MatrixELL_client() {}

    MatrixELL_client(hpx::id_type where, tbsla::hpx::MatrixELL const& data)
      : base_type(hpx::new_<MatrixELL_server>(hpx::colocated(where), data))
    {
    }

    MatrixELL_client(hpx::id_type where, std::size_t i, std::size_t n, std::string matrix_file)
      : base_type(hpx::new_<MatrixELL_server>(hpx::colocated(where), i, n, matrix_file))
    {
    }

    MatrixELL_client(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : base_type(hpx::new_<MatrixELL_server>(hpx::colocated(where), nr, nc, cdiag, pos, nt))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    MatrixELL_client(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    MatrixELL_client(hpx::future<MatrixELL_client>&& c)
      : base_type(std::move(c))
    {
    }

    ::hpx::future<tbsla::hpx::MatrixELL> get_data() const
    {
        MatrixELL_server::get_data_action act;
        return ::hpx::async(act, get_id());
    }

};

Vector_client do_spmv_ell(std::size_t N, std::string matrix_file);
Vector_client do_spmv_ell_cdiag(Vector_client v, std::size_t N, int nr, int nc, int cdiag);

#endif
