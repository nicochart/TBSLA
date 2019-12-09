#ifndef TBSLA_HPX_MatrixCSR
#define TBSLA_HPX_MatrixCSR

#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/hpx/Matrix.hpp>
#include <tbsla/hpx/Vector.hpp>

#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

namespace tbsla { namespace hpx {

class MatrixCSR : public tbsla::cpp::MatrixCSR, public tbsla::hpx::Matrix {
  public:
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    void fill_cdiag(int n_row, int n_col, int cdiag, int rp = 0, int RN = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed, int rp = 0, int RN = 1);
  protected:
    int row_incr = 0; // index of the first value of the local array in the global array

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& rowptr& colidx& values& gnnz& row_incr;
    }
};

}}

struct HPX_COMPONENT_EXPORT MatrixCSR_server : ::hpx::components::component_base<MatrixCSR_server>
{
    // construct new instances
    MatrixCSR_server() {}

    MatrixCSR_server(tbsla::hpx::MatrixCSR const& data)
      : data_(data)
    {
    }

    MatrixCSR_server(std::size_t i, std::size_t n, std::string matrix_file)
      : data_()
    {
       std::ifstream is(matrix_file, std::ifstream::binary);
       data_.read(is, i, n);
       is.close();
    }

    MatrixCSR_server(std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : data_()
    {
       data_.fill_cdiag(nr, nc, cdiag, pos, nt);
    }

    MatrixCSR_server(std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pos, std::size_t nt)
      : data_()
    {
       data_.fill_cqmat(nr, nc, c, q, seed, pos, nt);
    }

    // Access data.
    tbsla::hpx::MatrixCSR get_data() const
    {
        return data_;
    }

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(MatrixCSR_server, get_data, get_data_action);

private:
    tbsla::hpx::MatrixCSR data_;
};



struct MatrixCSR_client : ::hpx::components::client_base<MatrixCSR_client, MatrixCSR_server>
{
    typedef ::hpx::components::client_base<MatrixCSR_client, MatrixCSR_server> base_type;

    MatrixCSR_client() {}

    MatrixCSR_client(hpx::id_type where, tbsla::hpx::MatrixCSR const& data)
      : base_type(hpx::new_<MatrixCSR_server>(hpx::colocated(where), data))
    {
    }

    MatrixCSR_client(hpx::id_type where, std::size_t i, std::size_t n, std::string matrix_file)
      : base_type(hpx::new_<MatrixCSR_server>(hpx::colocated(where), i, n, matrix_file))
    {
    }

    MatrixCSR_client(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : base_type(hpx::new_<MatrixCSR_server>(hpx::colocated(where), nr, nc, cdiag, pos, nt))
    {
    }

    MatrixCSR_client(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pos, std::size_t nt)
      : base_type(hpx::new_<MatrixCSR_server>(hpx::colocated(where), nr, nc, c, q, seed, pos, nt))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    MatrixCSR_client(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    MatrixCSR_client(hpx::future<MatrixCSR_client>&& c)
      : base_type(std::move(c))
    {
    }

    ::hpx::future<tbsla::hpx::MatrixCSR> get_data() const
    {
        MatrixCSR_server::get_data_action act;
        return ::hpx::async(act, get_id());
    }

};

Vector_client do_spmv_csr(std::size_t N, std::string matrix_file);
Vector_client do_spmv_csr_cdiag(Vector_client v, std::size_t N, int nr, int nc, int cdiag);
Vector_client do_spmv_csr_cqmat(Vector_client v, std::size_t N, int nr, int nc, int c, double q, unsigned int seed);
Vector_client do_a_axpx__csr_cdiag(Vector_client v, std::size_t N, int nr, int nc, int cdiag);

#endif
