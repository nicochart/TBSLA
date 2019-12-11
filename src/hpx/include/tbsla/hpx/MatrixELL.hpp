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

namespace tbsla { namespace hpx_ { namespace detail {

class MatrixELL : public tbsla::cpp::MatrixELL, public tbsla::hpx_::detail::Matrix {
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
        ar& columns& values& gnnz& row_incr;
    }
};

}}}


namespace tbsla { namespace hpx_ { namespace server {
struct HPX_COMPONENT_EXPORT MatrixELL : ::hpx::components::component_base<MatrixELL>
{
    // construct new instances
    MatrixELL() {}

    MatrixELL(tbsla::hpx_::detail::MatrixELL const& data)
      : data_(data)
    {
    }

    MatrixELL(std::size_t i, std::size_t n, std::string matrix_file)
      : data_()
    {
       std::ifstream is(matrix_file, std::ifstream::binary);
       data_.read(is, i, n);
       is.close();
    }

    MatrixELL(std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : data_()
    {
       data_.fill_cdiag(nr, nc, cdiag, pos, nt);
    }

    MatrixELL(std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pos, std::size_t nt)
      : data_()
    {
       data_.fill_cqmat(nr, nc, c, q, seed, pos, nt);
    }

    // Access data.
    tbsla::hpx_::detail::MatrixELL get_data() const
    {
        return data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(MatrixELL, get_data);


private:
    tbsla::hpx_::detail::MatrixELL data_;
};


}}}

HPX_REGISTER_ACTION_DECLARATION(tbsla::hpx_::server::MatrixELL::get_data_action)

namespace tbsla { namespace hpx_ { namespace client {

struct MatrixELL : ::hpx::components::client_base<MatrixELL, tbsla::hpx_::server::MatrixELL>
{
    typedef ::hpx::components::client_base<MatrixELL, tbsla::hpx_::server::MatrixELL> base_type;

    MatrixELL() {}

    MatrixELL(hpx::id_type where, tbsla::hpx_::detail::MatrixELL const& data)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixELL>(hpx::colocated(where), data))
    {
    }

    MatrixELL(hpx::id_type where, std::size_t i, std::size_t n, std::string matrix_file)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixELL>(hpx::colocated(where), i, n, matrix_file))
    {
    }

    MatrixELL(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixELL>(hpx::colocated(where), nr, nc, cdiag, pos, nt))
    {
    }

    MatrixELL(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t pos, std::size_t nt)
      : base_type(hpx::new_<tbsla::hpx_::server::MatrixELL>(hpx::colocated(where), nr, nc, c, q, seed, pos, nt))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    MatrixELL(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    MatrixELL(hpx::future<MatrixELL>&& c)
      : base_type(std::move(c))
    {
    }

    ::hpx::future<tbsla::hpx_::detail::MatrixELL> get_data() const
    {
        tbsla::hpx_::server::MatrixELL::get_data_action act;
        return ::hpx::async(act, get_id());
    }

};

}}}

namespace tbsla { namespace hpx_ { namespace detail {

static Vector_client spmv_part(tbsla::hpx_::client::MatrixELL const& A_p, Vector_client const& v_p, Vector_client const& r_p);

}}}

namespace tbsla { namespace hpx_ {

class MatrixELL : public tbsla::hpx_::Matrix {
  public:
    void fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t nt);
    void fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t nt);
    void wait();
    void read(std::vector<hpx::id_type> localities, std::string matrix_file, std::size_t nt);
    std::size_t get_n_col();
    std::size_t get_n_row();
    Vector_client spmv(Vector_client v);
    Vector_client a_axpx_(Vector_client v);
  private:
    std::vector<tbsla::hpx_::client::MatrixELL> tiles;
    std::vector<hpx::id_type> localities;
};

}}

#endif
