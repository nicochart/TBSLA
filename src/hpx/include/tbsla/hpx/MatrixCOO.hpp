#ifndef TBSLA_HPX_MatrixCOO
#define TBSLA_HPX_MatrixCOO

#include <tbsla/hpx/Matrix.hpp>
#include <tbsla/hpx/Vector.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <hpx/hpx.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>

namespace tbsla { namespace hpx {

class MatrixCOO : public tbsla::cpp::MatrixCOO, public virtual tbsla::hpx::Matrix {
  public:
    using tbsla::cpp::MatrixCOO::spmv;
    using tbsla::cpp::MatrixCOO::fill_cdiag;

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& row& col& values& gnnz;
    }
};

}}


struct HPX_COMPONENT_EXPORT MatrixCOO_server : ::hpx::components::component_base<MatrixCOO_server>
{
    // construct new instances
    MatrixCOO_server() {}

    MatrixCOO_server(tbsla::hpx::MatrixCOO const& data)
      : data_(data)
    {
    }

    MatrixCOO_server(std::size_t i, std::size_t n, std::string matrix_file)
      : data_()
    {
       std::ifstream is(matrix_file, std::ifstream::binary);
       data_.read(is, i, n);
       is.close();
    }

    MatrixCOO_server(std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : data_()
    {
       data_.fill_cdiag(nr, nc, cdiag, pos, nt);
    }

    // Access data.
    tbsla::hpx::MatrixCOO get_data() const
    {
        return data_;
    }

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(MatrixCOO_server, get_data, get_data_action);

private:
    tbsla::hpx::MatrixCOO data_;
};



struct MatrixCOO_client : ::hpx::components::client_base<MatrixCOO_client, MatrixCOO_server>
{
    typedef ::hpx::components::client_base<MatrixCOO_client, MatrixCOO_server> base_type;

    MatrixCOO_client() {}

    MatrixCOO_client(hpx::id_type where, tbsla::hpx::MatrixCOO const& data)
      : base_type(hpx::new_<MatrixCOO_server>(hpx::colocated(where), data))
    {
    }

    MatrixCOO_client(hpx::id_type where, std::size_t i, std::size_t n, std::string matrix_file)
      : base_type(hpx::new_<MatrixCOO_server>(hpx::colocated(where), i, n, matrix_file))
    {
    }

    MatrixCOO_client(hpx::id_type where, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t pos, std::size_t nt)
      : base_type(hpx::new_<MatrixCOO_server>(hpx::colocated(where), nr, nc, cdiag, pos, nt))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    MatrixCOO_client(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    MatrixCOO_client(hpx::future<MatrixCOO_client>&& c)
      : base_type(std::move(c))
    {
    }

    ::hpx::future<tbsla::hpx::MatrixCOO> get_data() const
    {
        MatrixCOO_server::get_data_action act;
        return ::hpx::async(act, get_id());
    }

};

Vector_client do_spmv_coo(std::size_t N, std::string matrix_file);

Vector_client do_spmv_coo_cdiag(Vector_client v, std::size_t N, int nr, int nc, int cdiag);

#endif
