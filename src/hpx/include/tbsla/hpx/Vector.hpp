#ifndef TBSLA_HPX_Vector
#define TBSLA_HPX_Vector

#include <hpx/hpx.hpp>
#include <hpx/serialization/serialize.hpp>
#include <vector>
#include <numeric>

namespace tbsla { namespace hpx_ { namespace detail {

class Vector {
  public:
    std::vector<double> get_vect() const { return data_; }
    Vector(std::vector<double> v) : data_(v) {}
    Vector(std::size_t n) : data_(n)
    {
      std::iota (std::begin(data_), std::end(data_), 0);
    }
    Vector(Vector const& other) : data_(other.data_) {}
    Vector() : data_() {}

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& data_;
    }

   std::vector<double> data_;
};

}}}

namespace tbsla { namespace hpx_ { namespace server {

struct Vector : hpx::components::component_base<Vector>
{
    // construct new instances
    Vector()
      : data_()
    {
    }

    Vector(tbsla::hpx_::detail::Vector const& data)
      : data_(data)
    {
    }

    Vector(std::size_t n)
      : data_(n)
    {
    }

    // Access data.
    tbsla::hpx_::detail::Vector get_data() const
    {
        return data_;
    }
    HPX_DEFINE_COMPONENT_ACTION(Vector, get_data);

private:
    tbsla::hpx_::detail::Vector data_;
};

}}}

HPX_REGISTER_ACTION_DECLARATION(tbsla::hpx_::server::Vector::get_data_action)

namespace tbsla { namespace hpx_ { namespace client {

struct Vector : hpx::components::client_base<Vector, tbsla::hpx_::server::Vector>
{
    typedef hpx::components::client_base<Vector, tbsla::hpx_::server::Vector> base_type;

    Vector() {}

    // Create new component on locality 'where' and initialize the held data
    Vector(hpx::id_type where, std::size_t n)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(where, n))
    {
    }

    Vector(hpx::id_type where, tbsla::hpx_::detail::Vector const& data)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(hpx::colocated(where), data))
    {
    }

    Vector(hpx::id_type where)
      : base_type(hpx::new_<tbsla::hpx_::server::Vector>(hpx::colocated(where)))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    Vector(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    Vector(hpx::future<tbsla::hpx_::client::Vector>&& c)
      : base_type(std::move(c))
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<tbsla::hpx_::detail::Vector> get_data() const
    {
        tbsla::hpx_::server::Vector::get_data_action act;
        return hpx::async(act, get_id());
    }

};

}}}

tbsla::hpx_::client::Vector reduce(std::vector<tbsla::hpx_::client::Vector>);
tbsla::hpx_::client::Vector add_vectors(hpx::id_type where, tbsla::hpx_::client::Vector v1, tbsla::hpx_::client::Vector v2);

namespace tbsla { namespace hpx_ { namespace detail {
  static tbsla::hpx_::client::Vector reduce_part(tbsla::hpx_::client::Vector const& vc1, tbsla::hpx_::client::Vector const& vc2);
}}}



#endif
