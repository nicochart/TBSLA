#ifndef TBSLA_HPX_Vector
#define TBSLA_HPX_Vector

#include <hpx/hpx.hpp>
#include <hpx/serialization/serialize.hpp>
#include <vector>

class Vector_data {
  public:
    std::vector<double> get_vect() const { return data_; }
    Vector_data(std::vector<double> v) : data_(v) {}
    Vector_data(std::size_t n) : data_(n) {}
    Vector_data(Vector_data const& other) : data_(other.data_) {}
    Vector_data() : data_() {}

  private:
    friend ::hpx::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar& data_;
    }

   std::vector<double> data_;
};

struct Vector_server : hpx::components::component_base<Vector_server>
{
    // construct new instances
    Vector_server() {}

    Vector_server(Vector_data const& data)
      : data_(data)
    {
    }

    Vector_server(std::size_t n)
      : data_(n)
    {
    }

    // Access data.
    Vector_data get_data() const
    {
        return data_;
    }

    HPX_DEFINE_COMPONENT_DIRECT_ACTION(
        Vector_server, get_data, get_Vector_data_action);

private:
    Vector_data data_;
};



struct Vector_client : hpx::components::client_base<Vector_client, Vector_server>
{
    typedef hpx::components::client_base<Vector_client, Vector_server> base_type;

    Vector_client() {}

    // Create new component on locality 'where' and initialize the held data
    Vector_client(hpx::id_type where, std::size_t n)
      : base_type(hpx::new_<Vector_server>(where, n))
    {
    }

    Vector_client(hpx::id_type where, Vector_data const& data)
      : base_type(hpx::new_<Vector_server>(hpx::colocated(where), data))
    {
    }

    // Attach a future representing a (possibly remote) partition.
    Vector_client(hpx::future<hpx::id_type>&& id)
      : base_type(std::move(id))
    {
    }

    // Unwrap a future<partition> (a partition already holds a future to the
    // id of the referenced object, thus unwrapping accesses this inner future).
    Vector_client(hpx::future<Vector_client>&& c)
      : base_type(std::move(c))
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    // Invoke the (remote) member function which gives us access to the data.
    // This is a pure helper function hiding the async.
    hpx::future<Vector_data> get_data() const
    {
        Vector_server::get_Vector_data_action act;
        return hpx::async(act, get_id());
    }

};

Vector_client reduce(std::vector<Vector_client>);


#endif
