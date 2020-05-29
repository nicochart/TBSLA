#ifndef TBSLA_CPP_UTILS_CSR
#define TBSLA_CPP_UTILS_CSR

#include <cassert>
#include <vector>

namespace tbsla { namespace cpp { namespace utils { namespace csr {

template <typename T>
std::vector<T> applyPermutation(
    const std::vector<int>& order,
    std::vector<T>& t)
{
    assert(order.size() == t.size());
    std::vector<T> st(t.size());
    for(int i=0; i<t.size(); i++)
    {
        st[i] = t[order[i]];
    }
    return st;
}

bool compare_row(std::vector<int> row, std::vector<int> col, unsigned i, unsigned j);

}}}}

#endif
