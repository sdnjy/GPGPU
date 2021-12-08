#pragma once

#include <cstdlib>

template<typename T>
class Array
{
private:
    T* ptr;
    size_t size_dim1;
    size_t size_dim2;
public:
    Array(const size_t size_dim1, const size_t size_dim2);
    ~Array();

    T& at(const size_t i_dim1, const size_t i_dim2);
};

#include "array.hxx"