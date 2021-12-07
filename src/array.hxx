#pragma once

template<typename T>
Array<T>::Array(const size_t size_dim1, const size_t size_dim2)
{
    this->size_dim1 = size_dim1;
    this->size_dim2 = size_dim2;

    this->ptr = (T*) calloc(size_dim1 * size_dim2, sizeof(T));
}

template<typename T>
Array<T>::~Array()
{
    free(this->ptr);
}


template<typename T>
T& Array<T>::at(const size_t i_dim1, const size_t i_dim2)
{
    return this->ptr[this->size_dim1 * i_dim1 + i_dim2];
}
