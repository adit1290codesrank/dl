#pragma once
#include <cublas_v2.h>
#include <stdexcept>

class CudaContext
{
    private:
        cublasHandle_t cublas_handle;

        CudaContext()
        {
            if (cublasCreate(&cublas_handle)!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS initialization failed!");
        }

        ~CudaContext()
        {
            cublasDestroy(cublas_handle);
        }

    public:
        CudaContext(const CudaContext&) = delete;
        CudaContext& operator=(const CudaContext&) = delete;

        static CudaContext& get_instance()
        {
            static CudaContext instance;
            return instance;
        }

        cublasHandle_t get_cublas_handle() const
        {
            return cublas_handle;
        }
};