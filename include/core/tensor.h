#pragma once
#include <memory>
#include <vector>

class Tensor
{
    private:
        std::shared_ptr<float> d_ptr;
    
    public:
        std::vector<int> shape;
        int rows() const {return shape.empty()?0:shape[0];}
        int cols() const
        {
            if(shape.size()<2) return 1;
            int total=1;
            for(int i=1;i<shape.size();i++)total*=shape[i];
            return total;
        }
        
        Tensor(int rows, int cols);
        Tensor(std::vector<int> shape);

        float *data() const;

        Tensor reshape(std::vector<int> shape) const;
        Tensor flatten() const;

        Tensor im2col(int kernel_h,int kernel_w,int stride,int padding) const;
        Tensor col2im(const std::vector<int>& output_shape,int kernel_h,int kernel_w,int stride,int padding) const;

        static Tensor upload(const std::vector<float>& data,int r,int c);
        static Tensor upload(const std::vector<float>& data,const std::vector<int>& shape);
        std::vector<float> download() const;
        static Tensor zeros(int r,int c);
        Tensor operator*(const Tensor& other) const;
        Tensor slice(int start_row,int num_rows) const;
        size_t total_elements() const;
};