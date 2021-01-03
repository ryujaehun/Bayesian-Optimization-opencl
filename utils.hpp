//
// Created by jaehun on 20. 12. 12..
//

#ifndef CL_UTILS_HPP
#define CL_UTILS_HPP

#include <unistd.h>
#include "etc.h"
#include "etc.cpp"


size_t tile_size=16;
bool cpu=true;
bool cpu_kernel=true;
bool cpu_transpose=true;
bool cpu_inv=true;
bool cpu_dot=true;

Matrix kernel(const Matrix &X, const Matrix &Y)
        {
    if(cpu&&cpu_kernel) {
        Matrix temp(X.row, X.row);
        temp.data = new float[temp.col * temp.row];
        for (int i = 0; i < temp.row; i++)
            for (int j = 0; j < temp.col; j++) {
                float val = 0;
                for (int k = 0; k < X.col; k++)
                    val += (X.data[INDEX(i, k, X.col)] - Y.data[INDEX(j, k, X.col)]) *
                           (X.data[INDEX(i, k, X.col)] - Y.data[INDEX(j, k, X.col)]);
                temp.data[INDEX(i, j, temp.col)] = expf(-1 * val);
            }

        return temp;
    }
    else {
        CL_struct cl_struct;
        Matrix Y_T = Matrix(Y.col, Y.row);
        Y_T.data = new float[Y.col * Y.row];
        for (int i = 0; i < Y.row; i++)
            for (int j = 0; j < Y.col; j++)
                Y_T.data[INDEX(j, i, Y_T.col)] = Y.data[INDEX(i, j, Y.col)];
        Matrix C_matrix = dot(X, Y_T);
        Matrix result(C_matrix.row, C_matrix.col);
        Matrix x_square(X.row, 1);
        Matrix y_square(Y.row, 1);
        Matrix sum_square(X.row, 1);
        result.data = new float[result.row * result.col];
        x_square.data = new float[X.row * 1];
        y_square.data = new float[Y.row * 1];
        sum_square.data = new float[X.row * 1];
        string option = "";
        string file = string("rbf.cl");
        string name = string("matrix_square_add");
        cl_init(cl_struct, file, name, option);
        cl_int err = NULL;


        cl_kernel kernel2 = clCreateKernel(cl_struct.program, "matrix_vector_add", &err);
        if (!kernel2 || err != CL_SUCCESS) {
            error_print("matrix_vector_add ", err);
        }


        cl_mem x_device = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE, sizeof(float) * X.row * X.col, NULL,
                                         NULL);

        cl_mem y_device = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE, sizeof(float) * Y.row * Y.col, NULL,
                                         NULL);
        cl_mem sum_device_result = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE, sizeof(float) * Y.row, NULL,
                                                  NULL);

        cl_mem matrix = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE,
                                       sizeof(float) * C_matrix.row * C_matrix.col, NULL, NULL);
        cl_mem matrix_result = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE,
                                              sizeof(float) * C_matrix.row * C_matrix.col, NULL, NULL);

        err = clEnqueueWriteBuffer(cl_struct.queue, x_device, CL_TRUE, 0, sizeof(float) * X.row * X.col, X.data, 0,
                                   NULL, NULL);
        err |= clEnqueueWriteBuffer(cl_struct.queue, y_device, CL_TRUE, 0, sizeof(float) * Y.row * Y.col, Y.data, 0,
                                    NULL, NULL);

        err |= clEnqueueWriteBuffer(cl_struct.queue, sum_device_result, CL_TRUE, 0, sizeof(float) * Y.row,
                                    sum_square.data, 0, NULL, NULL);

        err |= clEnqueueWriteBuffer(cl_struct.queue, matrix, CL_TRUE, 0, sizeof(float) * C_matrix.row * C_matrix.col,
                                    C_matrix.data, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(cl_struct.queue, matrix_result, CL_TRUE, 0,
                                    sizeof(float) * C_matrix.row * C_matrix.col, result.data, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
            error_print("clEnqueueWriteBuffer !", err);

        }


        err = clSetKernelArg(cl_struct.kernel, 0, sizeof(cl_mem), &x_device);
        err |= clSetKernelArg(cl_struct.kernel, 1, sizeof(cl_mem), &y_device);
        err |= clSetKernelArg(cl_struct.kernel, 2, sizeof(cl_mem), &sum_device_result);
        err |= clSetKernelArg(cl_struct.kernel, 3, sizeof(int), &X.col);
        err |= clSetKernelArg(cl_struct.kernel, 4, X.col * sizeof(cl_float), NULL);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg", err);

        }

        err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &matrix);
        err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &sum_device_result);
        err |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &matrix_result);
        err |= clSetKernelArg(kernel2, 3, sizeof(int), &X.row);


        const size_t local1[2] = {1, (size_t) X.col};
        const size_t global1[2] = {(X.row + local1[0] - 1) / local1[0] * local1[0],
                                   (X.col + local1[1] - 1) / local1[1] * local1[1]};
        const size_t global2[2] = {(size_t) C_matrix.row, (size_t) C_matrix.col};

        clEnqueueNDRangeKernel(cl_struct.queue, cl_struct.kernel, 2, NULL, global1, local1, 0, NULL, NULL);
        clEnqueueNDRangeKernel(cl_struct.queue, kernel2, 2, NULL, global2, NULL, 0, NULL, NULL);

        clFinish(cl_struct.queue);
        err = clEnqueueReadBuffer(cl_struct.queue, matrix_result, CL_TRUE, 0,
                                  sizeof(float) * C_matrix.row * C_matrix.col, result.data, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
            error_print("clEnqueueReadBuffer rbf", err);
        }


        cl_complete(cl_struct);
        clReleaseKernel(kernel2);
        clReleaseMemObject(x_device);
        clReleaseMemObject(y_device);
        clReleaseMemObject(sum_device_result);
        clReleaseMemObject(matrix);
        clReleaseMemObject(matrix_result);

        return result;

    }

        };

class GP
{
private:
    float  sigma_y=1;
    Matrix X;
    Matrix Y;
    Matrix K;


public:
    void fit (Matrix &X,Matrix &Y)
    {
        this-> X.col=X.col;
        this-> X.row=X.row;
        this-> Y.col=Y.col;
        this-> Y.row=Y.row;
        this->X.data=new float[X.row*X.col];
        this->Y.data=new float[Y.row*Y.col];
        for(int i=0;i<X.row*X.col;i++)
            this->X.data[i]=X.data[i];
        for(int i=0;i<Y.row*Y.col;i++)
            this->Y.data[i]=Y.data[i];
        this->K=kernel(X,X);


    }

    Matrix predict (const Matrix & X_test)
    {
        Matrix k_inv=right_inv(this->K);
        Matrix K_star= kernel(this->X, X_test);
        Matrix K_star_star= kernel(X_test, X_test);
        Matrix A=transpose(K_star);
        Matrix A_=dot(A,k_inv);
        Matrix mu_test=dot(A_,this->Y);
        return mu_test;
    }
public:
    GP( float sigma_y)
           :sigma_y(sigma_y)
    {

    }

};



Matrix transpose(const Matrix &M)
{


        Matrix result(M.col, M.row);
        result.data = new float[M.col * M.row];
    if(cpu&&cpu_transpose) {
        for (int i = 0; i < M.row; i++)
            for (int j = 0; j < M.col; j++)
                result.data[INDEX(j, i, result.col)] = M.data[INDEX(i, j, M.col)];
    }else {
        CL_struct cl_struct;

        string option = "-DBLOCK_DIM=" + to_string(tile_size);
        string file = string("transpose.cl");
        string name = string("transpose");
        cl_init(cl_struct, file, name, option);

        cl_int err = NULL;
        cl_mem deviceInput = clCreateBuffer(cl_struct.context, CL_MEM_READ_ONLY, sizeof(float) * M.row * M.col, NULL,
                                            NULL);
        cl_mem deviceOutput = clCreateBuffer(cl_struct.context, CL_MEM_WRITE_ONLY, sizeof(float) * M.row * M.col, NULL,
                                             NULL);

        //@@ Write our data set into the input array in device memory
        err = clEnqueueWriteBuffer(cl_struct.queue, deviceInput, CL_TRUE, 0, sizeof(float) * M.row * M.col, M.data, 0,
                                   NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueWriteBuffer deviceInput", err);
        }

        err = clEnqueueWriteBuffer(cl_struct.queue, deviceOutput, CL_TRUE, 0, sizeof(float) * M.row * M.col,
                                   result.data, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueWriteBuffer deviceOutput", err);
        }

        //@@ Set the arguments to our compute kernel
        err = clSetKernelArg(cl_struct.kernel, 0, sizeof(cl_mem), &deviceOutput);
        err |= clSetKernelArg(cl_struct.kernel, 1, sizeof(cl_mem), &deviceInput);
        err |= clSetKernelArg(cl_struct.kernel, 2, sizeof(int), &M.row);
        err |= clSetKernelArg(cl_struct.kernel, 3, sizeof(int), &M.row);
        err |= clSetKernelArg(cl_struct.kernel, 4, (tile_size + 1) * tile_size * sizeof(float), 0);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg", err);

        }


        //@@ Execute the kernel over the entire range of the data set
        const size_t local[2] = {tile_size, tile_size};
        const size_t global[2] = {(M.row + local[0] - 1) / local[0] * local[0],
                                  (M.col + local[1] - 1) / local[1] * local[1]};
        err = clEnqueueNDRangeKernel(cl_struct.queue,
                                     cl_struct.kernel,
                                     2,
                                     NULL,
                                     global,
                                     local,
                                     0,
                                     NULL,
                                     NULL);

        if (err != CL_SUCCESS) {
            error_print("clEnqueueNDRangeKernel", err);
        }
        //@@ Wait for the command queue to get serviced before reading back results
        clFinish(cl_struct.queue);


        //@@ Read the results from the device
        err = clEnqueueReadBuffer(cl_struct.queue, deviceOutput, CL_TRUE, 0, sizeof(float) * M.row * M.col, result.data,
                                  0, NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueReadBuffer transpose", err);
        }

        cl_complete(cl_struct);
        clReleaseMemObject(deviceInput);
        clReleaseMemObject(deviceOutput);
    }
    return result;


}

Matrix dot(const Matrix &X1,const Matrix &X2) {

    Matrix result(X1.row, X2.col);
    result.data = new float[X1.row * X2.col];
    if (cpu&&cpu_dot){

        for (int i = 0; i < X1.row; i++)
            for (int j = 0; j < X2.col; j++) {
                float temp = 0.f;
                for (int k = 0; k < X1.col; k++)
                    temp += X1.data[INDEX(i, k, X1.col)] * X2.data[INDEX(k, j, X2.col)];
                result.data[INDEX(i, j, result.col)] = temp;
            }
    }
    else {
        CL_struct cl_struct;
        string option = "-DTS=" + to_string(tile_size);
        string file = string("gemm.cl");
        string name = string("gemm");
        cl_init(cl_struct, file, name, option);

        cl_int err = NULL;
        cl_mem deviceInput1 = clCreateBuffer(cl_struct.context, CL_MEM_READ_ONLY, sizeof(float) * X1.row * X1.col, NULL,
                                             NULL);
        cl_mem deviceInput2 = clCreateBuffer(cl_struct.context, CL_MEM_READ_ONLY, sizeof(float) * X2.row * X2.col, NULL,
                                             NULL);
        cl_mem deviceOutput = clCreateBuffer(cl_struct.context, CL_MEM_WRITE_ONLY, sizeof(float) * X1.row * X2.col,
                                             NULL, NULL);

        //@@ Write our data set into the input array in device memory
        err = clEnqueueWriteBuffer(cl_struct.queue, deviceInput1, CL_TRUE, 0, sizeof(float) * X1.row * X1.col, X1.data,
                                   0, NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueWriteBuffer deviceInput1", err);
        }
        err = clEnqueueWriteBuffer(cl_struct.queue, deviceInput2, CL_TRUE, 0, sizeof(float) * X2.row * X2.col, X2.data,
                                   0, NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueWriteBuffer deviceInput2", err);
        }
        err = clEnqueueWriteBuffer(cl_struct.queue, deviceOutput, CL_TRUE, 0, sizeof(float) * X1.row * X2.col,
                                   result.data, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueWriteBuffer deviceOutput", err);
        }

        //@@ Set the arguments to our compute kernel
        err = 0;
        err = clSetKernelArg(cl_struct.kernel, 0, sizeof(int), &X1.row);
        err |= clSetKernelArg(cl_struct.kernel, 1, sizeof(int), &X2.col);
        err |= clSetKernelArg(cl_struct.kernel, 2, sizeof(int), &X2.row);
        err |= clSetKernelArg(cl_struct.kernel, 3, sizeof(cl_mem), &deviceInput1);
        err |= clSetKernelArg(cl_struct.kernel, 4, sizeof(cl_mem), &deviceInput2);
        err |= clSetKernelArg(cl_struct.kernel, 5, sizeof(cl_mem), &deviceOutput);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg", err);

        }


        const size_t local[2] = {tile_size, tile_size};
        const size_t global[2] = {(X1.row + local[0] - 1) / local[0] * local[0],
                                  (X2.col + local[1] - 1) / local[1] * local[1]};
        err = clEnqueueNDRangeKernel(cl_struct.queue,
                                     cl_struct.kernel,
                                     2,
                                     NULL,
                                     global,
                                     local,
                                     0,
                                     NULL,
                                     NULL);

        if (err != CL_SUCCESS) {
            error_print("clEnqueueNDRangeKernel", err);
        }

        clFinish(cl_struct.queue);


        err = clEnqueueReadBuffer(cl_struct.queue, deviceOutput, CL_TRUE, 0, sizeof(float) * X1.row * X2.col,
                                  result.data, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clEnqueueReadBuffer dot", err);
        }

        cl_complete(cl_struct);
        clReleaseMemObject(deviceInput1);
        clReleaseMemObject(deviceInput2);
        clReleaseMemObject(deviceOutput);

    }
    return result;


}



Matrix inv(const Matrix &M)
{
    assert(M.row== M.row);
    const int n= M.row;
    Matrix result(n, n);
    result.data=new float[n*n];
    if(cpu&&cpu_inv) {
        Matrix JD(n, n * 2);
        JD.data = new float[JD.row * JD.col];
        float ratio;

        for (int i = 0; i < JD.row; i++) {
            for (int j = 0; j < JD.col; j++) {
                if (j < n) {
                    JD.data[INDEX(i, j, JD.col)] = M.data[INDEX(i, j, M.col)];
                } else if ((j - i) == n)
                    JD.data[INDEX(i, j, JD.col)] = 1;
                else {
                    JD.data[INDEX(i, j, JD.col)] = 0;
                }

            }

        }


        for (int i = 0; i < n; i++) {

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    ratio = JD.data[INDEX(j, i, JD.col)] / JD.data[INDEX(i, i, JD.col)];
                    for (int k = 1; k < 2 * n; k++)
                        JD.data[INDEX(j, k, JD.col)] =
                                JD.data[INDEX(j, k, JD.col)] - ratio * JD.data[INDEX(i, k, JD.col)];
                }
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = n; j < 2 * n; j++) {
                JD.data[INDEX(i, j, JD.col)] = JD.data[INDEX(i, j, JD.col)] / JD.data[INDEX(i, i, JD.col)];
                result.data[INDEX(i, j - n, result.col)] = JD.data[INDEX(i, j, JD.col)];
            }

        return result;
    }
    else {
        CL_struct cl_struct;
        Matrix I(n, n);
        I.data = new float[n * n];
        // init
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                if (i == j)
                    I.data[INDEX(i, j, n)] = 1;
                else
                    I.data[INDEX(i, j, n)] = 0;
                result.data[INDEX(i, j, n)] = M.data[INDEX(i, j, n)];
            }
        string option = "";
        cl_int err = NULL;
        string file = string("inv.cl");
        string name = string("nodiag_normalize");
        cl_init(cl_struct, file, name, option);
        const size_t local[2] = {tile_size, tile_size};
        const size_t global[2] = {(n + local[0] - 1) / local[0] * local[0], (n + local[1] - 1) / local[1] * local[1]};
        cl_kernel kernel2 = clCreateKernel(cl_struct.program, "diag_normalize", &err);
        if (!kernel2 || err != CL_SUCCESS) {
            error_print("diag_normalize 4", err);
        }
        cl_kernel kernel3 = clCreateKernel(cl_struct.program, "gaussjordan", &err);
        if (!kernel3 || err != CL_SUCCESS) {
            error_print("gaussjordan 4", err);
        }
        cl_kernel kernel4 = clCreateKernel(cl_struct.program, "set_zero", &err);
        if (!kernel4 || err != CL_SUCCESS) {
            error_print("set_zero 4", err);
        }


        cl_mem deviceInput = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);
        cl_mem deviceOutput = clCreateBuffer(cl_struct.context, CL_MEM_READ_WRITE, sizeof(float) * n * n, NULL, NULL);
        err = 0;
        err |= clSetKernelArg(cl_struct.kernel, 0, sizeof(cl_mem), &deviceInput);
        err |= clSetKernelArg(cl_struct.kernel, 1, sizeof(cl_mem), &deviceOutput);
        err |= clSetKernelArg(cl_struct.kernel, 2, sizeof(int), &n);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg 1", err);
        }
        err = 0;
        err |= clSetKernelArg(kernel2, 0, sizeof(cl_mem), &deviceInput);
        err |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &deviceOutput);
        err |= clSetKernelArg(kernel2, 2, sizeof(int), &n);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg 2", err);
        }
        err = 0;
        err |= clSetKernelArg(kernel3, 0, sizeof(cl_mem), &deviceInput);
        err |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &deviceOutput);
        err |= clSetKernelArg(kernel3, 2, sizeof(int), &n);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg 3", err);
        }
        err = 0;
        err |= clSetKernelArg(kernel4, 0, sizeof(cl_mem), &deviceInput);
        err |= clSetKernelArg(kernel4, 1, sizeof(cl_mem), &deviceOutput);
        err |= clSetKernelArg(kernel4, 2, sizeof(int), &n);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg 4", err);
        }
        for (int i = 0; i < n; i++) {
            clSetKernelArg(cl_struct.kernel, 3, sizeof(int), &i);
            clEnqueueNDRangeKernel(cl_struct.queue, cl_struct.kernel, 2, NULL, global, local, 0, NULL, NULL);

            clSetKernelArg(kernel2, 3, sizeof(int), &i);
            clEnqueueNDRangeKernel(cl_struct.queue, kernel2, 2, NULL, global, local, 0, NULL, NULL);

            clSetKernelArg(kernel3, 3, sizeof(int), &i);
            clEnqueueNDRangeKernel(cl_struct.queue, kernel3, 2, NULL, global, local, 0, NULL, NULL);

            clSetKernelArg(kernel4, 3, sizeof(int), &i);
            clEnqueueNDRangeKernel(cl_struct.queue, kernel4, 2, NULL, global, local, 0, NULL, NULL);
        }
        clFinish(cl_struct.queue);

        //@@ Read the results from the device
        err = clEnqueueReadBuffer(cl_struct.queue, deviceOutput, CL_TRUE, 0, sizeof(float) * n * n, result.data, 0,
                                  NULL, NULL);
        if (err != CL_SUCCESS) {
            error_print("clSetKernelArg inv", err);
        }
        cl_complete(cl_struct);
        clReleaseMemObject(deviceInput);
        clReleaseMemObject(deviceOutput);
        clReleaseKernel(kernel2);
        clReleaseKernel(kernel3);
        clReleaseKernel(kernel4);
        return result;
    }

}




Matrix right_inv(Matrix &M)
{
    Matrix MT=transpose(M);
    Matrix _MT=dot(MT,M);
    Matrix _MT_inv=inv(_MT);
    Matrix inverse=dot(_MT_inv,MT);
    return inverse;
//  A가 full row rank 가로로긴 ! 일반적으로 이러한 solution을 가질 것으로 예상됨
//
}









void sample_point(Matrix &X,vector<Variable> &variables)
{

    if (!X.data)
    {
        // 초기 시행
        X.data=new float [++X.row*X.col];
        for(int j=0;j<X.col;j++)
            X.data[INDEX(0,j,X.col)]=variables[j]();
    } else
    {
        float *temp=new float [++X.row*X.col];
        for(int i=0;i<X.row-1;i++)
            for(int j=0;j<X.col;j++)
                temp[INDEX(i,j,X.col)]=X.data[INDEX(i,j,X.col)];

        if(X.data)
            delete[]X.data;
        X.data=temp;
        for(int i=0;i<X.col;i++)
            X.data[INDEX(X.row-1,i,X.col)]=variables[i]();
    }
}
void sample_point(Matrix &X,float point1,float point2)
{


    float *temp=new float [++X.row*X.col];
    for(int i=0;i<X.row-1;i++)
        for(int j=0;j<X.col;j++)
            temp[INDEX(i,j,X.col)]=X.data[INDEX(i,j,X.col)];

    if(X.data)
        delete[]X.data;
    X.data=temp;
    X.data[INDEX(X.row-1,0,X.col)]=point1;
    X.data[INDEX(X.row-1,1,X.col)]=point2;
}
void sample_point(Matrix &X,float point1)
{


    float *temp=new float [++X.row*X.col];
    for(int i=0;i<X.row-1;i++)
        for(int j=0;j<X.col;j++)
            temp[INDEX(i,j,X.col)]=X.data[INDEX(i,j,X.col)];
    if(X.data)
        delete[]X.data;
    X.data=temp;
    X.data[INDEX(X.row-1,0,X.col)]=point1;
}

void sample_point(Matrix &Y,Matrix &X,vector<Variable> &variables)
{
    X.row++;
    Y.row++;
    if(!X.data)
    {
        X.data=new float [X.row*X.col];
    }
    else
    {
        float *temp=new float [X.row*X.col];
        for(int i=0;i<X.row-1;i++)
            for(int j=0;j<X.col;j++)
                temp[INDEX(i,j,X.col)]=X.data[INDEX(i,j,X.col)];
        if(X.data)
            delete[]X.data;
        X.data=temp;
    }
    for(int i=0;i<X.col;i++)
        X.data[INDEX(X.row-1,i,X.col)]=variables[i]();

    if(!Y.data)
    {
        Y.data=new float [Y.row*Y.col];
    }
    else
    {
        float *temp=new float [Y.row*Y.col];
        for(int i=0;i<Y.row-1;i++)
            for(int j=0;j<Y.col;j++)
                temp[INDEX(i,j,Y.col)]=Y.data[INDEX(i,j,Y.col)];
        if(Y.data)
            delete[]Y.data;
        Y.data=temp;
    }
    Y.data[INDEX(Y.row-1,0,Y.col)]=branin(X.data[INDEX(X.row-1,0,X.col)],X.data[INDEX(X.row-1,1,X.col)]);
}
Matrix standard(Matrix &M,const float & eps,float*&mu,float*&variance)
{
    if(mu)
        delete[]mu;
    if (variance)
        delete[]variance;
    mu=new float [M.col];
    for(int i=0;i<M.col;i++)
    {
        for(int j=0;j<M.row;j++)
            mu[i]+=M.data[INDEX(j,i,M.col)];
        mu[i]/=M.row;
    }

    variance=new float [M.col];
    for (int i=0;i<M.col;i++){
        for(int j = 0; j < M.row; j++) {
            variance[i] += powf(M.data[INDEX(j,i,M.col)] - mu[i], 2);
        }
        variance[i] /= M.row;
        variance[i] = sqrt(variance[i]);
    }
    Matrix temp(M.row,M.col);
    temp.data=new float [temp.row*temp.col];
    for(int i=0;i<M.row;i++)
        for(int j=0;j<M.col;j++)
        {
            temp.data[INDEX(i,j,temp.col)]=(M.data[INDEX(i,j,M.col)]-mu[j])/(eps+variance[j]);
        }

    return temp;
}
void standard2(Matrix &M,const float & eps)
{
    float * mu=new float [M.col];
    for(int i=0;i<M.col;i++)
    {
        for(int j=0;j<M.row;j++)
            mu[i]+=M.data[INDEX(j,i,M.col)];
        mu[i]/=M.row;
    }

    float * variance=new float [M.col];
    for (int i=0;i<M.col;i++){
        for(int j = 0; j < M.row; j++) {
            variance[i] += powf(M.data[INDEX(j,i,M.col)] - mu[i], 2);
        }
        variance[i] /= M.row;
        variance[i] = sqrt(variance[i]);
    }
    for(int i=0;i<M.row;i++)
        for(int j=0;j<M.col;j++)
        {
            M.data[INDEX(i,j,M.col)]=(M.data[INDEX(i,j,M.col)]-mu[j])/(eps+variance[j]);
        }

    delete []mu;
    delete []variance;
}
#endif //CL_UTILS_HPP
