//
// Created by jaehun on 20. 12. 17..
//

#ifndef CL_ETC_H
#define CL_ETC_H
#include <fstream>
#include <string>
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <cassert>
#include <CL/opencl.h>
#include <cstring>
using namespace std;
#define INDEX(i,j,col) (col)*(i)+(j) // Row Major
struct CL_struct{
    // opencl object
     cl_context context =NULL;
     cl_command_queue queue =NULL;
     cl_program program=NULL;
     cl_kernel kernel=NULL;
};

float branin(float x0, float x1)
{
    // objective function
    // https://www.sfu.ca/~ssurjano/branin.html
    const float b = 0.12918450914398066f;
    const float c = 1.5915494309189535f;
    const float t =  0.039788735772973836f;
    const float u = x1 - b * x0* x0  + c * x0 - 6;
    const float r = 10. * (1. - t) * cosf(x0) + 10;
    float Z = u * u + r;
    return Z;
}

class Variable
{
    // sample variable
    // value with in bound
    // sampling using uniform dist
private:

    float sample_value() const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> distribution(this->min_bound,this->max_bound);
        return distribution(gen);
    }
    float value=0;
    std::string name="";
    float min_bound=0;
    float max_bound=0;
public:
    Variable(float value,std::string name, float min_bound,float max_bound)
            :value(value), name(name),min_bound(min_bound),max_bound(max_bound)
    {

    }
    float operator()() const
    {
        return sample_value();
    }
};

typedef struct _matrix {
    // matrix struct
    // 1d array
    // using row major
    // assign operator and copy operator was implemented.
    int row=NULL;
    int col=NULL;
    float* data= nullptr;
    _matrix(){}
    _matrix(int row,int col):row(row),col(col)
    {

    }
    ~_matrix()
    {
        delete[]data;
    }

    _matrix(const _matrix &M)
    {
        this->row=M.row;
        this->col=M.row;
        this->data=new float [this->row* this->col];
        for(int i=0;i<this->row* this->col;i++)
            this->data[i]=M.data[i];
    }
    _matrix& operator=(const _matrix&M)
    {
        this->row=M.row;
        this->col=M.row;
        this->data=new float [this->row* this->col];
        for(int i=0;i<this->row* this->col;i++)
            this->data[i]=M.data[i];
    }

} Matrix;
void sample_point(Matrix &X,vector<Variable> &variables);
void sample_point(Matrix &Y,Matrix &X,vector<Variable> &variables);
Matrix standard(Matrix &M,const float & eps,float*&mu,float*&variance);
void standard2(Matrix &M,const float & eps);
Matrix right_inv(Matrix &M);
Matrix inv(const Matrix &M);
Matrix dot(const Matrix &X1,const Matrix &X2);
Matrix transpose(const Matrix &M);
void sample_point(Matrix &X,float point1,float point2);
void sample_point(Matrix &X,float point1);
void cl_init( CL_struct&cl_struct,string  file,string  name,string  option);
const char *getErrorString(cl_int error);
std::string readStringFromFile(std::string filename );
void error_print(const char * meg,cl_int err_code);
void cl_complete( CL_struct&cl_struct);



#endif //CL_ETC_H
