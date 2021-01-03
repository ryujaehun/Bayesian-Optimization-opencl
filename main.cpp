
#include <CL/opencl.h>
#include "utils.hpp"
#include <iostream>
#include <string>
#include <chrono>
extern bool cpu;
extern bool cpu_kernel;
extern bool cpu_transpose;
extern bool cpu_inv;
extern bool cpu_dot;
extern size_t tile_size;
using namespace std;
int main(int argc,char **argv) {



//    if (argc < 3)
//    {
//        cout << "Usage: " << argv[0] << " size iteration ratio tile\n";
//        exit(1);
//    }

    int flag=atoi(argv[1]);
    int size=atoi(argv[2]);

    //int size=2048;
    int iteration=700;
    int ratio=180;
    tile_size=4;
    vector<Variable> variables;
    auto t1 = std::chrono::high_resolution_clock::now();
    variables.push_back(Variable(0, string("X1"), -5, 10));
    variables.push_back(Variable(0, string("X2"), 0, 15));
    float*x_obj_mu= nullptr;
    float *x_obj_std= nullptr;
    float*y_obj_mu= nullptr;
    float *y_obj_std= nullptr;

    Matrix X(0,2);
    Matrix Y(0,1);
    float EPS = 1e-8;
    branin(1, 1);
    GP gp( 0.0);
    for (int i = 0; i < 5; i++)
        sample_point(Y, X, variables);

    Matrix X_stand=standard(X, EPS,x_obj_mu,x_obj_std);
    Matrix Y_stand=standard(Y, EPS,y_obj_mu,y_obj_std);

    gp.fit(X_stand,  Y_stand);

    Matrix  min_obj_X = X;
    Matrix  min_obj_Y = Y;


    switch ( flag ) {
        case 1:
            cpu= false;
            break;
        case 2:
            //with rbf 
            cpu_kernel= false;
            break;
        case 3:
            //with transpose
            cpu_transpose= false;
            break;
        case 4:
            //with inverse
            cpu_inv= false;
            break;
        case 5:
            //with gemm
            cpu_dot= false;
            break;
        case 6:
            //without rbf
            cpu_dot= false;
            cpu_transpose= false;
            cpu_inv= false;
            break;
        case 7:
            //without transpose
            cpu_dot= false;
            cpu_kernel= false;
            cpu_inv= false;
            break;
        case 8:
            //without inverse
            cpu_dot= false;
            cpu_kernel= false;
            cpu_transpose= false;
            break;
        case 9:
            //without gemm
            cpu_transpose= false;
            cpu_kernel= false;
            cpu_inv= false;
        default:
            break;
    }
    for (int i = 0; i < iteration; i++) {
        auto in1 = std::chrono::high_resolution_clock::now();



        Matrix sample(0,2);
        for (int j = 0; j < size; j++) {
            sample_point(sample, variables);
        }
        standard2(sample,EPS);
        Matrix mu=gp.predict(sample);
        int min_index=0;
        float min_value=mu.data[0];
        for(int k=1;k<mu.row;k++)
            if(mu.data[k]<min_value)
            {
                min_index=k;
                min_value=mu.data[k];
            }
        // indexing
        float X1_min = sample.data[INDEX(min_index,0,sample.col)];
        float X2_min = sample.data[INDEX(min_index,1,sample.col)];
        X1_min=X1_min*(x_obj_std[0] + EPS) + x_obj_mu[0];
        X2_min=X2_min*(x_obj_std[1] + EPS) + x_obj_mu[1];
        sample_point(X,X1_min,X2_min);
        float new_obj_val=branin(X1_min,X2_min);
//        cout<<i<<" X1 min : "<<X1_min<<"\t"<<" X2 min : "<<X2_min<<"\t"<<" value min : "<<new_obj_val<<endl;

        sample_point(Y,new_obj_val);

        Matrix X_stand_2=standard(X, EPS,x_obj_mu,x_obj_std);
        Matrix Y_stand_2=standard(Y, EPS,y_obj_mu,y_obj_std);
        gp.fit(X_stand_2,  Y_stand_2);
        auto in2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds >( in2 - in1 ).count();
        cout<<i<<" "<<duration2<<endl;
    }


    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds >( t2 - t1 ).count();

    std::cout <<size<<" "<<iteration<<" "<<ratio<<" "<<tile_size<<" "<< duration;



    return 0;
}
