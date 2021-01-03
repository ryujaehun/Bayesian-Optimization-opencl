__kernel void matrix_square(__global float* input,__global float* output,int col,__local float *temp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int t=get_local_id(1);

    int idx=x*col+y;
    temp[t]=input[idx]*input[idx];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (t==0)
    {
    for (int i=0;i<col;i++)
        output[x]+=temp[i];
    }
}
__kernel void vector_add(__global float* A,__global float* B,__global float* C)
{
    int idx = get_global_id(0);
    C[idx]=A[idx]+B[idx];
}
__kernel void matrix_square_add(__global float* input,__global float* input2,__global float* output,int col,__local float *temp)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int t=get_local_id(1);

    int idx=x*col+y;
    temp[t]=input[idx]*input[idx]+input2[idx]*input2[idx];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (t==0)
    {
    for (int i=0;i<col;i++)
        output[x]+=temp[i];
    }
}
__kernel void matrix_vector_add(__global float* A,__global float* B,__global float* C, int col)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);
    int index=idx*col+idy;
    C[index]=exp(2*A[index]-B[idx]);
}