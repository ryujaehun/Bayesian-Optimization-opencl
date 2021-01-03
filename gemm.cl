
__kernel void gemm(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    
    const int row = get_local_id(0); 
    const int col = get_local_id(1); 
    const int globalRow = TS*get_group_id(0) + row; 
    const int globalCol = TS*get_group_id(1) + col; 
 
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    float acc = 0.0f;
    
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        const int tiledRow = TS*t + row;
        const int tiledCol = TS*t + col;
        Asub[col][row] = A[tiledCol*M + globalRow];
        Bsub[col][row] = B[globalCol*K + tiledRow];
 
        barrier(CLK_LOCAL_MEM_FENCE);
 
        for (int k=0; k<TS; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}