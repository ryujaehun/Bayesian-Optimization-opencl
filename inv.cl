__kernel void nodiag_normalize(__global float *A,__global float *I, int n, int i){
	// kernel 1
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= (A[i*n + i]+1e-10);
		A[x*n + y] /= (A[i*n + i]+1e-10);
	}

}

__kernel void diag_normalize(__global float *A,__global float *I, int n, int i){
	// kernel 2
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= (A[i*n + i]+1e-10);
		A[x*n + y] /= (A[i*n + i]+1e-10);
	}

}

__kernel void gaussjordan(__global float *A,__global float *I, int n, int i)
{
    // kernel 3
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}
		}
	}

}

__kernel void set_zero(__global float *A,__global float *I, int n, int i){
    // kernel 4
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}
