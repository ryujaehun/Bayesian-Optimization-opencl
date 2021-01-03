
//#define TS 4                        // The square-root of the 2D tile-size (== work-group dims)


#define WPT TS/2                        // The amount of work-per-thread, i.e. the thread-coarsening factor
#define RTS (TS/WPT)                 // The reduced tile-size in one dimension

// Macros for host and kernel code
#define MIN(a,b) ((a) > (b)) ? (b) : (a)
#define MAX(a,b) ((a) > (b)) ? (a) : (b)
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y))
#define MOD2(x,y) ((x) % (y))
#define DIV2(x,y) ((x) / (y))
__kernel void gemm(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

       const int row = get_local_id(0); // Local row ID (max: TS)
       const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
       const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
       const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

       // Local memory to fit a tile of TS*TS elements of A and B
       __local float Asub[TS][TS];
       __local float Bsub[TS][TS];

       // Initialise the accumulation registers
       float acc[WPT];
       for (int w=0; w<WPT; w++) {
           acc[w] = 0.0f;
       }

       // Loop over all tiles
       const int numTiles = CEIL_DIV(K,TS);
       for (int t=0; t<numTiles; t++) {

           // Load one tile of A and B into local memory
           for (int w=0; w<WPT; w++) {
               const int tiledRow = TS*t + row;
               const int tiledCol = TS*t + col;
               Asub[col + w*RTS][row] = A[(tiledCol + w*RTS) + globalRow*K];
               Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)+ tiledRow*N];
           }

           // Synchronise to make sure the tile is loaded
           barrier(CLK_LOCAL_MEM_FENCE);

           // Perform the computation for a single tile
           for (int k=0; k<TS; k++) {
               for (int w=0; w<WPT; w++) {
                   acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
               }
           }

           // Synchronise before loading the next tile
           barrier(CLK_LOCAL_MEM_FENCE);
       }

       // Store the final results in C
       for (int w=0; w<WPT; w++) {
           C[(globalCol + w*RTS)+  globalRow*N] = acc[w];
       }
}