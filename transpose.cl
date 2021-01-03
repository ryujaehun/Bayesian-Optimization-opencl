__kernel void transpose(__global float *odata, __global float *idata,  int width, int height, __local float* block)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex  < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex + height * xIndex ;
        block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
    if((xIndex < height) && (yIndex  < width))
    {
        unsigned int index_out = yIndex + width * xIndex;
        odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
    }
}