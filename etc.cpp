//
// Created by jaehun on 20. 12. 20..
//
#include "etc.h"

void cl_complete(CL_struct&cl_struct)
{
    clReleaseProgram(cl_struct.program);
    clReleaseKernel(cl_struct.kernel);
    clReleaseCommandQueue(cl_struct.queue);
    clReleaseContext(cl_struct.context);
}

void cl_init(
        CL_struct&cl_struct,
        string  file,
        string  name,
        string  option
)
{
    cl_platform_id cl_platform; // OpenCL platform
    cl_device_id device_id;
    cl_uint numDevice, numPlatform;
    cl_int err=0;
    err = clGetPlatformIDs(1, &cl_platform, &numPlatform); // get platform id and number
    if (err != CL_SUCCESS)
    {
    error_print("clGetPlatformIDs",err);

    }
    //@@ Get ID for the device
    err = clGetDeviceIDs(cl_platform, CL_DEVICE_TYPE_GPU, 1, &device_id,  &numDevice);
    if (err != CL_SUCCESS)
    {
    error_print("clGetDeviceIDs",err);
    }
    //@@ Create a context
    cl_struct.context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
    error_print("clCreateContext",err);
    }
    //@@ Create a command queue
    cl_struct.queue = clCreateCommandQueue(cl_struct.context, device_id, 0, &err);
    if (err != CL_SUCCESS)
    {
    error_print("clCreateCommandQueue",err);
    }
    string filename= "/home/jaehun/CLionProjects/cl.bak/"+file;
    string source=readStringFromFile(filename);

    const char * src=source.c_str();
    cl_struct.program=clCreateProgramWithSource(cl_struct.context,
                                                1,&src , NULL, &err);
    if (err != CL_SUCCESS)
    {
    error_print("clCreateProgramWithSource",err);
    }
    err=clBuildProgram(cl_struct.program, 0, NULL,option.c_str() , NULL, NULL);
    if (err != CL_SUCCESS)
    {
    error_print("clBuildProgram",err);
    }
    const char* kernel_name=name.c_str();
    cl_struct.kernel = clCreateKernel(cl_struct.program, kernel_name, &err);
    if (!cl_struct.kernel || err != CL_SUCCESS)
    {
    error_print("clCreateKernel",err);
    }

}


const char *getErrorString(cl_int error)
{
    /*
    *  Purpose
    ------- Translate error code to Error Msg

    Arguments
    ---------
    @param[in]
    error       cl_int
    error flag
    */
    switch(error){

        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}


void error_print(const char * meg,cl_int err_code)
{
    /*
     *  Purpose
    ------- Print Error meg and code

    Arguments
    ---------
    @param[in]
    meg       const char *
    Message to print
    err_code       cl_code
    Error code
     */
    cerr<<"Error : "<<meg<<"\t"<<getErrorString(err_code)<<endl;
}



std::string readStringFromFile(
        std::string filename )
{
    /*
     *  Purpose
    ------- Translate kernel File string to string object

    Arguments
    ---------
    @param[in]
    filename       std::string
    Absolute path of Kernel File
     */
    std::ifstream is(filename, std::ios::binary);
    if (!is.good()) {
        printf("Couldn't open file '%s'!\n", filename.c_str());
        return "";
    }

    size_t filesize = 0;
    is.seekg(0, std::ios::end);
    filesize = (size_t)is.tellg();
    is.seekg(0, std::ios::beg);

    std::string source{
            std::istreambuf_iterator<char>(is),
            std::istreambuf_iterator<char>() };
    return source;
}
