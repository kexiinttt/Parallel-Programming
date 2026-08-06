#include <wb.h>

//@@ Insert code to implement vector addition here

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
  
    args = wbArg_read(argc, argv);
  
    hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    //@@ Allocate GPU memory here

    //@@ Copy memory to the GPU here

    //@@ Initialize the grid and block dimensions here

    //@@ Launch the GPU Kernel here

    //@@ Copy the GPU memory back to the CPU here

    //@@ Free the GPU memory here

    return 0;
}
