# Instruction

In this MP, you are supposed to achieve a simple vector addition kernel and its associated host code.

The program reads whitespace-separated floating-point values from three ordinary text files:

```text
input1.txt   # first input vector
input2.txt   # second input vector
expected.txt # expected vector (input1 + input2)
```

Test the code by running:
```bash
nvcc solution.cu -o solution
./solution input1.txt input2.txt expected.txt
```
