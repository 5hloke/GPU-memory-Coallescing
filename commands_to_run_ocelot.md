g++ -o tracegen tracegen1.o trace.o -locelot
g++ -std=c++0x trace.cpp -c -l
nvcc -c -arch=sm_20 tracegen1.cu
