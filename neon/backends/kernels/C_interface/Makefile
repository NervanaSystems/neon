NVCC=nvcc

test: test.cu libnervana.a
	$(NVCC) -o test test.cu -L . -lnervana -lcuda -std=c++11

libnervana.a: nervana_c_api.cu nervana_c_api.h
	$(NVCC) -c nervana_c_api.cu -std=c++11 -Xcompiler -fPIC
	ar rvs libnervana.a nervana_c_api.o

clean:
	rm -rf *.o test libnervana.a
