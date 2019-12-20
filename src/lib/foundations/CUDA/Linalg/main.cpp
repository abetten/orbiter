#ifdef __unix__
# include <unistd.h>
#elif defined _WIN32
# include <windows.h>
#define sleep(x) Sleep(1000 * (x))
#endif

#include <iostream>
#include <iomanip>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include "linalg.h"
#include "gpu_table.h"

using std::cout;
using std::endl;



int main () {

	size_t perspective_size=10;
	size_t x = pow(2, 23), y = pow(2, 5), z = pow(2, 5);

	linalg::Matrix<int> M1 (x, y);
	linalg::Matrix<int> M2 (y, z);
	linalg::Matrix<int> M3 (x, z);
	linalg::Matrix<int> Mcpu(x,z);
	linalg::Matrix<int> McpuP(x,z);


	linalg::random_fill(M1, 0, 9);
	linalg::random_fill(M2, 0, 9);

//	linalg::print(M1); cout << endl; linalg::print(M2); cout << endl;


//	goto cuda_exec_1; //
//	goto cuda_exec_2; // 1.25

	printf("Starting matrix multiplication\n");

	clock_t t;
//	t = clock();
//	linalg::cpu_mat_mul(M1, M2, Mcpu, 0, 0, false);
//	printf("non-tiling time: %f sec\n", ((clock()-t)/(float)CLOCKS_PER_SEC));
//
//	t = clock();
//	linalg::mat_mul_block(M1, M2, McpuP);
//	printf("tiling time: %f sec\n", ((clock()-t)/(float)CLOCKS_PER_SEC));

//	(McpuP == Mcpu) ? printf("PASS\n") : printf("FAIL\n");
//	Mcpu.perspective(perspective_size, perspective_size);
//	linalg::print(Mcpu); cout << endl;
//	McpuP.perspective(perspective_size, perspective_size);
//	linalg::print(McpuP); cout << endl;
//	exit(0);


//	cuda_exec_1:
	t=clock();
	linalg::cuda_mat_mul_stream(8, M1, M2, M3);
	printf("cuda time: %f sec\n", ((clock()-t)/(float)CLOCKS_PER_SEC));
//	exit (0);


//	M3.perspective(perspective_size, perspective_size);
//	linalg::print(M3);
//	M3.reset_dimensions();
	(M3 == McpuP) ? printf("PASS\n") : printf("FAIL\n");

}
