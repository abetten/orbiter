// Anton Betten, September 14 2008
//
// 

#include <iostream>
#include <iomanip>

using namespace std;

#define N 30

int F[N];
int S[N];

#define TRUE 1
#define FALSE 0

void compute_fibonacci(int verbose_level);
void convert(int n, int verbose_level);
void convert_to_fibonacci(int n, int *S, int &i_max, int verbose_level);
void print_number(int *S, int i_max);



int main(int argc, char **argv)
{
	int verbose_level = 1;
	int i;
	
	compute_fibonacci(verbose_level);


	convert(202, verbose_level);
	convert(105, verbose_level);
	convert(128, verbose_level);
	convert(243, verbose_level);

	for (i = 1; i < 200; i++) {
		convert(i, verbose_level);
		}
}

void compute_fibonacci(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	F[0] = 0;
	F[1] = 1;
	for (i = 0; i < N; i++) {
		if (i >= 2) {
			F[i] = F[i - 1] + F[i - 2];
			}
		if (f_v) {
			cout << setw(3) << i << " : " << setw(10) << F[i] << endl;
			}
		}
}

void convert(int n, int verbose_level)
{
	int i_max;
	
	convert_to_fibonacci(n, S, i_max, verbose_level);
	cout << n << " = ";
	print_number(S, i_max);
}

void convert_to_fibonacci(int n, int *S, int &i_max, int verbose_level)
{
	int i, f_first = TRUE;
	
	for (i = 0; i < N; i++) {
		S[i] = 0;
		}
	while (n) {
		for (i = 0; i < N; i++) {
			if (F[i] > n) {
				break;
				}
			}
		if (f_first) {
			i_max = i - 1;
			f_first = FALSE;
			}
		S[i - 1] = 1;
		n -= F[i - 1];
		}
}

void print_number(int *S, int i_max)
{
	int i;
		
	for (i = 0; i <= i_max; i++) {
		if (S[i]) {
			cout << "+ " << F[i];
			}
		}
	cout << endl;
}

