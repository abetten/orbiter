// packing.C
// Anton Betten
//
// started:  Feb 1 2008

#include "orbiter.h"

#define N 100

int t0;

int main(int argc, char **argv)
{
	t0 = os_ticks();
	int *P;
	int i, j, a, b, f_one_seen;
	
	a = maxfit(35, 7);
	b = maxfit(35, 8);
	cout << "maxfit 35 7 = " << a << endl;
	cout << "maxfit 35 8 = " << b << endl;
	P = new int[N * N];
	for (i = N - 1; i >= 1; i--) {
		for (j = i; j >= 2; j--) {
			a = TDO_upper_bound(i, j);
			P[(i - 1) * N + j - 1] = a;
			}
		}
	cout << " & ";
	for (j = 1; j <= N; j++) {
		cout << setw(4) << j << " & ";
		}
	cout << "\\\\" << endl;
	
	for (i = 1; i <= N; i++) {
		cout << setw(3) << i << " & ";
		f_one_seen = FALSE;
		for (j = 1; j <= N; j++) {
			if (j <= i) {
				a = P[(i - 1) * N + j - 1];
				if (a == 1) {
					if (f_one_seen) {
						cout << "     & ";
						}
					else {
						cout << setw(4) << a << " & ";
						}
					f_one_seen = TRUE;
					}
				else {
					cout << setw(4) << a << " & ";
					}
				}
			else {
				cout << "     & ";
				}
			}
		cout << "\\\\" << endl;
		}
}
