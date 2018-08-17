// partitions.C
//
// Anton Betten
// Dec 16, 2009

#include <iostream>
#include <iomanip>
#include <stdlib.h>

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

using namespace std;

int next_partition(int n, int *part);

int main(int argc, char **argv)
{
	int n, cnt, i, j, a;

	if (argc < 1) {
		cout << "usage: part.out <n>" << endl;
		exit(1);
		}
	n = atoi(argv[1]);

	int *part;

	part = new int[n + 1];
	
	for (i = 0; i <= n; i++)
		part[i] = 0;
	part[n] = 1;


	cnt = 1;
	
	while (TRUE) {
		cout << setw(4) << cnt << " : ";
		for (i = n; i >= 1; i--) {
			a = part[i];
			for (j = 0; j < a; j++) {
				cout << i << " ";
				}
			}
		cout << endl;

	
		if (!next_partition(n, part))
			break;
		cnt++;
		}
	
}

int next_partition(int n, int *part)
{
	int s, i, j, q, r;

	s = part[1];
	for (i = 2; i <= n; i++) {
		if (part[i]) {
			s += i;
			part[i]--;
			break;
			}
		}
	if (i == n + 1) {
		return FALSE;
		}
	//cout << "s=" << s << endl;
	for (j = i - 1; j >= 1; j--) {
		q = s / j;
		r = s - q * j;
		part[j] = q;
		s = r;
		}
	return TRUE;
}



