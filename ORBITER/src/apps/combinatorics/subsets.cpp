// subsets.C
// 
// Anton Betten
// Oct 9, 2015
//
//
// lists all subsets of an N-element set in lexicographic order
//
//

#include <iostream>

using namespace std;

#define N 8

int v[1000];
int cnt = 1;

void list(int n, int d, int a);

int main(int argc, const char **argv)
{
	list(N, 0, 1);
}


void list(int n, int d, int a)
{
	int i, h;

	
	// print out the current set:
	if (d == 0) {
		cout << cnt << " : empty set" << endl;
		cnt++;
		}
	else {
		cout << cnt << " : ";
		for (h = 0; h < d; h++) {
			cout << v[h];
			if (h < d - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		cnt++;
		}

	// depth first search:

	for (i = a; i <= n; i++) {
		v[d] = i;
		list(n, d + 1, i + 1);
		}
}

