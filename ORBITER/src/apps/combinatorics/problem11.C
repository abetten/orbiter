// Anton Betten, September 9 2008
//
// count the number of even integers between 1 and 10^n 
// with no repeated digits

#include <iostream>
#include <iomanip>

using namespace std;

int nb_sol;


int count(int n);
void count_recursion(int n, int *digits, int depth);


int main(int argc, char **argv)
{
	int i, n;
	
	for (i = 0; i <= 15; i++) {
		nb_sol = 0;
		count(i);
		cout << setw(3) << i << " : " << setw(10) << nb_sol << endl;
		}
}



int count(int n)
{
	int *digits;
	
	digits = new int[n];
	nb_sol = 0;
	count_recursion(n, digits, 0);
	
	delete [] digits;
	
	return nb_sol;
}

void count_recursion(int n, int *digits, int depth)
{
	int i, f;
	
	if (depth == n) {
		if ((digits[n - 1] % 2) == 1)
			return;
		nb_sol++;
#if 0
		cout << setw(10) << nb_sol << " : ";
		for (i = 0; i < depth; i++) {
			cout << digits[i];
			}
		cout << endl;
#endif
		return;
		}
	if (depth == 0)
		f = 1;
	else
		f = 0;
	for (i = f; i <= 9; i++) {
		if (depth) {
			if (digits[depth - 1] == i)
				continue;
			}
		digits[depth] = i;
		count_recursion(n, digits, depth + 1);
		}

}
