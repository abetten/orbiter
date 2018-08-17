// Anton Betten, September 8 2008
//
// count the number of integers between 1 and 10^n 
// whose digits are nondecreasing

#include <iostream>
#include <iomanip>

using namespace std;

int nb_sol;


int count(int n);
void count_recursion(int n, int *digits, int depth);


int main(int argc, char **argv)
{
	int i;
	
	for (i = 0; i <= 15; i++) {
		nb_sol = 0;
		count(i);
		cout << setw(3) << i << " : " << setw(10) << nb_sol - 1 << endl;
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
		f = 0;
	else
		f = digits[depth - 1];
	for (i = f; i <= 9; i++) {
		digits[depth] = i;
		count_recursion(n, digits, depth + 1);
		}

}
