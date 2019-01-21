// finite_field.C
// 

#include "orbiter.h"

using namespace orbiter;

int main(int argc, char **argv)
{
	finite_field *F;
	int verbose_level = 0;
	int i, j, a;
	int q = 4;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	cout << "addition table of GF(" << q << "):" << endl;
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			
			a = F->add(i, j);

			cout << setw(2) << a << " ";
			}
		cout << endl;
		}

	cout << "multiplication table of GF(" << q << "):" << endl;
	for (i = 1; i < 4; i++) {
		for (j = 1; j < 4; j++) {
			
			a = F->mult(i, j);

			cout << setw(2) << a << " ";
			}
		cout << endl;
		}

	FREE_OBJECT(F);
}


