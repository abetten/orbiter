// projective_space2.C
// 

#include "orbiter.h"


int main(int argc, char **argv)
{
	finite_field *F;
	projective_space *P;
	int f_init_incidence_structure = TRUE;
	int verbose_level = 0;
	int i;
	int q = 4;
	int n = 2; // projective dimension

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);


	P = NEW_OBJECT(projective_space);
	P->init(n, F, 
		f_init_incidence_structure, 
		verbose_level);

	for (i = 0; i < P->N_lines; i++) {

		cout << "Line " << setw(3) << i << " is ";
		int_vec_print(cout, P->Lines + i * P->k, P->k);
		cout << endl;
		}

	FREE_OBJECT(P);
	FREE_OBJECT(F);
}


