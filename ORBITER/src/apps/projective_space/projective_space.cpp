// projective_space.C
// 

#include "orbiter.h"

using namespace orbiter;


int main(int argc, char **argv)
{
	finite_field *F;
	int verbose_level = 0;
	int i, j, N;
	int q = 4;
	int n = 2; // projective dimension
	int *v;

	F = NEW_OBJECT(finite_field);
	F->init(q, verbose_level);

	
	N = F->nb_points_in_PG(n);

	v = NEW_int(n + 1);
	for (i = 0; i < N; i++) {
		F->unrank_point_in_PG(v, n + 1, i);
		j = F->rank_point_in_PG(v, n + 1);
		cout << setw(3) << i << " : ";
		int_vec_print(cout, v, n + 1);
		cout << " : " << j << endl;
		if (j != i) {
			cout << "j != i, this should not happen" << endl;
			exit(1);
			}
		}
	FREE_int(v);
	FREE_OBJECT(F);
}


