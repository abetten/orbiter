// points.C
// 
// Anton Betten
// 2/18/2011
//
// ranks and unranks the points in PG(n-1,q)
// 
//
//

#include "orbiter.h"
#include "discreta.h"


// global data:

INT t0; // the system time when the program started

void points(INT n, INT q, INT verbose_level);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_d = FALSE;
	INT d = 0;
	INT f_q = FALSE;
	INT q;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	if (!f_d) {
		cout << "please use -d option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}
	points(d, q, verbose_level);
	
	INT dt;
	dt = delta_time(t0);

	cout << "time in ticks " << dt << " tps=" << os_ticks_per_second() << endl;

	the_end(t0);
}

void points(INT d, INT q, INT verbose_level)
{
	INT N_points, n, i, j;
	finite_field F;
	INT *v;
	
	n = d + 1;
	v = NEW_INT(n);

	F.init(q, 0);
	
	N_points = generalized_binomial(n, 1, q);
	cout << "number of points = " << N_points << endl;
	for (i = 0; i < N_points; i++) {
		PG_element_unrank_modified(F, v, 1, n, i);
#if 0
		cout << "point " << i << " : ";
		INT_vec_print(cout, v, n);
		cout << " = ";
		PG_element_normalize_from_front(F, v, 1, n);
		INT_vec_print(cout, v, n);
		cout << endl;
#endif
		PG_element_rank_modified(F, v, 1, n, j);
		if (j != i) {
			cout << "error: i != j" << endl;
			exit(1);
			}
		}

	FREE_INT(v);
}

