// k_arc_generator_main.C
// 
// Anton Betten
//
// started May 14, 2018
//
// Searches for (k,d)-arcs in desarguesian projective planes
//
//

#include "orbiter.h"

// global data:

int t0; // the system time when the program started


int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	
	{
	int f_d = FALSE;
	int d = 0;
	int f_q = FALSE;
	int q = 0;
	int f_sz = FALSE;
	int sz = 0;
	int i;
	int verbose_level = 0;

	for (i = 1; i < argc; i++) {
		
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
		else if (strcmp(argv[i], "-sz") == 0) {
			f_sz = TRUE;
			sz = atoi(argv[++i]);
			cout << "-sz " << sz << endl;
			}
		}

	if (!f_d) {
		cout << "please use option -d <d>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_sz) {
		cout << "please use option -sz <sz>" << endl;
		exit(1);
		}

	finite_field *F;
	projective_space *P2;
	k_arc_generator *K;

	cout << "before creating the finite field" << endl;
	F = NEW_OBJECT(finite_field);
	P2 = NEW_OBJECT(projective_space);


	F->init(q, 0);
	
	P2->init(2, F, 
		TRUE /* f_init_incidence_structure */, 
		verbose_level);

	

	K = NEW_OBJECT(k_arc_generator);

	K->init(F, P2, 
		d, sz, 
		argc, argv, 
		verbose_level);


	FREE_OBJECT(K);
	FREE_OBJECT(P2);
	FREE_OBJECT(F);

	}
	cout << "Memory usage = " << os_memory_usage()
			<<  " Time = " << delta_time(t0)
			<< " tps = " << os_ticks_per_second() << endl;
	the_end(t0);
	//the_end_quietly(t0);
	
}


