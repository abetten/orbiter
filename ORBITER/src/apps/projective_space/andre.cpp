// andre.cpp
// 
// Anton Betten
// July 11, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;




// global data:

int t0; // the system time when the program started

void do_it(int q, int k, int no,
		int f_Fano, int f_arcs, int f_depth, int depth,
		int verbose_level);

int main(int argc, const char **argv)
{
	int i;
	int verbose_level = 0;
	int f_q = FALSE;
	int q = 0;
	int f_k = FALSE;
	int k = 0;
	int f_no = FALSE;
	int no = 0;
	int f_Fano = FALSE;
	int f_depth = FALSE;
	int depth = 0;
	int f_arcs = FALSE;
	
	t0 = os_ticks();
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-no") == 0) {
			f_no = TRUE;
			no = atoi(argv[++i]);
			cout << "-no " << no << endl;
			}
		else if (strcmp(argv[i], "-Fano") == 0) {
			f_Fano = TRUE;
			cout << "-Fano " << endl;
			}
		else if (strcmp(argv[i], "-arcs") == 0) {
			f_arcs = TRUE;
			cout << "-arcs " << endl;
			}
		else if (strcmp(argv[i], "-depth") == 0) {
			f_depth = TRUE;
			depth = atoi(argv[++i]);
			cout << "-depth " << depth << endl;
			}
		}
	
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_k) {
		cout << "please use option -k <k>" << endl;
		exit(1);
		}
	if (!f_no) {
		cout << "please use option -no <no>" << endl;
		exit(1);
		}
	do_it(q, k, no, f_Fano, f_arcs, f_depth, depth, verbose_level);

	the_end_quietly(t0);
}


void do_it(int q, int k, int no,
		int f_Fano, int f_arcs, int f_depth, int depth,
		int verbose_level)
{
	int *spread_elements_numeric; // do not free
	int n;
	action *An;
	vector_ge *gens;
	translation_plane_via_andre_model *Andre;
	finite_field *F;
	matrix_group *M; // do not free

	int f_basis = FALSE;
	int f_semilinear = FALSE;
	//int i;

	const char *stab_order;
	longinteger_object stab_go;
	int order_of_plane;
	number_theory_domain NT;
	knowledge_base K;


	//n = Andre->n;
	//n1 = n + 1;

	order_of_plane = NT.i_power_j(q, k);
	
	f_semilinear = FALSE;
	if (!NT.is_prime(q)) {
		f_semilinear = TRUE;
		}


	n = 2 * k;
	int sz;
	vector_ge *nice_gens;

	spread_elements_numeric = K.Spread_representative(q, k, no, sz);

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);
	
	An = NEW_OBJECT(action);
	An->init_projective_group(n, F, f_semilinear, 
		f_basis,
		nice_gens,
		0 /*verbose_level*/);
	FREE_OBJECT(nice_gens);

	M = An->G.matrix_grp;


#if 0
	int *data; // do not free
	int nb_gens, data_size;
	TP_stab_gens(order_of_plane, no, data, nb_gens, data_size, stab_order);
	gens = new vector_ge;
	gens->init(An);
	gens->allocate(nb_gens);
	cout << "Creating stabilizer generators:" << endl;
	for (i = 0; i < nb_gens; i++) {
		An->make_element(gens->ith(i), data + i * data_size, 0 /*verbose_level*/);
		}
#else
	An->stabilizer_of_spread_representative(q, k, no,
			gens, stab_order, verbose_level);

	stab_go.create_from_base_10_string(stab_order, 0 /* verbose_level */);
#endif

	cout << "Spread stabilizer has order " << stab_go << endl;

	Andre = NEW_OBJECT(translation_plane_via_andre_model);

	Andre->init(spread_elements_numeric, k, F, 
		gens /*spread_stab_gens*/, stab_go, verbose_level);


	if (f_Fano) {
		char prefix[1000];
		int nb_subplanes;

		sprintf(prefix, "Fano_TP_%d_", no);

		Andre->classify_subplanes(prefix, verbose_level);

		int target_depth;

		if (f_depth) {
			target_depth = depth;
			}
		else {
			target_depth = 7;
			}

		nb_subplanes = Andre->arcs->nb_orbits_at_level(target_depth);

		cout << "Translation plane " << q << "#" << no << " has "
				<<  nb_subplanes << " partial Fano subplanes "
						"(up to isomorphism) at depth "
				<< target_depth << endl;
		}
	else if (f_arcs) {
		char prefix[1000];
		int nb;

		int target_depth;

		if (f_depth) {
			target_depth = depth;
			}
		else {
			target_depth = order_of_plane + 2;
				// we are looking for hyperovals
			}


		sprintf(prefix, "Arcs_TP_%d_", no);

		Andre->classify_arcs(prefix, target_depth, verbose_level);


		nb = Andre->arcs->nb_orbits_at_level(target_depth);

		cout << "Translation plane " << q << "#" << no << " has "
				<<  nb << " Arcs of size " << target_depth
				<< " (up to isomorphism)" << endl;
		}

	FREE_OBJECT(Andre);
	FREE_OBJECT(gens);
	FREE_OBJECT(An);
	FREE_OBJECT(F);
}

