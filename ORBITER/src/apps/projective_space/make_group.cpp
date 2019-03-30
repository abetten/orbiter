// make_group.C
// 
// Anton Betten
// Apr 14, 2011
//
// 
//
//

#include "orbiter.h"

using namespace std;


using namespace orbiter;


// global data:

int t0; // the system time when the program started

void create_group(int verbose_level);
void projective_space_init_line_action(projective_space *P,
		action *A_points, action *&A_on_lines, int verbose_level);


int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	
	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		}


	
	create_group(verbose_level);

	the_end(t0);
}

void create_group(int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int n = 1;
	int q = 8;

	action *A;
	action *A2;
	finite_field *F;
	projective_space *P;
	int i;
	vector_ge *nice_gens;
	
	F = NEW_OBJECT(finite_field);
	P = NEW_OBJECT(projective_space);

	F->init(q, 0);
	P->init(n, F, 
		TRUE /* f_init_incidence_structure */, 
		verbose_level - 3);

	cout << "Creating linear group" << endl;
	A = NEW_OBJECT(action);
	A->init_general_linear_group(n + 1, F,
			TRUE /* f_semilinear */,
			TRUE /* f_basis */,
			nice_gens,
			verbose_level - 2);
	FREE_OBJECT(nice_gens);
	
	cout << "Creating action on lines" << endl;
	projective_space_init_line_action(P, A, A2, verbose_level);



	int generators[] = {
		0,1,1,1,2,
		1,3,2,0,2
		};
	vector_ge gens;
	int *Elt1;
	
	Elt1 = NEW_int(A->elt_size_in_int);

	//set1 = NEW_int(P->N_points);
	//set2 = NEW_int(P->N_points);

	gens.init(A);
	gens.allocate(2);
	for (i = 0; i < 2; i++) {
		A->make_element(Elt1, generators + i * 5, verbose_level);
		A->element_move(Elt1, gens.ith(i), 0);
		cout << "generator " << i << ":" << endl;
		A->element_print(gens.ith(i), cout);
		A->element_print_as_permutation(gens.ith(i), cout);
		}

	sims *S;
	int nb_times = 100;

	S = NEW_OBJECT(sims);
	S->init(A);
	S->init_trivial_group(0);
	S->init_generators(gens, verbose_level);
	S->compute_base_orbits(verbose_level);
	S->closure_group(nb_times, verbose_level);
	S->print(verbose_level);
	cout << "generators:" << endl;
	S->gens.print(cout);

	cout << "list of group elements:" << endl;
	S->print_all_group_elements();
}


void projective_space_init_line_action(projective_space *P,
		action *A_points, action *&A_on_lines, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	action_on_grassmannian *AoL;

	if (f_v) {
		cout << "projective_space_init_line_action" << endl;
		}
	A_on_lines = NEW_OBJECT(action);

	AoL = NEW_OBJECT(action_on_grassmannian);

	AoL->init(*A_points, P->Grass_lines, verbose_level - 5);


	if (f_v) {
		cout << "projective_space_init_line_action "
				"action on grassmannian established" << endl;
		}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing A_on_lines" << endl;
		}
	int f_induce_action = TRUE;
	sims S;
	longinteger_object go1;

	S.init(A_points);
	S.init_generators(*A_points->Strong_gens->gens,
			0/*verbose_level*/);
	S.compute_base_orbits_known_length(A_points->transversal_length,
			0/*verbose_level - 1*/);
	S.group_order(go1);
	if (f_v) {
		cout << "projective_space_init_line_action "
				"group order " << go1 << endl;
		}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing action on grassmannian" << endl;
		}
	A_on_lines->induced_action_on_grassmannian(A_points, AoL,
		f_induce_action, &S, verbose_level);
	if (f_v) {
		cout << "projective_space_init_line_action "
				"initializing A_on_lines done" << endl;
		A_on_lines->print_info();
		}

	if (f_v) {
		cout << "projective_space_init_line_action "
				"computing strong generators" << endl;
		}
	if (!A_on_lines->f_has_strong_generators) {
		cout << "projective_space_init_line_action "
				"induced action does not have strong generators" << endl;
		}
	if (f_v) {
		cout << "projective_space_init_line_action done" << endl;
		}
}

