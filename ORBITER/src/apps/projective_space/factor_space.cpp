// factor_space.C
// 
// Anton Betten
// 1/20/2010
//
//
// 
//
//

#include "orbiter.h"
#include "discreta.h"


// global data:

INT t0; // the system time when the program started

void test1(INT n, INT q, INT verbose_level);
void test2(INT n, INT q, INT verbose_level);
INT rank_point_func(INT *v, void *data);
void unrank_point_func(INT *v, INT rk, void *data);
INT test_func(INT len, INT *S, void *data, INT verbose_level);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_n = FALSE;
	INT n = 0;
	INT f_q = FALSE;
	INT q;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	if (!f_n) {
		cout << "please use -n option" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use -q option" << endl;
		exit(1);
		}
	//test1(n, q, verbose_level);
	test2(n, q, verbose_level);
	
	the_end(t0);
}


void test1(INT n, INT q, INT verbose_level)
{
	finite_field *F;
	projective_space *P;
	INT f_semilinear = TRUE;
	const char *override_poly = NULL;
	INT f_basis = TRUE;
	INT f_with_group = TRUE;

	F = new finite_field;
	P = new projective_space;

	F->init_override_polynomial(q, override_poly, 0);
	P->init(n, F, f_with_group, 
		FALSE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		f_semilinear, f_basis, verbose_level);

	action_on_factor_space *AF;
	INT basis_elts_ranks[] = {3};
	INT nb_basis_elts = 1;

	AF = new action_on_factor_space;
	AF->init_by_rank(*P->A, *P->A, n + 1, P->F, basis_elts_ranks, nb_basis_elts, 
		TRUE /*f_compute_tables*/, verbose_level);
	AF->list_all_elements();

	action *A;
	INT f_induce_action = TRUE;

	A = new action;

	if (!P->A->f_has_sims) {
		cout << "P->A does not have sims, so we create it" << endl;
		P->A->create_sims(verbose_level);
		}

	vector_ge SG1;
	INT *tl;

	tl = new INT[P->A->base_len];
	cout << "computing point stabilizer of " << basis_elts_ranks[0] << endl;
	P->A->Sims->point_stabilizer(SG1, tl, basis_elts_ranks[0], verbose_level);
	sims *S1;

	S1 = new sims;
	S1->init(P->A);
	S1->init_trivial_group(verbose_level - 1);
	S1->init_generators(SG1, FALSE);
	S1->compute_base_orbits_known_length(tl, verbose_level - 1);
	
	cout << "before A->induced_action_on_factor_space" << endl;
	A->induced_action_on_factor_space(P->A, AF, f_induce_action, S1, verbose_level);
	cout << "after A->induced_action_on_factor_space" << endl;

	longinteger_object go;

	A->group_order(go);
	cout << "group order = " << go << endl;


	schreier Sch;

	Sch.init(A);
	Sch.init_generators(SG1);
	Sch.compute_all_point_orbits(verbose_level);
	Sch.print_and_list_orbits(cout);
	Sch.print_tables(cout, TRUE /* f_with_cosetrep */);
	
#if 0
	INT *Elt;
	INT h, i, j;
	
	for (h = 0; h < SG1.len; h++) {
		Elt = SG1.ith(h);
		cout << "generator " << h << ":" << endl;
		A->element_print_quick(Elt, cout);
		for (i = 0; i < AF->degree; i++) {
			j = A->image_of(Elt, i);
			cout << setw(5) << i << " : " << setw(5) << j << endl;
			}
		}
#endif
}

void test2(INT n, INT q, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	finite_field *F;
	projective_space *P;
	INT f_semilinear = TRUE;
	INT f_basis = TRUE;
	INT depth = 2;
	INT f_with_group = TRUE;
	
	F = new finite_field;
	P = new projective_space;

	F->init(q, 0);
	P->init(n, F, f_with_group, 
		FALSE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		f_semilinear, f_basis, 0/*verbose_level*/);

	if (!P->A->f_has_sims) {
		cout << "P->A does not have sims, so we create it" << endl;
		P->A->create_sims(verbose_level);
		}
	if (!P->A->f_has_strong_generators) {
		cout << "P->A does not have strong generators" << endl;
		//P->A->create_sims(verbose_level);
		exit(1);
		}
	
	generator *Gen;

	Gen = new generator;


	sprintf(Gen->fname_base, "%s", Gen->prefix);
	
	
	Gen->depth = depth;
	
	if (f_v) {
		cout << "Gen->init" << endl;
		}
	Gen->init(P->A, P->A, 
		P->A->Strong_gens, 
		// P->A->strong_generators, P->A->tl, 
		Gen->depth /* sz */, verbose_level);
	Gen->init_check_func(
		test_func, 
		NULL /* candidate_check_data */);

	//Gen->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);

	Gen->init_vector_space_action(n + 1 /*vector_space_dimension*/, 
		P->F, 
		rank_point_func, 
		unrank_point_func, 
		Gen, 
		verbose_level);
#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif	

	INT nb_oracle_nodes = 1000;
	
	if (f_v) {
		cout << "Gen->init_oracle" << endl;
		}
	Gen->init_oracle(nb_oracle_nodes, verbose_level - 1);
	if (f_v) {
		cout << "calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, Gen->verbose_level);
	
	INT schreier_depth = Gen->depth;
	INT f_use_invariant_subset_if_available = FALSE;
	INT f_implicit_fusion = FALSE;
	INT f_debug = FALSE;
	
	if (f_v) {
		cout << "calling generator_main" << endl;
		}
	Gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		f_implicit_fusion, 
		f_debug, 
		verbose_level - 1);
	
	INT f, nb_orbits;
	
	if (f_v) {
		cout << "done with generator_main" << endl;
		}
	f = Gen->first_oracle_node_at_level[depth];
	nb_orbits = Gen->first_oracle_node_at_level[depth + 1] - f;
	if (f_v) {
		cout << "we found " << nb_orbits << " orbits at depth " << depth<< endl;
		}
	
}

INT rank_point_func(INT *v, void *data)
{
	generator *gen = (generator *) data;
	INT rk;
	
	PG_element_rank_modified(*gen->F, v, 1, gen->vector_space_dimension, rk);
	return rk;
}

void unrank_point_func(INT *v, INT rk, void *data)
{
	generator *gen = (generator *) data;

	PG_element_unrank_modified(*gen->F, v, 1, gen->vector_space_dimension, rk);
}


INT test_func(INT len, INT *S, void *data, INT verbose_level)
{
	//arc *Arc = (arc *) data;
	//INT i, p, j;
	INT f_OK = TRUE;
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		}
	
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		return FALSE;
		}
}


