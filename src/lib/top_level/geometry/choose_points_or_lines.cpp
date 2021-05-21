// choose_points_or_lines.cpp
//
// Anton Betten
//
// started: Nov 21, 2010
// moved to TOP_LEVEL: Nov 29, 2010

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


choose_points_or_lines::choose_points_or_lines()
{
	null();
}

choose_points_or_lines::~choose_points_or_lines()
{
	freeself();
}

void choose_points_or_lines::null()
{
	//label;
	data = NULL;
	//Arc = NULL;
	//input_set = NULL;
	//f_is_in_input_set = NULL;
	transporter = NULL;
	transporter_inv = NULL;
	gen = NULL;
	Control = NULL;
	Poset = NULL;
	f_has_favorite = FALSE;
	f_iso_test_only = FALSE;
	f_has_orbit_select = FALSE;
	print_generators_verbose_level = 0;
	null_representative();
	
}

void choose_points_or_lines::freeself()
{
#if 0
	if (input_set) {
		FREE_int(input_set);
	}
	if (f_is_in_input_set) {
		FREE_int(f_is_in_input_set);
	}
#endif
	if (transporter) {
		FREE_int(transporter);
	}
	if (transporter_inv) {
		FREE_int(transporter_inv);
	}
	if (gen) {
		FREE_OBJECT(gen);
	}
	if (Control) {
		FREE_OBJECT(Control);
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	free_representative();
	null();
}

void choose_points_or_lines::null_representative()
{
	representative = NULL;
	stab_order = NULL;
	Stab_Strong_gens = NULL;
	//stab_gens = NULL;
	//stab_tl = NULL;
	stab = NULL;
}

void choose_points_or_lines::free_representative()
{
	int f_v = FALSE;

	if (f_v) {
		cout << "choose_points_or_lines::free_representative freeing representative" << endl;
	}
	if (representative) {
		FREE_lint(representative);
	}
	if (stab_order) {
		FREE_OBJECT(stab_order);
	}
	if (Stab_Strong_gens) {
		FREE_OBJECT(Stab_Strong_gens);
	}
	if (f_v) {
		cout << "choose_points_or_lines::free_representative freeing stab" << endl;
	}
	if (stab) {
		FREE_OBJECT(stab);
	}
	null_representative();
}

#if 0
void poset::add_testing_without_group(
		void (*func)(int *S, int len,
				int *candidates, int nb_candidates,
				int *good_candidates, int &nb_good_candidates,
				void *data, int verbose_level),
		void *data,
		int verbose_level)
#endif

void choose_points_or_lines::init(const char *label, void *data, 
	action *A, action *A_lines, 
	int f_choose_lines, 
	int nb_points_or_lines, 
	int (*check_function)(int len, long int *S, void *data, int verbose_level),
	int t0, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "choose_points_or_lines::init " << label << endl;
		}
	
	choose_points_or_lines::label.assign(label);
	//choose_points_or_lines::Arc = Arc;
	choose_points_or_lines::data = data;
	choose_points_or_lines::A = A;
	choose_points_or_lines::A_lines = A_lines;
	choose_points_or_lines::f_choose_lines = f_choose_lines;
	choose_points_or_lines::nb_points_or_lines = nb_points_or_lines;
	choose_points_or_lines::check_function = check_function;
	choose_points_or_lines::t0 = t0;
	if (f_choose_lines) {
		A2 = A_lines;
	}
	else {
		A2 = A;
	}
	transporter = NEW_int(A->elt_size_in_int);
	transporter_inv = NEW_int(A->elt_size_in_int);
}

void choose_points_or_lines::compute_orbits_from_sims(sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//vector_ge *gens;
	//int *tl;
	strong_generators *Strong_gens;
	
	if (f_v) {
		cout << "choose_points_or_lines::compute_orbits_from_sims " << label << endl;
	}
	Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_from_sims(G, verbose_level - 2);
	compute_orbits(Strong_gens, verbose_level);
	FREE_OBJECT(Strong_gens);
}

void choose_points_or_lines::compute_orbits(strong_generators *Strong_gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "choose_points_or_lines::compute_orbits " << label << endl;
	}
	
	if (gen) {
		FREE_OBJECT(gen);
	}
	if (Poset) {
		FREE_OBJECT(Poset);
	}
	
	gen = NEW_OBJECT(poset_classification);

	//sprintf(gen->fname_base, "%s", label);
	
	
	//gen->depth = nb_points_or_lines;
	
	if (f_vv) {
		cout << "choose_points_or_lines::compute_orbits "
				<< label << " calling gen->init" << endl;
	}

	Control = NEW_OBJECT(poset_classification_control);
	Control->f_depth = TRUE;
	Control->depth = nb_points_or_lines;

	Poset = NEW_OBJECT(poset_with_group_action);
	Poset->init_subset_lattice(A, A2, Strong_gens, verbose_level);

	gen->initialize_and_allocate_root_node(Control, Poset,
			nb_points_or_lines /* sz */, verbose_level - 2);
	if (f_vv) {
		cout << "choose_points_or_lines::compute_orbits "
				<< label << " calling gen->init_check_func" << endl;
	}


#if 0
	gen->f_print_function = TRUE;
	gen->print_function = print_set;
	gen->print_function_data = this;
#endif	

	
	int f_use_invariant_subset_if_available = TRUE;
	//int f_implicit_fusion = FALSE;
	int f_debug = FALSE;
	
	if (f_vv) {
		cout << "choose_points_or_lines::compute_orbits "
				<< label << " calling generator_main" << endl;
	}
	gen->main(t0, 
		Control->depth,
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		f_debug, 
		verbose_level - 2);
	
	
	if (f_vv) {
		cout << "choose_points_or_lines::compute_orbits "
				<< label << " done with generator_main" << endl;
	}
	nb_orbits = gen->nb_orbits_at_level(nb_points_or_lines);

	if (f_v) {
		cout << "choose_points_or_lines::compute_orbits "
				<< label << " we found " << nb_orbits
				<< " orbits on " << nb_points_or_lines;
		if (f_choose_lines) {
			cout << "-sets of lines" << endl;
		}
		else {
			cout << "-sets of points" << endl;
		}
	}
}

void choose_points_or_lines::choose_orbit(int orbit_no,
		int &f_hit_favorite, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	//int f_v4 = (verbose_level >= 4);
	int f, nd, i;
	int f_changed;
	long int *the_favorite_representative;
	group_container *G;
	poset_orbit_node *O;
	
	f_hit_favorite = FALSE;
	if (f_v) {
		cout << "choose_points_or_lines::choose_orbit " << label
				<< " orbit_no " << orbit_no << " / " << nb_orbits << endl;
	}
	
	
	free_representative();


	f = gen->first_node_at_level(nb_points_or_lines);
	nd = f + orbit_no;
	
	longinteger_object go;
	
	G = NEW_OBJECT(group_container);
	representative = NEW_lint(nb_points_or_lines);
	the_favorite_representative = NEW_lint(nb_points_or_lines);
	
	O = gen->get_node(nd);
	O->store_set_to(gen, nb_points_or_lines - 1, representative);
	
	if (f_vv) {
		cout << "##############################################################################" << endl;
		cout << "choose_points_or_lines::choose_orbit " << label
				<< " choosing orbit " << orbit_no << " / " << nb_orbits << endl;
	}
	if (f_vvv) {
		cout << "choose_points_or_lines::choose_orbit " << label << " the ";
		if (f_choose_lines) {
			cout << "lines";
		}
		else {
			cout << "points";
		}
		cout << " representing orbit " << orbit_no << " / "
				<< nb_orbits << " are ";
		Orbiter->Lint_vec.set_print(representative, nb_points_or_lines);
		cout << endl;
	}


	f_changed = FALSE;
	if (f_has_favorite) {
		f_hit_favorite = favorite_orbit_representative(
				transporter, transporter_inv,
				the_favorite_representative, verbose_level - 3);
		if (f_hit_favorite) {
			if (f_iso_test_only) {
				f_changed = FALSE;
				if (f_v) {
					cout << "choose_points_or_lines::choose_orbit "
							<< label << " isomorphism test only" << endl;
					cout << "element mapping the favorite set to "
							"the canonical set:" << endl;
					A->print_for_make_element(cout, transporter_inv);
					cout << endl;					
					A->print_quick(cout, transporter_inv);
				}
			}
			else {
				f_changed = TRUE;
			}
		}
	}
	if (f_changed) {
		for (i = 0; i < nb_points_or_lines; i++) {
			representative[i] = the_favorite_representative[i];
		}
		if (f_vvv) {
			cout << "choose_points_or_lines::choose_orbit " << label
				<< " / " << nb_orbits << " after changing, the "
				"representative set for orbit " << orbit_no << " are ";
			Orbiter->Lint_vec.set_print(representative, nb_points_or_lines);
			cout << endl;
		}
	}
	

	O->get_stabilizer(
			gen,
			*G, go,
			verbose_level);
#if 0
	G->init(A, verbose_level - 2);
	G->init_strong_generators_by_hdl(O->nb_strong_generators,
			O->hdl_strong_generators, O->tl, FALSE);
	G->schreier_sims(0);
	G->group_order(go);
#endif

	if (f_vv) {
		cout << "stabilizer of the chosen set has order " << go << endl;
	}
	
	stab_order = NEW_OBJECT(longinteger_object);
	Stab_Strong_gens = NEW_OBJECT(strong_generators);

	if (f_changed) {
		
		longinteger_object go1;
		sims *NewStab;
		
		if (f_vvv) {
			cout << "computing NewStab (because we changed)" << endl;
		}
		NewStab = NEW_OBJECT(sims);
		NewStab->init(A, verbose_level - 2);
		NewStab->init_trivial_group(0/*verbose_level - 1*/);
		//NewStab->group_order(stab_order);

		NewStab->conjugate(A, G->S, transporter_inv,
				FALSE, 0 /*verbose_level - 1*/);
		NewStab->group_order(go1);
		if (f_vv) {
			cout << "The conjugated stabilizer has order " << go1 << endl;
		}
		Stab_Strong_gens->init_from_sims(NewStab, 0);
		NewStab->group_order(*stab_order);

		FREE_OBJECT(NewStab);
	}
	else {
		Stab_Strong_gens->init_from_sims(G->S, 0);
		G->S->group_order(*stab_order);
	}

	if (f_v) {
		cout << "choose_points_or_lines::choose_orbit " << label
			<< " orbit_no " << orbit_no << " / " << nb_orbits
			<< " done, chosen the set ";
		Orbiter->Lint_vec.set_print(representative, nb_points_or_lines);
		cout << "_" << *stab_order;
		cout << endl;
	}
#if 0
	if (f_vv) {
		cout << "tl: ";
		int_vec_print(cout, stab_tl, A->base_len);
		cout << endl;
	}
#endif
	if (f_vv) {
		cout << "choose_points_or_lines::choose_orbit " << label
			<< " we have " << Stab_Strong_gens->gens->len
			<< " strong generators" << endl;
	}
	if (f_vv || print_generators_verbose_level) {
		for (i = 0; i < Stab_Strong_gens->gens->len; i++) {
			cout << i << " : " << endl;
			
			A->element_print_quick(Stab_Strong_gens->gens->ith(i), cout);
			cout << endl;

			if (print_generators_verbose_level > 1) {
				cout << "in action on points:" << endl;
				A->element_print_as_permutation(
						Stab_Strong_gens->gens->ith(i), cout);
				cout << endl;
				cout << "in action on lines:" << endl;
				A_lines->element_print_as_permutation(
						Stab_Strong_gens->gens->ith(i), cout);
				cout << endl;
			}
		}
	}
	


	FREE_OBJECT(G);
	FREE_lint(the_favorite_representative);
	
}

int choose_points_or_lines::favorite_orbit_representative(
	int *transporter, int *transporter_inv,
	long int *the_favorite_representative,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_implicit_fusion = FALSE;
	long int *canonical_set1;
	long int *canonical_set2;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int f_OK = TRUE;
	int i;
	
	if (f_v) {
		cout << "choose_points_or_lines::favorite_orbit_representative "
				<< label << endl;
	}
	if (nb_points_or_lines != favorite_size) {
		cout << "choose_points_or_lines::favorite_orbit_representative "
				<< label << " not the right size" << endl;
		cout << "nb_points_or_lines=" << nb_points_or_lines << endl;		
		cout << "favorite_size=" << favorite_size << endl;		
		return FALSE;
	}
	
	canonical_set1 = NEW_lint(nb_points_or_lines);
	canonical_set2 = NEW_lint(nb_points_or_lines);
	Elt1 = NEW_int(A2->elt_size_in_int);
	Elt2 = NEW_int(A2->elt_size_in_int);
	Elt3 = NEW_int(A2->elt_size_in_int);
	gen->trace_set(
		favorite, nb_points_or_lines,
		nb_points_or_lines /* level */,
		canonical_set1, 
		transporter_inv, 
		//f_implicit_fusion, 
		verbose_level - 2);
	
	
	for (i = 0; i < nb_points_or_lines; i++) {
		if (canonical_set1[i] != representative[i]) {
			f_OK = FALSE;
			break;
		}
	}
	A2->element_invert(transporter_inv, transporter, FALSE);
	if (f_OK) {
		if (f_v) {
			cout << "arc::favorite_zero_lines: transporter "
					"to favorite set:" << endl;
			A2->element_print(transporter, cout);
			A2->element_print_as_permutation(transporter, cout);
		}
		for (i = 0; i < nb_points_or_lines; i++) {
			the_favorite_representative[i] = favorite[i];
		}
	}
	FREE_lint(canonical_set1);
	FREE_lint(canonical_set2);
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_OK) {
		return TRUE;
	}
	else {
		return FALSE;
	}
	
}

void choose_points_or_lines::print_rep()
{
	Orbiter->Lint_vec.set_print(representative, nb_points_or_lines);
	cout << "_" << *stab_order;
}

void choose_points_or_lines::print_stab()
{
	int i;

	if (Stab_Strong_gens->gens->len == 0)
		return;
	cout << "generators:" << endl;
	for (i = 0; i < Stab_Strong_gens->gens->len; i++) {
		A->element_print_quick(Stab_Strong_gens->gens->ith(i), cout);

#if 0
		cout << "as permutation of points:" << endl;
		A->element_print_as_permutation(Stab_Strong_gens->gens->ith(i), cout);
		cout << endl;
		cout << "as permutation of lines:" << endl;
		A_lines->element_print_as_permutation(Stab_Strong_gens->gens->ith(i), cout);
		cout << endl;
#endif

	}
}

int choose_points_or_lines::is_in_rep(int a)
{
	int i;

	for (i = 0; i < nb_points_or_lines; i++) {
		if (a == representative[i]) {
			return TRUE;
		}
	}
	return FALSE;
}

}}

