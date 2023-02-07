// set_and_stabilizer.cpp
//
// Anton Betten
// September 18, 2016

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


set_and_stabilizer::set_and_stabilizer()
{
	A = NULL;
	A2 = NULL;
	data = NULL;
	sz = 0;
	//ring_theory::longinteger_object target_go;
	Strong_gens = NULL;
	Stab = NULL;
}


set_and_stabilizer::~set_and_stabilizer()
{
	if (data) {
		FREE_lint(data);
		}
	if (Strong_gens) {
		FREE_OBJECT(Strong_gens);
		}
	if (Stab) {
		FREE_OBJECT(Stab);
		}
}

void set_and_stabilizer::init(actions::action *A,
		actions::action *A2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::init" << endl;
		}
	set_and_stabilizer::A = A;
	set_and_stabilizer::A2 = A2;
	if (f_v) {
		cout << "set_and_stabilizer::init done" << endl;
		}
}

void set_and_stabilizer::group_order(ring_theory::longinteger_object &go)
{
	if (Strong_gens == NULL) {
		cout << "set_and_stabilizer::group_order "
				"Strong_gens == NULL" << endl;
		exit(1);
	}
	Strong_gens->group_order(go);
}

long int set_and_stabilizer::group_order_as_lint()
{
	if (Strong_gens == NULL) {
		cout << "set_and_stabilizer::group_order_as_int "
				"Strong_gens == NULL" << endl;
		exit(1);
	}
	return Strong_gens->group_order_as_lint();
}

void set_and_stabilizer::init_everything(
		actions::action *A,
		actions::action *A2,
		long int *Set, int set_sz,
		groups::strong_generators *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::init_everything" << endl;
		}
	set_and_stabilizer::A = A;
	set_and_stabilizer::A2 = A2;
	set_and_stabilizer::data = Set;
	set_and_stabilizer::sz = set_sz;
	set_and_stabilizer::Strong_gens = gens;
	Strong_gens->group_order(target_go);
	Stab = Strong_gens->create_sims(verbose_level);
	if (f_v) {
		cout << "set_and_stabilizer::init_everything done" << endl;
		}
}

set_and_stabilizer *set_and_stabilizer::create_copy(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_and_stabilizer *SaS;

	if (f_v) {
		cout << "set_and_stabilizer::create_copy" << endl;
		}

	SaS = NEW_OBJECT(set_and_stabilizer);
	SaS->A = A;
	SaS->A2 = A2;
	SaS->data = NEW_lint(sz);
	Lint_vec_copy(data, SaS->data, sz);
	SaS->sz = sz;
	target_go.assign_to(SaS->target_go);

	SaS->Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_copy(SaS->Strong_gens, 0 /* verbose_level*/);
	SaS->Stab = SaS->Strong_gens->create_sims(verbose_level);
	
	if (f_v) {
		cout << "set_and_stabilizer::create_copy done" << endl;
		}
	return SaS;
}

void set_and_stabilizer::allocate_data(int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::allocate_data" << endl;
		}
	set_and_stabilizer::sz = sz;
	set_and_stabilizer::data = NEW_lint(sz);
	if (f_v) {
		cout << "set_and_stabilizer::allocate_data done" << endl;
		}
}

void set_and_stabilizer::init_data(long int *data, int sz, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::init_data" << endl;
		}
	set_and_stabilizer::sz = sz;
	set_and_stabilizer::data = NEW_lint(sz);
	Lint_vec_copy(data, set_and_stabilizer::data, sz);
	if (f_v) {
		cout << "set_and_stabilizer::init_data done" << endl;
		}
}

void set_and_stabilizer::init_stab_from_data(int *data_gens, 
	int data_gens_size, int nb_gens,
	std::string &ascii_target_go,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_data" << endl;
		}
	vector_ge *gens;

	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);
	target_go.create_from_base_10_string(ascii_target_go);


	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		A->Group_element->make_element(gens->ith(i), data_gens + i * data_gens_size, 0);
		}

	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong generators are:" << endl;
		Strong_gens->print_generators(cout);
		}
	
	Stab = Strong_gens->create_sims(verbose_level);

	FREE_OBJECT(gens);
	
	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_data done" << endl;
		}
}

void set_and_stabilizer::init_stab_from_file(
	const char *fname_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	vector_ge *gens;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_file" << endl;
		}

	if (Fio.file_size(fname_gens) <= 0) {
		cout << "set_and_stabilizer::init_stab_from_file "
				"the file " << fname_gens
				<< " does not exist or is empty" << endl;
		exit(1);
		}

	{
	ifstream f(fname_gens);
	int nb_gens;
	int *data;
	char target_go_ascii[1000];

	f >> nb_gens;
	f >> target_go_ascii;


	target_go.create_from_base_10_string(target_go_ascii);


	data = NEW_int(A->make_element_size);


	gens = NEW_OBJECT(vector_ge);
	gens->init(A, verbose_level - 2);


	gens->allocate(nb_gens, verbose_level - 2);
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < A->make_element_size; j++) {
			f >> data[j];
			}
		A->Group_element->make_element(gens->ith(i), data, 0);
		}

	FREE_int(data);
	}

	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong generators are:" << endl;
		Strong_gens->print_generators(cout);
		}
	
	Stab = Strong_gens->create_sims(verbose_level);

	FREE_OBJECT(gens);
	
	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_file done" << endl;
		}
}

void set_and_stabilizer::print_set_tex(std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;

	L.lint_set_print_tex(ost, data, sz);
	ost << "_{";
	target_go.print_not_scientific(ost);
	ost << "}";
}

void set_and_stabilizer::print_set_tex_for_inline_text(std::ostream &ost)
{
	orbiter_kernel_system::latex_interface L;

	L.lint_set_print_tex_for_inline_text(ost, data, sz);
	ost << "_{";
	target_go.print_not_scientific(ost);
	ost << "}";
}

void set_and_stabilizer::print_generators_tex(std::ostream &ost)
{
	Strong_gens->print_generators_tex();
}

void set_and_stabilizer::apply_to_self(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *data2;
	int i;
	vector_ge *gens;
	groups::strong_generators *sg;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self" << endl;
		}
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self Elt=" << endl;
		A->Group_element->element_print_quick(Elt, cout);
		}

	data2 = NEW_lint(sz);
	A2->Group_element->map_a_set(data, data2, sz, Elt, 0 /* verbose_level */);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self "
				"mapping the set under action " << A2->label << ":" << endl;
		for (i = 0; i < sz; i++) {
			cout << i << " : " << data[i] << " : " << data2[i] << endl;
			}
		}

	gens = NEW_OBJECT(vector_ge);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self "
				"before conjugating generators" << endl;
		}
	gens->init_conjugate_svas_of(Strong_gens->gens, Elt,
			0 /* verbose_level */);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self "
				"before testing the n e w generators" << endl;
		}
	for (i = 0; i < Strong_gens->gens->len; i++) {
		if (!A2->Group_element->check_if_in_set_stabilizer(
				gens->ith(i), sz, data2, 0 /*verbose_level*/)) {
			cout << "set_and_stabilizer::apply_to_self "
					"conjugate element does not stabilize the set" << endl;
			}
		}
	A->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		gens, sg, 
		0 /*verbose_level*/);
	Lint_vec_copy(data2, data, sz);
	FREE_OBJECT(gens);
	FREE_OBJECT(Strong_gens);
	Strong_gens = sg;
	if (Stab) {
		FREE_OBJECT(Stab);
		Stab = Strong_gens->create_sims(verbose_level);
		}
	FREE_lint(data2);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self done" << endl;
		}
}

void set_and_stabilizer::apply_to_self_inverse(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_inverse" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);

	A->Group_element->element_invert(Elt, Elt1, 0);
	apply_to_self(Elt1, verbose_level);

	FREE_int(Elt1);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_inverse done" << endl;
		}
}

void set_and_stabilizer::apply_to_self_element_raw(
		int *Elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_element_raw" << endl;
		}

	Elt = NEW_int(A->elt_size_in_int);
	A->Group_element->make_element(Elt, Elt_data, 0);
	apply_to_self(Elt, verbose_level);
	FREE_int(Elt);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_element_raw done" << endl;
		}
}

void set_and_stabilizer::apply_to_self_inverse_element_raw(
		int *Elt_data, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_"
				"inverse_element_raw" << endl;
		}

	Elt = NEW_int(A->elt_size_in_int);
	A->Group_element->make_element(Elt, Elt_data, 0);
	apply_to_self_inverse(Elt, verbose_level);
	FREE_int(Elt);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_"
				"inverse_element_raw done" << endl;
		}
}


void set_and_stabilizer::rearrange_by_orbits(
	int *&orbit_first, int *&orbit_length,
	int *&orbit, int &nb_orbits,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits" << endl;
		}
	
	actions::action *A_on_set;

			
	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits "
				"creating restricted action on the set "
				"of lines" << endl;
		}
	A_on_set = A2->Induced_action->restricted_action(data, sz, verbose_level);
	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits "
				"creating restricted action on the set of "
				"lines done" << endl;
		}

	groups::schreier *Orb;
	long int *data2;
	int f, l, h, cur, j, a, b;
			
	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits "
				"computing orbits on set:" << endl;
		}
	Orb = Strong_gens->orbits_on_points_schreier(
			A_on_set, verbose_level);

	data2 = NEW_lint(sz);

	nb_orbits = Orb->nb_orbits;
	orbit_first = NEW_int(nb_orbits);
	orbit_length = NEW_int(nb_orbits);
	orbit = NEW_int(sz);
	

	cur = 0;
	orbit_first[0] = 0;


	data_structures::tally C;
	int t, ff, c, d;
	//int d;

	d = 0;
	C.init(Orb->orbit_len, Orb->nb_orbits, FALSE, 0);
	for (t = 0; t < C.nb_types; t++) {
		ff = C.type_first[t];
		c = C.data_sorted[ff + 0];
		for (h = 0; h < Orb->nb_orbits; h++) {
			f = Orb->orbit_first[h];
			l = Orb->orbit_len[h];
#if 1
			if (l != c) {
				continue;
				}
#endif
			orbit_length[d] = l;
			for (j = 0; j < l; j++) {
				a = Orb->orbit[f + j];
				b = data[a];
				orbit[cur] = a;
				data2[cur++] = b;
				}
			if (d < Orb->nb_orbits - 1) {
				orbit_first[d + 1] = orbit_first[d] + l;
				}
			d++;
			}
		}
	Lint_vec_copy(data2, data, sz);

	FREE_OBJECT(Orb);
	FREE_OBJECT(A_on_set);

	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits done" << endl;
		}
}

actions::action *set_and_stabilizer::create_restricted_action_on_the_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::create_restricted_"
				"action_on_the_set" << endl;
		}
	
	actions::action *A_on_set;

			
	if (f_v) {
		cout << "set_and_stabilizer::create_restricted_"
				"action_on_the_set creating restricted "
				"action on the set" << endl;
		}
	A_on_set = A2->Induced_action->restricted_action(data, sz, verbose_level);
	
	Strong_gens->print_with_given_action(cout, A_on_set);
	
	if (f_v) {
		cout << "set_and_stabilizer::create_restricted_"
				"action_on_the_set creating restricted "
				"action on the set done" << endl;
		}

	return A_on_set;
}

void set_and_stabilizer::print_restricted_action_on_the_set(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::print_restricted_"
				"action_on_the_set" << endl;
		}
	
	actions::action *A_on_set;

			
	if (f_v) {
		cout << "set_and_stabilizer::print_restricted_action_"
				"on_the_set creating restricted action on the set" << endl;
		}
	A_on_set = A2->Induced_action->restricted_action(data, sz, verbose_level);
	
	Strong_gens->print_with_given_action(cout, A_on_set);
	
	if (f_v) {
		cout << "set_and_stabilizer::print_restricted_action_"
				"on_the_set creating restricted action "
				"on the set done" << endl;
		}

	FREE_OBJECT(A_on_set);
}

void set_and_stabilizer::test_if_group_acts(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::test_if_group_acts" << endl;
		}
	
	if (f_v) {
		cout << "set_and_stabilizer::test_if_group_acts done" << endl;
		}
}

int set_and_stabilizer::find(long int pt)
{
	data_structures::sorting Sorting;
	int idx;

	if (!Sorting.lint_vec_search(
			data,
			sz, pt, idx, 0)) {
		cout << "set_and_stabilizer::find" << endl;
		exit(1);
		}
	return idx;
}


}}}



