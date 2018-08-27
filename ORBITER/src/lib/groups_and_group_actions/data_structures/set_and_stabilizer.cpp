// set_and_stabilizer.C
//
// Anton Betten
// September 18, 2016

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

set_and_stabilizer::set_and_stabilizer()
{
	null();
}

set_and_stabilizer::~set_and_stabilizer()
{
	freeself();
}

void set_and_stabilizer::null()
{
	A = NULL;
	A2 = NULL;
	data = NULL;
	Strong_gens = NULL;
	Stab = NULL;
}

void set_and_stabilizer::freeself()
{
	if (data) {
		FREE_INT(data);
		}
	if (Strong_gens) {
		FREE_OBJECT(Strong_gens);
		}
	if (Stab) {
		FREE_OBJECT(Stab);
		}
	null();
};

void set_and_stabilizer::init(action *A, action *A2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::init" << endl;
		}
	set_and_stabilizer::A = A;
	set_and_stabilizer::A2 = A2;
	if (f_v) {
		cout << "set_and_stabilizer::init done" << endl;
		}
}

void set_and_stabilizer::init_everything(
	action *A, action *A2, INT *Set, INT set_sz,
	strong_generators *gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

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

set_and_stabilizer *set_and_stabilizer::create_copy(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	set_and_stabilizer *SaS;

	if (f_v) {
		cout << "set_and_stabilizer::create_copy" << endl;
		}

	SaS = NEW_OBJECT(set_and_stabilizer);
	SaS->A = A;
	SaS->A2 = A2;
	SaS->data = NEW_INT(sz);
	INT_vec_copy(data, SaS->data, sz);
	SaS->sz = sz;
	target_go.assign_to(SaS->target_go);

	SaS->Strong_gens = NEW_OBJECT(strong_generators);
	Strong_gens->init_copy(SaS->Strong_gens, 0 /* verbose_level*/);
	SaS->Stab = SaS->Strong_gens->create_sims(verbose_level);
	
	if (f_v) {
		cout << "set_and_stabilizer::create_copy done" << endl;
		}
	return SaS;
}

void set_and_stabilizer::allocate_data(INT sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::allocate_data" << endl;
		}
	set_and_stabilizer::sz = sz;
	set_and_stabilizer::data = NEW_INT(sz);
	if (f_v) {
		cout << "set_and_stabilizer::allocate_data done" << endl;
		}
}

void set_and_stabilizer::init_data(INT *data, INT sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::init_data" << endl;
		}
	set_and_stabilizer::sz = sz;
	set_and_stabilizer::data = NEW_INT(sz);
	INT_vec_copy(data, set_and_stabilizer::data, sz);
	if (f_v) {
		cout << "set_and_stabilizer::init_data done" << endl;
		}
}

void set_and_stabilizer::init_stab_from_data(INT *data_gens, 
	INT data_gens_size, INT nb_gens, const BYTE *ascii_target_go, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_data" << endl;
		}
	vector_ge *gens;

	gens = NEW_OBJECT(vector_ge);
	gens->init(A);
	target_go.create_from_base_10_string(ascii_target_go);


	gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A->make_element(gens->ith(i), data_gens + i * data_gens_size, 0);
		}

	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong generators are:" << endl;
		Strong_gens->print_generators();
		}
	
	Stab = Strong_gens->create_sims(verbose_level);

	FREE_OBJECT(gens);
	
	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_data done" << endl;
		}
}

void set_and_stabilizer::init_stab_from_file(
	const BYTE *fname_gens,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;
	vector_ge *gens;

	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_file" << endl;
		}

	if (file_size(fname_gens) <= 0) {
		cout << "set_and_stabilizer::init_stab_from_file "
				"the file " << fname_gens
				<< " does not exist or is empty" << endl;
		exit(1);
		}

	{
	ifstream f(fname_gens);
	INT nb_gens;
	INT *data;
	BYTE target_go_ascii[1000];

	f >> nb_gens;
	f >> target_go_ascii;


	target_go.create_from_base_10_string(target_go_ascii);


	data = NEW_INT(A->make_element_size);


	gens = NEW_OBJECT(vector_ge);
	gens->init(A);


	gens->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < A->make_element_size; j++) {
			f >> data[j];
			}
		A->make_element(gens->ith(i), data, 0);
		}

	FREE_INT(data);
	}

	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		0 /*verbose_level*/);

	if (FALSE) {
		cout << "strong generators are:" << endl;
		Strong_gens->print_generators();
		}
	
	Stab = Strong_gens->create_sims(verbose_level);

	FREE_OBJECT(gens);
	
	if (f_v) {
		cout << "set_and_stabilizer::init_stab_from_file done" << endl;
		}
}

void set_and_stabilizer::print_set_tex(ostream &ost)
{
	INT_set_print_tex(ost, data, sz);
	ost << "_{";
	target_go.print_not_scientific(ost);
	ost << "}";
}

void set_and_stabilizer::print_set_tex_for_inline_text(ostream &ost)
{
	INT_set_print_tex_for_inline_text(ost, data, sz);
	ost << "_{";
	target_go.print_not_scientific(ost);
	ost << "}";
}

void set_and_stabilizer::print_generators_tex(ostream &ost)
{
	Strong_gens->print_generators_tex();
}

void set_and_stabilizer::apply_to_self(INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *data2;
	INT i;
	vector_ge *gens;
	strong_generators *sg;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self" << endl;
		}
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self Elt=" << endl;
		A->element_print_quick(Elt, cout);
		}

	data2 = NEW_INT(sz);
	A2->map_a_set(data, data2, sz, Elt, 0 /* verbose_level */);
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
		if (!A2->check_if_in_set_stabilizer(
				gens->ith(i), sz, data2, 0 /*verbose_level*/)) {
			cout << "set_and_stabilizer::apply_to_self "
					"conjugate element does not stabilize the set" << endl;
			}
		}
	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		gens, sg, 
		0 /*verbose_level*/);
	INT_vec_copy(data2, data, sz);
	FREE_OBJECT(gens);
	FREE_OBJECT(Strong_gens);
	Strong_gens = sg;
	if (Stab) {
		FREE_OBJECT(Stab);
		Stab = Strong_gens->create_sims(verbose_level);
		}
	FREE_INT(data2);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self done" << endl;
		}
}

void set_and_stabilizer::apply_to_self_inverse(INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_inverse" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);

	A->element_invert(Elt, Elt1, 0);
	apply_to_self(Elt1, verbose_level);

	FREE_INT(Elt1);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_inverse done" << endl;
		}
}

void set_and_stabilizer::apply_to_self_element_raw(INT *Elt_data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_element_raw" << endl;
		}

	Elt = NEW_INT(A->elt_size_in_INT);
	A->make_element(Elt, Elt_data, 0);
	apply_to_self(Elt, verbose_level);
	FREE_INT(Elt);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_element_raw done" << endl;
		}
}

void set_and_stabilizer::apply_to_self_inverse_element_raw(INT *Elt_data, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt;

	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_inverse_element_raw" << endl;
		}

	Elt = NEW_INT(A->elt_size_in_INT);
	A->make_element(Elt, Elt_data, 0);
	apply_to_self_inverse(Elt, verbose_level);
	FREE_INT(Elt);
	if (f_v) {
		cout << "set_and_stabilizer::apply_to_self_inverse_element_raw done" << endl;
		}
}


void set_and_stabilizer::rearrange_by_orbits(INT *&orbit_first, INT *&orbit_length, 
	INT *&orbit, INT &nb_orbits, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits" << endl;
		}
	
	action *A_on_set;

			
	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits creating restricted action on the set of lines" << endl;
		}
	A_on_set = A2->restricted_action(data, sz, verbose_level);
	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits creating restricted action on the set of lines done" << endl;
		}

	schreier *Orb;
	INT *data2;
	INT f, l, h, cur, j, a, b;
			
	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits computing orbits on set:" << endl;
		}
	Orb = Strong_gens->orbits_on_points_schreier(A_on_set, verbose_level);

	data2 = NEW_INT(sz);

	nb_orbits = Orb->nb_orbits;
	orbit_first = NEW_INT(nb_orbits);
	orbit_length = NEW_INT(nb_orbits);
	orbit = NEW_INT(sz);
	

	cur = 0;
	orbit_first[0] = 0;


	classify C;
	INT t, ff, c, d;
	//INT d;

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
	INT_vec_copy(data2, data, sz);

	FREE_OBJECT(Orb);
	FREE_OBJECT(A_on_set);

	if (f_v) {
		cout << "set_and_stabilizer::rearrange_by_orbits done" << endl;
		}
}

action *set_and_stabilizer::create_restricted_action_on_the_set(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::create_restricted_"
				"action_on_the_set" << endl;
		}
	
	action *A_on_set;

			
	if (f_v) {
		cout << "set_and_stabilizer::create_restricted_"
				"action_on_the_set creating restricted "
				"action on the set" << endl;
		}
	A_on_set = A2->restricted_action(data, sz, verbose_level);
	
	Strong_gens->print_with_given_action(cout, A_on_set);
	
	if (f_v) {
		cout << "set_and_stabilizer::create_restricted_"
				"action_on_the_set creating restricted "
				"action on the set done" << endl;
		}

	return A_on_set;
}

void set_and_stabilizer::print_restricted_action_on_the_set(
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::print_restricted_"
				"action_on_the_set" << endl;
		}
	
	action *A_on_set;

			
	if (f_v) {
		cout << "set_and_stabilizer::print_restricted_action_"
				"on_the_set creating restricted action on the set" << endl;
		}
	A_on_set = A2->restricted_action(data, sz, verbose_level);
	
	Strong_gens->print_with_given_action(cout, A_on_set);
	
	if (f_v) {
		cout << "set_and_stabilizer::print_restricted_action_"
				"on_the_set creating restricted action "
				"on the set done" << endl;
		}

	FREE_OBJECT(A_on_set);
}

void set_and_stabilizer::test_if_group_acts(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::test_if_group_acts" << endl;
		}
	
	if (f_v) {
		cout << "set_and_stabilizer::test_if_group_acts done" << endl;
		}
}

void set_and_stabilizer::init_surface(surface *Surf,
		action *A, action *A2, INT q, INT no,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "set_and_stabilizer::init_surface" << endl;
		}

	INT *data;
	INT nb_gens;
	INT data_size;
	const BYTE *stab_order;
	//INT nb_E;
	INT *starter_configuration;
	INT nb_lines;
	INT *Lines;
	INT *Lines_wedge;
	INT double_six[12];
	
	starter_configuration = cubic_surface_single_six(q, no);
	if (f_v) {
		cout << "set_and_stabilizer::init_surface "
				"before create_double_six_from_five_lines_"
				"with_a_common_transversal ";
		}
	if (Surf->create_double_six_from_five_lines_with_a_common_transversal(
			starter_configuration + 1,
			double_six, verbose_level)) {
		cout << "set_and_stabilizer::init_surface "
				"The starter configuration is good, a double "
				"six has been computed:" << endl;
		INT_matrix_print(double_six, 2, 6);
		}
	else {
		cout << "set_and_stabilizer::init_surface "
				"The starter configuration is bad, there is "
				"no double six" << endl;
		exit(1);
		}
	nb_lines = 27;
	Lines = NEW_INT(nb_lines);
	Lines_wedge = NEW_INT(nb_lines);
	INT_vec_copy(double_six, Lines, 12);
	Surf->create_remaining_fifteen_lines(
			double_six, Lines + 12, 0 /* verbose_level */);

	//Surf->line_to_wedge_vec(Lines, Lines_wedge, nb_lines);

#if 0
	Lines = cubic_surface_Lines(q, no);

	if (FALSE) {
		cout << "The lines are: ";
		INT_vec_print(cout, Lines, 27);
		cout << endl;
		}
#endif


	cubic_surface_stab_gens(q, no,
			data, nb_gens, data_size, stab_order);


	init(A, A2, verbose_level);

	init_data(Lines, nb_lines, verbose_level);

	init_stab_from_data(data, 
		data_size, nb_gens, 
		stab_order, 
		verbose_level);

	FREE_INT(Lines);
	FREE_INT(Lines_wedge);
	
	if (f_v) {
		cout << "set_and_stabilizer::init_surface done" << endl;
		}
}


