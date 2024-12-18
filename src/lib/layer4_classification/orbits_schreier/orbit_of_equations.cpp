// orbit_of_equations.cpp
// 
// Anton Betten
// May 29, 2018
//
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace orbits_schreier {

static int orbit_of_equations_compare_func(
		void *a, void *b, void *data);



orbit_of_equations::orbit_of_equations()
{
	Record_birth();
	A = NULL;
	AonHPD = NULL;
	F = NULL;
	SG = NULL;
	nb_monomials = 0;
	sz = 0;
	sz_for_compare = 0;
	data_tmp = NULL;

	position_of_original_object = 0;
	allocation_length = 0;
	used_length = 0;
	Equations = NULL;
	prev = NULL;
	label = NULL;

	f_has_print_function = false;
	print_function = NULL;
	print_function_data = NULL;

	f_has_reduction = false;
	reduction_function = NULL;
	reduction_function_data = NULL;
}

orbit_of_equations::~orbit_of_equations()
{
	Record_death();
	int i;
	
	if (Equations) {
		for (i = 0; i < used_length; i++) {
			FREE_int(Equations[i]);
		}
		FREE_pint(Equations);
	}
	if (prev) {
		FREE_int(prev);
	}
	if (label) {
		FREE_int(label);
	}
	if (data_tmp) {
		FREE_int(data_tmp);
	}
}

void orbit_of_equations::init(
		actions::action *A,
		algebra::field_theory::finite_field *F,
		induced_actions::action_on_homogeneous_polynomials
			*AonHPD,
	groups::strong_generators *SG,
	int *coeff_in,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_equations::init" << endl;
	}
	orbit_of_equations::A = A;
	orbit_of_equations::F = F;
	orbit_of_equations::SG = SG;
	orbit_of_equations::AonHPD = AonHPD;
	
	nb_monomials = AonHPD->HPD->get_nb_monomials();
	if (f_v) {
		cout << "orbit_of_equations::init "
				"nb_monomials = " << nb_monomials << endl;
	}
	sz = 1 + nb_monomials;
	sz_for_compare = 1 + nb_monomials;
	
	data_tmp = NEW_int(sz);
	
	if (f_v) {
		cout << "orbit_of_equations::init "
				"computing orbit of ";
		Int_vec_print(cout, coeff_in, nb_monomials);
		cout << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::init "
				"before compute_orbit" << endl;
	}
	compute_orbit(
			coeff_in,
			0 /* verbose_level */);
	if (f_v) {
		cout << "orbit_of_equations::init "
				"after compute_orbit" << endl;
	}

	if (f_v) {
		cout << "orbit_of_equations::init "
				"printing the orbit" << endl;
		print_orbit();
	}

	if (f_v) {
		cout << "orbit_of_equations::init done" << endl;
	}
}

void orbit_of_equations::map_an_equation(
		int *object_in, int *object_out,
	int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_equations::map_an_equation" << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation "
				"object_in=";
		Int_vec_print(cout, object_in + 1, nb_monomials);
		cout << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation Elt=" << endl;
		A->Group_element->element_print(Elt, cout);
	}
	AonHPD->compute_image_int_low_level(
		Elt,
		object_in + 1,
		object_out + 1,
		verbose_level - 2);
	object_out[0] = 0;
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation "
				"object_out=";
		Int_vec_print(cout, object_out + 1, nb_monomials);
		cout << endl;
	}
	if (f_has_reduction) {
		if (f_v) {
			cout << "orbit_of_equations::map_an_equation "
					"before reduction_function" << endl;
		}
		(*reduction_function)(
				object_out + 1,
				reduction_function_data);
		if (f_v) {
			cout << "orbit_of_equations::map_an_equation "
					"after reduction_function" << endl;
		}
	}
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation "
				"before F->PG_element_normalize_from_front" << endl;
	}
	F->Projective_space_basic->PG_element_normalize_from_front(
		object_out + 1, 1, nb_monomials);
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation "
				"after F->PG_element_normalize_from_front" << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation done" << endl;
	}
}

void orbit_of_equations::print_orbit()
{
	int i;
	
	cout << "orbit_of_equations::print_orbit We found an orbit of "
			"length " << used_length << endl;

	int *transporter;
	int *data;


	transporter = NEW_int(A->elt_size_in_int);
	data = NEW_int(A->make_element_size);

	for (i = 0; i < used_length; i++) {

		cout << i << " : ";

		Int_vec_print(cout,
				Equations[i] + 1,
				nb_monomials);

		cout << " : ";

		AonHPD->HPD->print_equation(cout, Equations[i] + 1);
		if (f_has_print_function) {
			(*print_function)(Equations[i], sz, print_function_data);
		}

		cout << " : ";


		get_transporter(
				i,
				transporter,
				0 /* verbose_level*/);

		A->Group_element->element_code_for_make_element(
				transporter, data);

		string s;

		s = Int_vec_stringify(data, A->make_element_size);

		cout << "\"" << s << "\"";

		cout << endl;
	}

	FREE_int(transporter);
	FREE_int(data);
}


void orbit_of_equations::print_orbit_as_equations_tex(
		std::ostream &ost)
{
	int i;

	ost << "We found an orbit of "
			"length " << used_length << "\\\\" << endl;
	for (i = 0; i < used_length; i++) {
		ost << i << " : ";
		//Int_vec_print(ost, Equations[i] + 1, nb_monomials);
		//ost << " : ";
		ost << "$";
		AonHPD->HPD->print_equation_tex(ost, Equations[i] + 1);
		ost << "$";
		ost << "\\\\";
		ost << endl;
	}
}


void orbit_of_equations::compute_orbit(
		int *coeff, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, cur, j, idx;
	int *cur_object;
	int *new_object;
	int *Q;
	int Q_len;

	if (f_v) {
		cout << "orbit_of_equations::compute_orbit" << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::compute_orbit sz=" << sz << endl;
	}


	cur_object = NEW_int(sz);
	new_object = NEW_int(sz);

	allocation_length = 1000;
	Equations = NEW_pint(allocation_length);
	prev = NEW_int(allocation_length);
	label = NEW_int(allocation_length);
	Equations[0] = NEW_int(sz);

	prev[0] = -1;
	label[0] = -1;


	if (f_v) {
		cout << "orbit_of_equations::compute_orbit "
				"init Equations[0]" << endl;
	}
	Int_vec_copy(
			coeff,
			Equations[0] + 1,
			nb_monomials);

	Equations[0][0] = 0;

	F->Projective_space_basic->PG_element_normalize_from_front(
			Equations[0] + 1, 1, nb_monomials);

	position_of_original_object = 0;

	used_length = 1;
	Q = NEW_int(allocation_length);
	Q[0] = 0;
	Q_len = 1;
	while (Q_len) {
		if (f_vv) {
			cout << "orbit_of_equations::compute_orbit  "
					"Q_len = " << Q_len << " : "
					"used_length=" << used_length
					<< " : ";
			Int_vec_print(cout, Q, Q_len);
			cout << endl;
		}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
		}
		Q_len--;

		Int_vec_copy(Equations[cur], cur_object, sz);


		for (j = 0; j < SG->gens->len; j++) {
			if (f_vvv) {
				cout << "orbit_of_equations::compute_orbit  "
						"applying generator " << j << endl;
			}

			map_an_equation(
					cur_object, new_object,
					SG->gens->ith(j), verbose_level);

			
			if (f_vvv) {
				cout << "orbit_of_equations::compute_orbit  "
						"before search_data" << endl;
			}
			if (search_data(new_object, idx, false)) {
				if (f_vvv) {
					cout << "orbit_of_equations::compute_orbit "
							"image object is already in the list, "
							"at position " << idx << endl;
				}
			}
			else {
				if (f_vvv) {
					cout << "orbit_of_equations::compute_orbit "
							"Found a new object : ";
					Int_vec_print(cout, new_object, sz);
					if (f_has_print_function) {
						(*print_function)(new_object, sz, print_function_data);
					}
					cout << endl;
				}
				
				if (used_length == allocation_length) {

					if (f_v) {
						cout << "orbit_of_equations::compute_orbit "
								"before reallocate" << endl;
					}
					reallocate(Q, Q_len, verbose_level - 1);
					if (f_v) {
						cout << "orbit_of_equations::compute_orbit "
								"after reallocate" << endl;
					}

				}
				for (i = used_length; i > idx; i--) {
					Equations[i] = Equations[i - 1];
				}
				for (i = used_length; i > idx; i--) {
					prev[i] = prev[i - 1];
				}
				for (i = used_length; i > idx; i--) {
					label[i] = label[i - 1];
				}
				Equations[idx] = NEW_int(sz);
				prev[idx] = cur;
				label[idx] = j;

				Int_vec_copy(new_object, Equations[idx], sz);

				if (position_of_original_object >= idx) {
					position_of_original_object++;
				}
				if (cur >= idx) {
					cur++;
				}
				for (i = 0; i < used_length + 1; i++) {
					if (prev[i] >= 0 && prev[i] >= idx) {
						prev[i]++;
					}
				}
				for (i = 0; i < Q_len; i++) {
					if (Q[i] >= idx) {
						Q[i]++;
					}
				}
				used_length++;
				if ((used_length % 10000) == 0) {
					cout << "orbit_of_equations::compute_orbit  "
							<< used_length << endl;
				}
				Q[Q_len++] = idx;
				if (f_vvv) {
					cout << "orbit_of_equations::compute_orbit  "
							"storing new equation at position "
							<< idx << endl;
				}

#if 0
				for (i = 0; i < used_length; i++) {
					cout << i << " : ";
					int_vec_print(cout, Equations[i], sz);
					cout << endl;
				}
#endif
			}
		}
	}
	if (f_v) {
		cout << "orbit_of_equations::compute_orbit "
				"found an orbit of length "
				<< used_length << endl;
	}


	FREE_int(Q);
	FREE_int(new_object);
	FREE_int(cur_object);
	if (f_v) {
		cout << "orbit_of_equations::compute_orbit done" << endl;
	}
}

void orbit_of_equations::reallocate(
		int *&Q, int Q_len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "orbit_of_equations::reallocate" << endl;
	}

	int al2 = allocation_length + 1000;
	int **Equations2;
	int *prev2;
	int *label2;
	int *Q2;
	int i;

	if (f_vv) {
		cout << "orbit_of_equations::compute_orbit "
				"reallocating to length " << al2 << endl;
	}
	Equations2 = NEW_pint(al2);
	prev2 = NEW_int(al2);
	label2 = NEW_int(al2);
	for (i = 0; i < allocation_length; i++) {
		Equations2[i] = Equations[i];
	}
	Int_vec_copy(prev, prev2, allocation_length);
	Int_vec_copy(label, label2, allocation_length);
	FREE_pint(Equations);
	FREE_int(prev);
	FREE_int(label);

	Equations = Equations2;
	prev = prev2;
	label = label2;

	Q2 = NEW_int(al2);
	Int_vec_copy(Q, Q2, Q_len);
	FREE_int(Q);
	Q = Q2;

	allocation_length = al2;

	if (f_v) {
		cout << "orbit_of_equations::reallocate done" << endl;
	}
}

void orbit_of_equations::get_table(
		std::string *&Table, std::string *&Headings,
		int &nb_rows, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_equations::get_table" << endl;
	}
	nb_rows = used_length;
	nb_cols = 4;
	int i;

	Table = new string [nb_rows * nb_cols];
	Headings = new string [nb_cols];

	for (i = 0; i < nb_rows; i++) {
		Table[i * nb_cols + 0] = std::to_string(i);
		Table[i * nb_cols + 1] = "\"" + Int_vec_stringify(Equations[i] + 1, sz - 1) + "\"";
		Table[i * nb_cols + 2] = std::to_string(prev[i]);
		Table[i * nb_cols + 3] = std::to_string(label[i]);
	}

	Headings[0] = "Row";
	Headings[1] = "Equation";
	Headings[2] = "Parent";
	Headings[3] = "Label";

	if (f_v) {
		cout << "orbit_of_equations::get_table done" << endl;
	}

}


void orbit_of_equations::get_transporter(
		int idx,
		int *transporter, int verbose_level)
// transporter is an element which maps 
// the orbit representative to the given orbit element.
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2;
	int idx0, idx1, l;

	if (f_v) {
		cout << "orbit_of_equations::get_transporter" << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::get_transporter "
				"position_of_original_object = "
				<< position_of_original_object << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->Group_element->element_one(Elt1, 0);
	idx1 = idx;
	idx0 = prev[idx1];
	if (f_v) {
		cout << "orbit_of_equations::get_transporter "
				"at idx1 = " << idx1 << " idx0 = " << idx0 << endl;
	}
	while (idx0 >= 0) {
		l = label[idx1];
		if (f_v) {
			cout << "orbit_of_equations::get_transporter "
					"at idx1 = " << idx1 << " idx0 = " << idx0 << " l=" << l << endl;
		}

		A->Group_element->element_mult(
				SG->gens->ith(l),
				Elt1,
				Elt2, 0);

		A->Group_element->element_move(Elt2, Elt1, 0);
		idx1 = idx0;
		idx0 = prev[idx1];
	}
	if (idx1 != position_of_original_object) {
		cout << "orbit_of_equations::get_transporter "
				"idx1 != position_of_original_object" << endl;
		exit(1);
	}
	A->Group_element->element_move(Elt1, transporter, 0);

	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "orbit_of_equations::get_transporter done" << endl;
	}
}

void orbit_of_equations::get_random_schreier_generator(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int len, r1, r2, pt1, pt2, pt3;
	int *E1, *E2, *E3, *E4, *E5;
	int *cur_object;
	int *new_object;
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "orbit_of_equations::get_random_schreier_generator" << endl;
	}
	E1 = NEW_int(A->elt_size_in_int);
	E2 = NEW_int(A->elt_size_in_int);
	E3 = NEW_int(A->elt_size_in_int);
	E4 = NEW_int(A->elt_size_in_int);
	E5 = NEW_int(A->elt_size_in_int);
	cur_object = NEW_int(sz);
	new_object = NEW_int(sz);
	len = used_length;
	pt1 = position_of_original_object;
	
	// get a random coset:
	r1 = Os.random_integer(len);
	get_transporter(r1, E1, 0);
		
	// get a random generator:
	r2 = Os.random_integer(SG->gens->len);
	if (f_vv) {
		cout << "r2=" << r2 << endl;
	}
	if (f_vv) {
		cout << "random coset " << r1
				<< ", random generator " << r2 << endl;
	}
	
	A->Group_element->element_mult(
			E1, SG->gens->ith(r2), E2, 0);

	// compute image of original subspace under E2:
	Int_vec_copy(
			Equations[pt1],
			cur_object,
			sz);

	map_an_equation(
			cur_object,
			new_object,
			E2, 0 /* verbose_level*/);

	if (search_data(new_object, pt2, false)) {
		if (f_vv) {
			cout << "orbit_of_equations::get_random_schreier_generator "
					"the new object is at position " << pt2 << endl;
		}
	}
	else {
		cout << "orbit_of_equations::get_random_schreier_generator "
				"image space is not found in the orbit" << endl;
		exit(1);
	}
	

	get_transporter(pt2, E3, 0);
	A->Group_element->element_invert(E3, E4, 0);
	A->Group_element->element_mult(E2, E4, E5, 0);

	// test:
	map_an_equation(
			cur_object, new_object, E5,
			0 /* verbose_level*/);
	if (search_data(new_object, pt3, false)) {
		if (f_vv) {
			cout << "testing: new object is at position " << pt3 << endl;
		}
	}
	else {
		cout << "orbit_of_equations::get_random_schreier_generator "
				"(testing) image space is not found in the orbit" << endl;
		exit(1);
	}

	if (pt3 != position_of_original_object) {
		cout << "orbit_of_equations::get_random_schreier_generator "
				"pt3 != position_of_original_subspace" << endl;
		exit(1);
	}



	A->Group_element->element_move(E5, Elt, 0);


	FREE_int(E1);
	FREE_int(E2);
	FREE_int(E3);
	FREE_int(E4);
	FREE_int(E5);
	FREE_int(cur_object);
	FREE_int(new_object);
	if (f_v) {
		cout << "orbit_of_equations::get_random_schreier_generator "
				"done" << endl;
	}
}

void orbit_of_equations::get_canonical_form(
		int *canonical_equation,
		int *transporter_to_canonical_form,
		groups::strong_generators
			*&gens_stab_of_canonical_equation,
			algebra::ring_theory::longinteger_object
			&full_group_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx = 0;

	if (f_v) {
		cout << "orbit_of_equations::get_canonical_form" << endl;
		cout << "orbit_of_equations::get_canonical_form verbose_level = " << verbose_level << endl;
	}

	Int_vec_copy(
			Equations[0] + 1,
			canonical_equation,
			nb_monomials);

	if (f_v) {
		cout << "orbit_of_equations::get_canonical_form "
				"before stabilizer_any_point" << endl;
	}
	gens_stab_of_canonical_equation =
			stabilizer_any_point(
		full_group_order, idx,
		verbose_level - 1);
	if (f_v) {
		cout << "orbit_of_equations::get_canonical_form "
				"after stabilizer_any_point" << endl;
	}


	if (f_v) {
		cout << "orbit_of_equations::get_canonical_form "
				"before get_transporter" << endl;
	}
	get_transporter(idx, transporter_to_canonical_form, 0);
	if (f_v) {
		cout << "orbit_of_equations::get_canonical_form "
				"after get_transporter" << endl;
	}


	if (f_v) {
		cout << "orbit_of_equations::get_canonical_form done" << endl;
	}
}

groups::strong_generators *orbit_of_equations::stabilizer_orbit_rep(
		algebra::ring_theory::longinteger_object &full_group_order,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::strong_generators *gens;
	groups::sims *Stab;

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep" << endl;
	}

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep "
				"default action = " << A->label << endl;
		cout << "orbit_of_equations::stabilizer_orbit_rep "
				"full_group_order = " << full_group_order << endl;
	}
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep "
				"before stabilizer_orbit_rep_work" << endl;
	}
	stabilizer_orbit_rep_work(
			A /* default_action */, full_group_order,
		Stab, verbose_level - 2);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep "
				"after stabilizer_orbit_rep_work" << endl;
	}

	algebra::ring_theory::longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep "
				"found a stabilizer group of order "
				<< stab_order << endl;
	}

	gens = NEW_OBJECT(groups::strong_generators);
	gens->init(A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep done" << endl;
	}
	return gens;
}

void orbit_of_equations::stabilizer_orbit_rep_work(
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &go,
		groups::sims *&Stab, int verbose_level)
// this function allocates a sims structure into Stab.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);


	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep_work" << endl;
	}

	Stab = NEW_OBJECT(groups::sims);
	algebra::ring_theory::longinteger_object cur_go, target_go;
	algebra::ring_theory::longinteger_domain D;
	int len, r, cnt = 0, f_added, drop_out_level, image;
	int *residue;
	int *E1;
	
	
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep_work "
				"computing the stabilizer inside a group of order "
				<< go << " in action ";
		default_action->print_info();
		cout << endl;
	}
	E1 = NEW_int(default_action->elt_size_in_int);
	residue = NEW_int(default_action->elt_size_in_int);
	len = used_length;
	D.integral_division_by_int(go, len, target_go, r);
	if (r) {	
		cout << "orbit_of_equations::stabilizer_orbit_rep_work "
				"error: orbit length "
				"does not divide group order" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "orbit_of_equations::stabilizer_orbit_rep_work "
				"expecting a group of order " << target_go << endl;
	}
	
	Stab->init(default_action, verbose_level - 2);
	Stab->init_trivial_group(verbose_level - 1);
	while (true) {
		Stab->group_order(cur_go);
		if (D.compare(cur_go, target_go) == 0) {
			break;
		}
		if (cnt % 2 || Stab->nb_gen[0] == 0) {
			get_random_schreier_generator(
					E1, 0 /* verbose_level */);
			if (f_vvv) {
				cout << "orbit_of_equations::stabilizer_orbit_rep_work "
						"created random Schreier generator" << endl;
				//default_action->element_print(E1, cout);
			}
		}
		else {
			Stab->random_schreier_generator(E1, 0 /* verbose_level */);
			//A->element_move(Stab->schreier_gen, E1, 0);
			if (f_v4) {
				cout << "orbit_of_equations::stabilizer_orbit_rep_work "
						"created random schreier generator from sims"
						<< endl;
				//default_action->element_print(E1, cout);
			}
		}



		if (Stab->strip(
				E1, residue, drop_out_level, image,
				0 /*verbose_level - 3*/)) {
			if (f_vvv) {
				cout << "orbit_of_equations::stabilizer_orbit_rep_work "
						"element strips through" << endl;
				if (false) {
					cout << "residue:" << endl;
					A->Group_element->element_print(residue, cout);
					cout << endl;
				}
			}
			f_added = false;
		}
		else {
			f_added = true;
			if (f_vvv) {
				cout << "orbit_of_equations::stabilizer_orbit_rep_work "
						"element needs to be inserted at level = "
					<< drop_out_level << " with image " << image << endl;
				if (false) {
					A->Group_element->element_print(residue, cout);
					cout  << endl;
				}
			}
			Stab->add_generator_at_level(residue, drop_out_level,
					verbose_level - 4);
		}
		Stab->group_order(cur_go);
		if ((f_vv && f_added) || f_vvv) {
			cout << "iteration " << cnt
				<< " the new group order is " << cur_go
				<< " expecting a group of order " << target_go << endl; 
		}
		cnt++;
	}
	FREE_int(E1);
	FREE_int(residue);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_orbit_rep_work finished" << endl;
	}
}


groups::strong_generators *orbit_of_equations::stabilizer_any_point(
		algebra::ring_theory::longinteger_object &full_group_order,
		int idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	groups::strong_generators *gens0;
	groups::strong_generators *gens;
	int *transporter;
	int *transporter_inv;

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point" << endl;
	}

	transporter = NEW_int(A->elt_size_in_int);
	transporter_inv = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"before stabilizer_orbit_rep" << endl;
	}
	gens0 = stabilizer_orbit_rep(
			full_group_order, verbose_level - 2);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"after stabilizer_orbit_rep" << endl;
	}

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"before get_transporter" << endl;
	}
	get_transporter(idx,
			transporter_inv, 0 /* verbose_level */);
	// transporter_inv is an element which maps
	// the orbit representative to the given object.
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"after get_transporter" << endl;
	}

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"before A->Group_element->element_invert" << endl;
	}
	A->Group_element->element_invert(
			transporter_inv, transporter, 0);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"after A->Group_element->element_invert" << endl;
	}




	gens = NEW_OBJECT(groups::strong_generators);


	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"before gens->init_generators_for_the_conjugate_group_aGav" << endl;
	}
	gens->init_generators_for_the_conjugate_group_aGav(
			gens0,
		transporter,
		verbose_level - 2);
	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point "
				"after gens->init_generators_for_the_conjugate_group_aGav" << endl;
	}

	FREE_int(transporter);
	FREE_int(transporter_inv);

	if (f_v) {
		cout << "orbit_of_equations::stabilizer_any_point done" << endl;
	}
	FREE_OBJECT(gens0);
	return gens;
}


int orbit_of_equations::search_equation(
		int *eqn, int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;
	int p[1];
	p[0] = sz_for_compare;
	int ret;
	int *data;

	if (f_v) {
		cout << "orbit_of_equations::search_data" << endl;
	}
	if (f_v) {
		cout << "The equation is:";
		AonHPD->HPD->print_equation_simple(cout, eqn);
		cout << endl;
	}
	data = NEW_int(sz_for_compare);
	data[0] = 0;

	Int_vec_copy(eqn,
			data + 1,
			AonHPD->HPD->get_nb_monomials());

	if (Sorting.vec_search(
			(void **)Equations,
			orbit_of_equations_compare_func,
			p,
			used_length, data, idx,
			0 /* verbose_level */)) {
		if (f_v) {
			cout << "orbit_of_equations::search_data "
					"we found the equation at " << idx << endl;
		}
		ret = true;
	}
	else {
		if (f_v) {
			cout << "orbit_of_equations::search_data "
					"we did not find the equation" << endl;
		}
		ret = false;
	}
	if (f_v) {
		cout << "orbit_of_equations::search_data done" << endl;
	}
	FREE_int(data);
	return ret;
}


int orbit_of_equations::search_data(
		int *data, int &idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	other::data_structures::sorting Sorting;
	int p[1];
	p[0] = sz_for_compare;
	int ret;

	if (f_v) {
		cout << "orbit_of_equations::search_data" << endl;
	}
	if (f_v) {
		cout << "The data set is:";
		Int_vec_print(cout, data, sz_for_compare);
		cout << endl;
	}
	if (Sorting.vec_search(
			(void **)Equations,
			orbit_of_equations_compare_func,
			p,
			used_length, data, idx,
			0 /* verbose_level */)) {
		if (f_v) {
			cout << "orbit_of_equations::search_data "
					"we found the equation at " << idx << endl;
		}
		ret = true;
	}
	else {
		if (f_v) {
			cout << "orbit_of_equations::search_data "
					"we did not find the equation" << endl;
		}
		ret = false;
	}
	if (f_v) {
		cout << "orbit_of_equations::search_data done" << endl;
	}
	return ret;
}

void orbit_of_equations::save_csv(
		std::string &fname, int verbose_level)
{
	int i;
	int *Data;
	other::orbiter_kernel_system::file_io Fio;

	Data = NEW_int(used_length * nb_monomials);
	for (i = 0; i < used_length; i++) {
		Int_vec_copy(
				Equations[i] + 1,
				Data + i * nb_monomials,
				nb_monomials);
	}
	Fio.Csv_file_support->int_matrix_write_csv(
			fname, Data, used_length, nb_monomials);
}


static int orbit_of_equations_compare_func(
		void *a, void *b, void *data)
{
	int *A = (int *)a;
	int *B = (int *)b;
	int *p = (int *) data;
	int n = *p;
	int i;

	for (i = 0; i < n; i++) {
		if (A[i] < B[i]) {
			return 1;
		}
		if (A[i] > B[i]) {
			return -1;
		}
	}
	return 0;
}

}}}




