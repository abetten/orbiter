// orbit_of_equations.C
// 
// Anton Betten
// May 29, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


orbit_of_equations::orbit_of_equations()
{
	null();
}

orbit_of_equations::~orbit_of_equations()
{
	freeself();
}

void orbit_of_equations::null()
{
	A = NULL;
	F = NULL;
	SG = NULL;
	AonHPD = NULL;
	Equations = NULL;
	prev = NULL;
	label = NULL;
	data_tmp = NULL;
}

void orbit_of_equations::freeself()
{
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
	null();
}

void orbit_of_equations::init(action *A, finite_field *F, 
	action_on_homogeneous_polynomials *AonHPD, 
	strong_generators *SG, int *coeff_in, 
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
	
	nb_monomials = AonHPD->HPD->nb_monomials;
	sz = 1 + nb_monomials;
	sz_for_compare = 1 + nb_monomials;
	
	data_tmp = NEW_int(sz);
	
	if (f_v) {
		cout << "orbit_of_equations::init before compute" << endl;
		}
	compute_orbit(coeff_in, verbose_level);
	if (f_v) {
		cout << "orbit_of_equations::init after compute" << endl;
		}

	if (f_v) {
		cout << "orbit_of_equations::init printing the orbit" << endl;
		print_orbit();
		}

	if (f_v) {
		cout << "orbit_of_equations::init done" << endl;
		}
}

void orbit_of_equations::map_an_equation(int *object_in, int *object_out, 
	int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbit_of_equations::map_an_equation" << endl;
		}
	AonHPD->compute_image_int_low_level(
		Elt, object_in + 1, object_out + 1, verbose_level - 2);
	F->PG_element_normalize_from_front(
		object_out + 1, 1, nb_monomials);
	if (f_v) {
		cout << "orbit_of_equations::map_an_equation done" << endl;
		}
}

void orbit_of_equations::print_orbit()
{
	int i;
	
	cout << "orbit_of_equations::print_orbit We found an orbit of "
			"length " << used_length << endl;
	for (i = 0; i < used_length; i++) {
		cout << i << " : ";
		int_vec_print(cout, Equations[i] + 1, nb_monomials);
		cout << " : ";
		AonHPD->HPD->print_equation(cout, Equations[i] + 1);
		cout << endl;
		}
}

void orbit_of_equations::compute_orbit(int *coeff, int verbose_level)
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
		cout << "orbit_of_equations::compute_orbit init Equations[0]" << endl;
		}
	int_vec_copy(coeff, Equations[0] + 1, nb_monomials);
	Equations[0][0] = 0;
	F->PG_element_normalize_from_front(
		Equations[0] + 1, 1, nb_monomials);

	position_of_original_object = 0;

	used_length = 1;
	Q = NEW_int(allocation_length);
	Q[0] = 0;
	Q_len = 1;
	while (Q_len) {
		if (f_vv) {
			cout << "orbit_of_equations::compute_orbit  Q_len = "
					<< Q_len << " : used_length=" << used_length << " : ";
			int_vec_print(cout, Q, Q_len);
			cout << endl;
			}
		cur = Q[0];
		for (i = 1; i < Q_len; i++) {
			Q[i - 1] = Q[i];
			}
		Q_len--;

		int_vec_copy(Equations[cur], cur_object, sz);


		for (j = 0; j < SG->gens->len; j++) {
			if (f_vvv) {
				cout << "orbit_of_equations::compute_orbit  "
						"applying generator " << j << endl;
				}

			map_an_equation(cur_object, new_object,
					SG->gens->ith(j),  verbose_level - 4);

			
			if (search_data(new_object, idx)) {
				if (f_vvv) {
					cout << "orbit_of_equations::compute_orbit "
							"n e w object is already in the list, "
							"at position " << idx << endl;
					}
				}
			else {
				if (f_vvv) {
					cout << "orbit_of_equations::compute_orbit "
							"Found a n e w object : ";
					int_vec_print(cout, new_object, sz);
					cout << endl;
					}
				
				if (used_length == allocation_length) {
					int al2 = allocation_length + 1000;
					int **Equations2;
					int *prev2;
					int *label2;
					int *Q2;
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
					int_vec_copy(prev, prev2, allocation_length);
					int_vec_copy(label, label2, allocation_length);
					FREE_pint(Equations);
					FREE_int(prev);
					FREE_int(label);
					Equations = Equations2;
					prev = prev2;
					label = label2;
					Q2 = NEW_int(al2);
					int_vec_copy(Q, Q2, Q_len);
					FREE_int(Q);
					Q = Q2;
					allocation_length = al2;
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

				int_vec_copy(new_object, Equations[idx], sz);

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
							"storing n e w equation at position "
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
		cout << "orbit_of_equations::compute_orbit found an orbit "
				"of length " << used_length << endl;
		}


	FREE_int(Q);
	FREE_int(new_object);
	FREE_int(cur_object);
	if (f_v) {
		cout << "orbit_of_equations::compute_orbit done" << endl;
		}
}

void orbit_of_equations::get_transporter(int idx,
		int *transporter, int verbose_level)
// transporter is an element which maps 
// the orbit representative to the given subspace.
{
	int f_v = (verbose_level >= 1);
	int *Elt1, *Elt2;
	int idx0, idx1, l;

	if (f_v) {
		cout << "orbit_of_equations::get_transporter" << endl;
		}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	A->element_one(Elt1, 0);
	idx1 = idx;
	idx0 = prev[idx1];
	while (idx0 >= 0) {
		l = label[idx1];
		A->element_mult(SG->gens->ith(l), Elt1, Elt2, 0);
		A->element_move(Elt2, Elt1, 0);
		idx1 = idx0;
		idx0 = prev[idx1];
		}
	if (idx1 != position_of_original_object) {
		cout << "orbit_of_equations::get_transporter "
				"idx1 != position_of_original_object" << endl;
		exit(1);
		}
	A->element_move(Elt1, transporter, 0);

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
	int f_vv = FALSE; //(verbose_level >= 2);
	int len, r1, r2, pt1, pt2, pt3;
	int *E1, *E2, *E3, *E4, *E5;
	int *cur_object;
	int *new_object;
	
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
	r1 = random_integer(len);
	get_transporter(r1, E1, 0);
		
	// get a random generator:
	r2 = random_integer(SG->gens->len);
	if (f_vv) {
		cout << "r2=" << r2 << endl;
		}
	if (f_vv) {
		cout << "random coset " << r1
				<< ", random generator " << r2 << endl;
		}
	
	A->element_mult(E1, SG->gens->ith(r2), E2, 0);

	// compute image of original subspace under E2:
	int_vec_copy(Equations[pt1], cur_object, sz);

	map_an_equation(cur_object, new_object, E2, 0 /* verbose_level*/);

	if (search_data(new_object, pt2)) {
		if (f_vv) {
			cout << "n e w object is at position " << pt2 << endl;
			}
		}
	else {
		cout << "orbit_of_equations::get_random_schreier_generator "
				"image space is not found in the orbit" << endl;
		exit(1);
		}
	

	get_transporter(pt2, E3, 0);
	A->element_invert(E3, E4, 0);
	A->element_mult(E2, E4, E5, 0);

	// test:
	map_an_equation(cur_object, new_object, E5, 0 /* verbose_level*/);
	if (search_data(new_object, pt3)) {
		if (f_vv) {
			cout << "testing: n e w object is at position " << pt3 << endl;
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



	A->element_move(E5, Elt, 0);


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

void orbit_of_equations::compute_stabilizer(action *default_action, 
	longinteger_object &go, 
	sims *&Stab, int verbose_level)
// this function allocates a sims structure into Stab.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);


	if (f_v) {
		cout << "orbit_of_equations::compute_stabilizer" << endl;
		}

	Stab = NEW_OBJECT(sims);
	longinteger_object cur_go, target_go;
	longinteger_domain D;
	int len, r, cnt = 0, f_added, drop_out_level, image;
	int *residue;
	int *E1;
	
	
	if (f_v) {
		cout << "orbit_of_equations::compute_stabilizer computing "
				"stabilizer inside a group of order " << go << " in action ";
		default_action->print_info();
		cout << endl;
		}
	E1 = NEW_int(default_action->elt_size_in_int);
	residue = NEW_int(default_action->elt_size_in_int);
	len = used_length;
	D.integral_division_by_int(go, len, target_go, r);
	if (r) {	
		cout << "orbit_of_equations::compute_stabilizer orbit length "
				"does not divide group order" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "orbit_of_equations::compute_stabilizer expecting "
				"group of order " << target_go << endl;
		}
	
	Stab->init(default_action);
	Stab->init_trivial_group(verbose_level - 1);
	while (TRUE) {
		Stab->group_order(cur_go);
		if (D.compare(cur_go, target_go) == 0) {
			break;
			}
		if (cnt % 2 || Stab->nb_gen[0] == 0) {
			get_random_schreier_generator(E1, 0 /* verbose_level */);
			if (f_vvv) {
				cout << "orbit_of_equations::compute_stabilizer "
						"created random Schreier generator" << endl;
				//default_action->element_print(E1, cout);
				}
			}
		else {
			Stab->random_schreier_generator(0 /* verbose_level */);
			A->element_move(Stab->schreier_gen, E1, 0);
			if (f_v4) {
				cout << "orbit_of_equations::compute_stabilizer "
						"created random schreier generator from sims"
						<< endl;
				//default_action->element_print(E1, cout);
				}
			}



		if (Stab->strip(E1, residue, drop_out_level, image,
				0 /*verbose_level - 3*/)) {
			if (f_vvv) {
				cout << "orbit_of_equations::compute_stabilizer "
						"element strips through" << endl;
				if (FALSE) {
					cout << "residue:" << endl;
					A->element_print(residue, cout);
					cout << endl;
					}
				}
			f_added = FALSE;
			}
		else {
			f_added = TRUE;
			if (f_vvv) {
				cout << "orbit_of_equations::compute_stabilizer "
						"element needs to be inserted at level = "
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					A->element_print(residue, cout);
					cout  << endl;
					}
				}
			Stab->add_generator_at_level(residue, drop_out_level,
					verbose_level - 4);
			}
		Stab->group_order(cur_go);
		if ((f_vv && f_added) || f_vvv) {
			cout << "iteration " << cnt
				<< " the n e w group order is " << cur_go
				<< " expecting a group of order " << target_go << endl; 
			}
		cnt++;
		}
	FREE_int(E1);
	FREE_int(residue);
	if (f_v) {
		cout << "orbit_of_equations::compute_stabilizer finished" << endl;
		}
}

strong_generators *orbit_of_equations::stabilizer_orbit_rep(
	longinteger_object &full_group_order, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	strong_generators *gens;
	sims *Stab;

	if (f_v) {
		cout << "orbit_of_equations::generators_for_"
				"stabilizer_of_orbit_rep" << endl;
		}

	compute_stabilizer(A /* default_action */, full_group_order, 
		Stab, 0 /*verbose_level*/);

	longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "orbit_of_equations::generators_for_stabilizer_"
				"of_orbit_rep found a stabilizer group of order "
				<< stab_order << endl;
		}
	
	gens = NEW_OBJECT(strong_generators);
	gens->init(A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "orbit_of_equations::generators_for_stabilizer_of_"
				"orbit_rep done" << endl;
		}
	return gens;
}



int orbit_of_equations::search_data(int *data, int &idx)
{
	sorting Sorting;
	int p[1];
	p[0] = sz_for_compare;

	if (Sorting.vec_search((void **)Equations,
			orbit_of_equations_compare_func,
			p,
		used_length, data, idx, 0 /* verbose_level */)) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}

void orbit_of_equations::save_csv(const char *fname, int verbose_level)
{
	int i;
	int *Data;

	Data = NEW_int(used_length * nb_monomials);
	for (i = 0; i < used_length; i++) {
		int_vec_copy(Equations[i] + 1,
				Data + i * nb_monomials, nb_monomials);
		}
	int_matrix_write_csv(fname, Data, used_length, nb_monomials);
}


int orbit_of_equations_compare_func(void *a, void *b, void *data)
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

}}



