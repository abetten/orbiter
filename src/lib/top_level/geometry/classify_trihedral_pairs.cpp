// classify_trihedral_pairs.cpp
// 
// Anton Betten
//
// October 9, 2017
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


classify_trihedral_pairs::classify_trihedral_pairs()
{
	q = 0;
	F = NULL;
	Surf_A = NULL;
	Surf = NULL;
	gens_type1 = NULL;
	gens_type2 = NULL;
	Poset1 = NULL;
	Poset2 = NULL;
	orbits_on_trihedra_type1 = NULL;
	orbits_on_trihedra_type2 = NULL;
	Flag_orbits = NULL;
	nb_orbits_trihedral_pairs = 0;
	Trihedral_pairs = NULL;
	null();
}

classify_trihedral_pairs::~classify_trihedral_pairs()
{
	freeself();
}

void classify_trihedral_pairs::null()
{
}

void classify_trihedral_pairs::freeself()
{
	if (gens_type1) {
		FREE_OBJECT(gens_type1);
	}
	if (gens_type2) {
		FREE_OBJECT(gens_type2);
	}
	if (Poset1) {
		FREE_OBJECT(Poset1);
	}
	if (Poset2) {
		FREE_OBJECT(Poset2);
	}
	if (orbits_on_trihedra_type1) {
		FREE_OBJECT(orbits_on_trihedra_type1);
	}
	if (orbits_on_trihedra_type2) {
		FREE_OBJECT(orbits_on_trihedra_type2);
	}
	if (Flag_orbits) {
		FREE_OBJECT(Flag_orbits);
	}
	if (Trihedral_pairs) {
		FREE_OBJECT(Trihedral_pairs);
	}
	null();
}

void classify_trihedral_pairs::init(surface_with_action *Surf_A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_trihedral_pairs::init" << endl;
	}
	classify_trihedral_pairs::Surf_A = Surf_A;
	F = Surf_A->F;
	q = F->q;
	A = Surf_A->A;
	Surf = Surf_A->Surf;
	
	
	if (f_v) {
		cout << "classify_trihedral_pairs::init computing stabilizer "
				"of three collinear points" << endl;
	}
	gens_type1 = NEW_OBJECT(strong_generators);
	gens_type1->generators_for_stabilizer_of_three_collinear_points_in_PGL4(
		A,
		A->G.matrix_grp, verbose_level - 1);

	if (f_v) {
		cout << "classify_trihedral_pairs::init computing stabilizer "
				"of a triangle of points" << endl;
	}
	gens_type2 = NEW_OBJECT(strong_generators);
	gens_type2->generators_for_stabilizer_of_triangle_in_PGL4(A, 
		A->G.matrix_grp, verbose_level - 1);

	longinteger_object go1, go2;

	gens_type1->group_order(go1);
	gens_type2->group_order(go2);



	if (f_v) {
		cout << "The group 1 has order " ;
		go1.print_not_scientific(cout); 
		cout << "\\\\" << endl;
		cout << "generators:" << endl;
		gens_type1->print_generators_tex(cout);

		cout << "The group 2 has order " ;
		go2.print_not_scientific(cout); 
		cout << "\\\\" << endl;
		cout << "generators:" << endl;
		gens_type2->print_generators_tex(cout);
	}




	if (f_v) {
		cout << "classify_trihedral_pairs::init done" << endl;
	}
}



void classify_trihedral_pairs::classify_orbits_on_trihedra(
		poset_classification_control *Control1,
		poset_classification_control *Control2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "classify_trihedral_pairs::classify_orbits_on_trihedra" << endl;
	}

	if (f_v) {
		cout << "classify_trihedral_pairs::classify_orbits_on_trihedra "
				"computing orbits on 3-subsets of points (type 1):" << endl;
	}

	Poset1 = NEW_OBJECT(poset);
	Poset1->init_subset_lattice(A, A, gens_type1,
			verbose_level);

	Poset1->add_testing_without_group(
			classify_trihedral_pairs_early_test_function_type1,
				this /* void *data */,
				verbose_level);


	orbits_on_trihedra_type1 = NEW_OBJECT(poset_classification);

	orbits_on_trihedra_type1->compute_orbits_on_subsets(
		3, /* target_depth */
		//"", /* const char *prefix, */
		Control1,
		Poset1,
		0 /*verbose_level*/);

	if (f_v) {
		cout << "classify_trihedral_pairs::classify_orbits_on_trihedra "
				"computing orbits on 3-subsets of points (type 1) done. "
				"We found "
			<< orbits_on_trihedra_type1->nb_orbits_at_level(3)
			<< " orbits on 3-subsets" << endl;
	}

	if (f_v) {
		cout << "classify_trihedral_pairs::classify_orbits_on_trihedra "
				"computing orbits on 3-subsets of points (type 2):" << endl;
	}
	
	Poset2 = NEW_OBJECT(poset);
	Poset2->init_subset_lattice(A, A, gens_type2,
			verbose_level);

	Poset2->add_testing_without_group(
			classify_trihedral_pairs_early_test_function_type2,
				this /* void *data */,
				verbose_level);

	orbits_on_trihedra_type2 = NEW_OBJECT(poset_classification);

	orbits_on_trihedra_type2->compute_orbits_on_subsets(
		3, /* target_depth */
		//"", /* const char *prefix, */
		Control2,
		Poset2,
		0 /*verbose_level*/);

	if (f_v) {
		cout << "classify_trihedral_pairs::classify_orbits_on_trihedra "
				"computing orbits on 3-subsets of points (type 2) done. "
				"We found "
			<< orbits_on_trihedra_type2->nb_orbits_at_level(3)
			<< " orbits on 3-subsets" << endl;
	}

	if (f_v) {
		cout << "classify_trihedral_pairs::classify_orbits_on_trihedra done" << endl;
	}
}


void classify_trihedral_pairs::report(ostream &ost)
{
	cout << "classify_trihedral_pairs::report "
			"before list_orbits_on_trihedra_type1" << endl;
	list_orbits_on_trihedra_type1(ost);

	cout << "classify_trihedral_pairs::report "
			"before list_orbits_on_trihedra_type2" << endl;
	list_orbits_on_trihedra_type2(ost);

	cout << "classify_trihedral_pairs::report "
			"before print_trihedral_pairs no stabs" << endl;
	print_trihedral_pairs(ost,
			FALSE /* f_with_stabilizers */);

	cout << "classify_trihedral_pairs::report "
			"before print_trihedral_pairs with stabs" << endl;
	print_trihedral_pairs(ost,
			TRUE /* f_with_stabilizers */);
}

void classify_trihedral_pairs::list_orbits_on_trihedra_type1(ostream &ost)
{
	int i, l;

	l = orbits_on_trihedra_type1->nb_orbits_at_level(3);

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Classification of Double Triplets of "
			"type 1 in $\\PG(3," << q << ")$}" << endl;



	{
		longinteger_object go;
		gens_type1->group_order(go);

		ost << "The order of the group of type 1 is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}



	longinteger_domain D;
	longinteger_object ol, Ol;
	Ol.create(0, __FILE__, __LINE__);

	ost << "The group of type 1 has " 
		<< l 
		<< " orbits on double triplets of type 1 in "
				"$\\PG(3," << q << ").$" << endl << endl;
	for (i = 0; i < l; i++) {
		set_and_stabilizer *R;

		R = orbits_on_trihedra_type1->get_set_and_stabilizer(
				3 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);
		orbits_on_trihedra_type1->orbit_length(
				i /* node */,
				3 /* level */,
				ol);
		D.add_in_place(Ol, ol);
		
		ost << "$" << i << " / " << l << "$ $" << endl;
		R->print_set_tex(ost);
		ost << "$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		FREE_OBJECT(R);
	}

	ost << "The overall number of double triplets of type 1 "
			"in $\\PG(3," << q << ")$ is: " << Ol << "\\\\" << endl;
}

void classify_trihedral_pairs::list_orbits_on_trihedra_type2(ostream &ost)
{
	int i, l;

	l = orbits_on_trihedra_type2->nb_orbits_at_level(3);

	//ost << "\\clearpage" << endl;
	ost << "\\subsection*{Classification of Double Triplets of "
			"type 2 in $\\PG(3," << q << ")$}" << endl;



	{
		longinteger_object go;
		gens_type2->group_order(go);

		ost << "The order of the group of type 2 is ";
		go.print_not_scientific(ost);
		ost << "\\\\" << endl;

		ost << "\\bigskip" << endl;
	}



	longinteger_domain D;
	longinteger_object ol, Ol;
	Ol.create(0, __FILE__, __LINE__);

	ost << "The group of type 2 has " 
		<< l 
		<< " orbits on double triplets of type 2 "
				"in $\\PG(3," << q << ").$" << endl << endl;
	for (i = 0; i < l; i++) {
		set_and_stabilizer *R;

		R = orbits_on_trihedra_type2->get_set_and_stabilizer(
				3 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);
		orbits_on_trihedra_type2->orbit_length(
				i /* node */,
				3 /* level */,
				ol);
		D.add_in_place(Ol, ol);
		
		ost << "$" << i << " / " << l << "$ $" << endl;
		R->print_set_tex(ost);
		ost << "$ orbit length $";
		ol.print_not_scientific(ost);
		ost << "$\\\\" << endl;

		FREE_OBJECT(R);
	}

	ost << "The overall number of double triplets of type 2 "
			"in $\\PG(3," << q << ")$ is: " << Ol << "\\\\" << endl;
}

void classify_trihedral_pairs::early_test_func_type1(
		long int *S, int len,
		long int *candidates, int nb_candidates,
		long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	long int a, rk; //, idx; //, f_bad, rk0, ;
	long int Lines[9];
	long int Lines2[9];
	sorting Sorting;
		
	if (f_v) {
		cout << "classify_trihedral_pairs::early_test_func_type1 "
				"checking set ";
		lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len > 2) {
		cout << "classify_trihedral_pairs::early_test_func_type1 "
				"len > 2" << endl;
		exit(1);
	}

#if 0
	rk0 = Surf->P->line_of_intersection_of_two_planes_in_three_
			space_using_dual_coordinates(0, 1, 0 /* verbose_level */);
	if (f_vv) {
		cout << "surface_with_action::early_test_func_type1 "
				"rk0 = " << rk0 << endl;
	}
#endif

	for (i = 0; i < len; i++) {
		Lines[i * 3 + 0] = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[i], 0, 0 /* verbose_level */);
		Lines[i * 3 + 1] = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[i], 1, 0 /* verbose_level */);
		Lines[i * 3 + 2] = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[i], 5, 0 /* verbose_level */);
	}
	if (f_vv) {
		cout << "classify_trihedral_pairs::early_test_func_type1 Lines=" << endl;
		lint_matrix_print(Lines, len, 3);
	}

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];

		if (a == 0 || a == 1 || a == 5) {
			continue;
		}
		
		if (f_vv) {
			cout << "classify_trihedral_pairs::early_test_func_type1 "
					"testing a=" << a << endl;
		}

		for (j = 0; j < len; j++) {
			if (a == S[j]) {
				break;
			}
		}
		if (j < len) {
			continue;
		}
		
		lint_vec_copy(Lines, Lines2, len * 3);
		
		rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, 0, 0 /* verbose_level */);


#if 0
		if (rk == rk0) {
			if (f_vv) {
				cout << "intersects 0 in the bad line" << endl;
			}
			continue;
		}
#endif
		Lines2[len * 3 + 0] = rk;

		rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, 1, 0 /* verbose_level */);
#if 0
		if (rk == rk0) {
			if (f_vv) {
				cout << "intersects 1 in the bad line" << endl;
			}
			continue;
		}
#endif
		Lines2[len * 3 + 1] = rk;

		rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, 5, 0 /* verbose_level */);
#if 0
		if (rk == rk0) {
			if (f_vv) {
				cout << "intersects 5 in the bad line" << endl;
			}
			continue;
		}
#endif
		Lines2[len * 3 + 2] = rk;

		if (f_vv) {
			cout << "classify_trihedral_pairs::early_test_func_type1 "
					"Lines2=" << endl;
			lint_matrix_print(Lines2, len + 1, 3);
		}


		Sorting.lint_vec_heapsort(Lines2, (len + 1) * 3);

		for (j = 1; j < (len + 1) * 3; j++) {
			if (Lines2[j] == Lines2[j - 1]) {
				if (f_vv) {
					cout << "classify_trihedral_pairs::early_test_func_type1 "
							"repeated line" << endl;
				}
				break;
			}
		}
		if (j < (len + 1) * 3) {
			continue;
		}

#if 0
		int f_bad;
		f_bad = FALSE;
		if (len == 0) {
			// nothing else to test
		}
		else if (len == 1) {
			rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, S[0], 0 /* verbose_level */);
			if (int_vec_search(Lines2, (len + 1) * 3, rk, idx)) {
				f_bad = TRUE;
			}
		}
		else if (len == 2) {
			rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, S[0], 0 /* verbose_level */);
			if (int_vec_search(Lines2, (len + 1) * 3, rk, idx)) {
				f_bad = TRUE;
			}
			rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, S[1], 0 /* verbose_level */);
			if (int_vec_search(Lines2, (len + 1) * 3, rk, idx)) {
				f_bad = TRUE;
			}
			rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[0], S[1], 0 /* verbose_level */);
			if (int_vec_search(Lines2, (len + 1) * 3, rk, idx)) {
				f_bad = TRUE;
			}
		}
#endif

		good_candidates[nb_good_candidates++] = candidates[i];
	} // next i
	if (f_v) {
		cout << "classify_trihedral_pairs::early_test_func_type1 "
				"checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "good_candidates set of size "
				<< nb_good_candidates << ":" << endl;
		lint_vec_print(cout, good_candidates, nb_good_candidates);
		cout << endl;
	}
	
}

void classify_trihedral_pairs::early_test_func_type2(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	long int a, rk;
	long int Lines[9];
	long int Lines2[9];
	int M1[8];
	//int M2[12];
	//int M3[16];
	//int base_cols[4];
	sorting Sorting;

	if (f_v) {
		cout << "classify_trihedral_pairs::early_test_func_type2 "
				"checking set ";
		lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
	}
	if (len > 2) {
		cout << "classify_trihedral_pairs::early_test_func_type2 "
				"len > 2" << endl;
		exit(1);
	}

	for (i = 0; i < len; i++) {
		Lines[i * 3 + 0] = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[i], 0, 0 /* verbose_level */);
		Lines[i * 3 + 1] = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[i], 1, 0 /* verbose_level */);
		Lines[i * 3 + 2] = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(S[i], 2, 0 /* verbose_level */);
	}
	if (f_vv) {
		cout << "classify_trihedral_pairs::early_test_func_type2 "
				"Lines=" << endl;
		lint_matrix_print(Lines, len, 3);
	}

	if (len == 2) {
		Surf->P->unrank_point(M1, S[0]);
		Surf->P->unrank_point(M1 + 4, S[1]);
	}

	nb_good_candidates = 0;
	for (i = 0; i < nb_candidates; i++) {
		a = candidates[i];

		if (a == 0 || a == 1 || a == 2) {
			continue;
		}


		if (f_vv) {
			cout << "classify_trihedral_pairs::early_test_func_type2 "
					"testing a=" << a << endl;
		}

		for (j = 0; j < len; j++) {
			if (a == S[j]) {
				break;
			}
		}
		if (j < len) {
			continue;
		}
		
		lint_vec_copy(Lines, Lines2, len * 3);
		
		rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, 0, 0 /* verbose_level */);
		Lines2[len * 3 + 0] = rk;
		rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, 1, 0 /* verbose_level */);
		Lines2[len * 3 + 1] = rk;
		rk = Surf->P->line_of_intersection_of_two_planes_in_three_space_using_dual_coordinates(a, 2, 0 /* verbose_level */);
		Lines2[len * 3 + 2] = rk;

		if (f_vv) {
			cout << "classify_trihedral_pairs::early_test_func_type1 "
					"Lines2=" << endl;
			lint_matrix_print(Lines2, len + 1, 3);
		}


		Sorting.lint_vec_heapsort(Lines2, (len + 1) * 3);

		for (j = 1; j < (len + 1) * 3; j++) {
			if (Lines2[j] == Lines2[j - 1]) {
				if (f_vv) {
					cout << "classify_trihedral_pairs::early_test_func_type2 "
							"repeated line" << endl;
				}
				break;
			}
		}
		if (j < (len + 1) * 3) {
			continue;
		}

#if 0
		if (len == 2) {
			int_vec_copy(M1, M2, 8);
			Surf->P->unrank_point(M2 + 8, a);
			rk = F->rank_of_rectangular_matrix_memory_given(M2, 3, 4,
					M3, base_cols, 0 /* verbose_level */);
			if (rk < 3) {
				continue;
			}
		}
#endif


		good_candidates[nb_good_candidates++] = candidates[i];
	} // next i
	
}

void classify_trihedral_pairs::identify_three_planes(
	int p1, int p2, int p3,
	int &type, int *transporter, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int M1[16];
	int M2[16];
	int M3[16 + 1]; // if semilinear
	int base_cols[4];
	int base_cols2[4];
	int rk;
	int size_complement;
	int c1, c2, c3, c4, a, b, c, d, e, f, lambda, mu, det, det_inv;
	combinatorics_domain Combi;

	if (f_v) {
		cout << "classify_trihedral_pairs::identify_three_planes" << endl;
	}
	Surf->P->unrank_point(M1, p1);
	Surf->P->unrank_point(M1 + 4, p2);
	Surf->P->unrank_point(M1 + 8, p3);
	int_vec_copy(M1, M2, 12);
	rk = F->rank_of_rectangular_matrix_memory_given(M2, 3, 4,
			M3, base_cols, 0 /* verbose_level */);
	Combi.set_complement(base_cols, rk, base_cols + rk, size_complement, 4);
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_three_planes "
				"rk=" << rk << endl;
	}

	if (rk == 2) {

		c3 = base_cols[rk];
		c4 = base_cols[rk + 1];

		int_vec_copy(M1, M2, 8);
		F->rank_of_rectangular_matrix_memory_given(M2, 2, 4, M3,
				base_cols2, 0 /* verbose_level */);


		c1 = base_cols2[0];
		c2 = base_cols2[1];
		a = M1[c1];
		b = M1[4 + c1];
		c = M1[8 + c1];
		d = M1[c2];
		e = M1[4 + c2];
		f = M1[8 + c2];
		det = F->add(F->mult(a, e), F->negate(F->mult(b, d)));
		det_inv = F->inverse(det);
		lambda = F->mult(F->add(F->mult(e, c),
				F->negate(F->mult(b, f))), det_inv);
		mu = F->mult(F->add(F->mult(a, f),
				F->negate(F->mult(d, c))), det_inv);

		int_vec_copy(M1, M2, 8);
		F->scalar_multiply_vector_in_place(lambda, M2, 4);
		F->scalar_multiply_vector_in_place(mu, M2 + 4, 4);
		int_vec_zero(M2 + 8, 8);
		M2[2 * 4 + c3] = 1;
		M2[3 * 4 + c4] = 1;
		type = 1;
	}
	else if (rk == 3) {
		int_vec_copy(M1, M2, 12);
		int_vec_zero(M2 + 12, 4);
		M2[3 * 4 + base_cols[3]] = 1;
		type = 2;
	}
	else {
		cout << "classify_trihedral_pairs::identify_three_planes "
				"the rank is not 2 or 3" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_three_planes "
				"M2=" << endl;
		int_matrix_print(M2, 4, 4);
	}
	F->matrix_inverse(M2, M3, 4, 0 /* verbose_level */);
	M3[16] = 0; // if semilinear
	A->make_element(transporter, M3, 0 /* verbose_level */);
	
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_three_planes "
				"done" << endl;
	}
}


void classify_trihedral_pairs::classify(
		poset_classification_control *Control1,
		poset_classification_control *Control2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_trihedral_pairs::classify" << endl;
	}

	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"before classify_orbits_on_trihedra" << endl;
	}
	classify_orbits_on_trihedra(Control1, Control2, verbose_level - 1);
	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"before after classify_orbits_on_trihedra" << endl;
	}

	nb_orbits_type1 = orbits_on_trihedra_type1->nb_orbits_at_level(3);
	nb_orbits_type2 = orbits_on_trihedra_type2->nb_orbits_at_level(3);
	nb_orbits_ordered_total = nb_orbits_type1 + nb_orbits_type2;
	if (f_v) {
		cout << "nb_orbits_type1 = " << nb_orbits_type1 << endl;
		cout << "nb_orbits_type2 = " << nb_orbits_type2 << endl;
		cout << "nb_orbits_ordered_total = "
				<< nb_orbits_ordered_total << endl;
	}

	// downstep:
	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"before downstep" << endl;
	}
	downstep(verbose_level - 2);
	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"after downstep" << endl;
	}


	// upstep:
	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"before upstep" << endl;
	}
	upstep(verbose_level - 2);
	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"after upstep" << endl;
	}

	if (f_v) {
		cout << "classify_trihedral_pairs::classify "
				"We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits of trihedral pairs" << endl;
	}


	if (f_v) {
		cout << "classify_trihedral_pairs::classify done" << endl;
	}
}

void classify_trihedral_pairs::downstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "classify_trihedral_pairs::downstep" << endl;
	}
	Flag_orbits = NEW_OBJECT(flag_orbits);
	Flag_orbits->init(A, A, 2 /* nb_primary_orbits_lower */, 
		3 /* pt_representation_sz */,
		nb_orbits_ordered_total /* nb_flag_orbits */,
		verbose_level);

	if (f_v) {
		cout << "classify_trihedral_pairs::downstep "
				"initializing flag orbits type 1" << endl;
	}
	for (i = 0; i < nb_orbits_type1; i++) {
		set_and_stabilizer *R;
		longinteger_object ol;
		longinteger_object go;

		R = orbits_on_trihedra_type1->get_set_and_stabilizer(
				3 /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);

		orbits_on_trihedra_type1->orbit_length(
				i /* node */,
				3 /* level */,
				ol);

		R->Strong_gens->group_order(go);

		Flag_orbits->Flag_orbit_node[i].init(
			Flag_orbits, i /* flag_orbit_index */,
			0 /* downstep_primary_orbit */,
			i /* downstep_secondary_orbit */,
			ol.as_int() /* downstep_orbit_len */,
			FALSE /* f_long_orbit */,
			R->data /* int *pt_representation */,
			R->Strong_gens,
			verbose_level - 2);
		R->Strong_gens = NULL;
		FREE_OBJECT(R);
		if (f_v) {
			cout << "flag orbit " << i << " / "
					<< nb_orbits_ordered_total << " is type 1 orbit "
					<< i << " / " << nb_orbits_type1
					<< " stab order " << go << endl;
		}
	}
	if (f_v) {
		cout << "classify_trihedral_pairs::downstep "
				"initializing flag orbits type 2" << endl;
	}
	for (i = 0; i < nb_orbits_type2; i++) {
		set_and_stabilizer *R;
		longinteger_object ol;
		longinteger_object go;

		R = orbits_on_trihedra_type2->get_set_and_stabilizer(
				3 /* level */, i /* orbit_at_level */,
				0 /* verbose_level */);

		orbits_on_trihedra_type2->orbit_length(
				i /* node */, 3 /* level */, ol);

		R->Strong_gens->group_order(go);

		Flag_orbits->Flag_orbit_node[nb_orbits_type1 + i].init(
			Flag_orbits,
			nb_orbits_type1 + i /* flag_orbit_index */,
			1 /* downstep_primary_orbit */,
			i /* downstep_secondary_orbit */,
			ol.as_int() /* downstep_orbit_len */,
			FALSE /* f_long_orbit */,
			R->data /* int *pt_representation */,
			R->Strong_gens,
			verbose_level - 2);
		R->Strong_gens = NULL;
		FREE_OBJECT(R);
		if (f_v) {
			cout << "flag orbit " << nb_orbits_type1 +  i
					<< " / " << nb_orbits_ordered_total
					<< " is type 2 orbit " << i << " / "
					<< nb_orbits_type2 << " stab order " << go << endl;
		}
	}
	if (f_v) {
		cout << "classify_trihedral_pairs::downstep "
				"initializing flag orbits done" << endl;
	}


	if (f_v) {
		cout << "classify_trihedral_pairs::downstep done" << endl;
	}
}

void classify_trihedral_pairs::upstep(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "classify_trihedral_pairs::upstep" << endl;
	}

	int *f_processed;
	int nb_processed, po, so, type, orb, f, f2;
	long int Planes[] = {0,1,5, 0,1,2};
	long int planes1[3];
	long int planes2[3];
	long int planes3[3];
	long int planes4[3];
	int *Elt1;
	int *Elt2;
	int *Elt3;

	f_processed = NEW_int(nb_orbits_ordered_total);
	int_vec_zero(f_processed, nb_orbits_ordered_total);
	nb_processed = 0;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	
	Trihedral_pairs = NEW_OBJECT(classification_step);

	longinteger_object go;
	A->group_order(go);

	Trihedral_pairs->init(A, A, nb_orbits_ordered_total,
			6, go, verbose_level);


	for (f = 0; f < nb_orbits_ordered_total; f++) {

		double progress;
		
		if (f_processed[f]) {
			continue;
		}

		progress = ((double)nb_processed * 100. ) /
				(double) nb_orbits_ordered_total;

		if (f_v) {
			cout << "Defining n e w orbit "
					<< Flag_orbits->nb_primary_orbits_upper
					<< " from flag orbit " << f << " / "
					<< nb_orbits_ordered_total
					<< " progress=" << progress << "%" << endl;
		}
		Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit =
				Flag_orbits->nb_primary_orbits_upper;
		

		
		po = Flag_orbits->Flag_orbit_node[f].downstep_primary_orbit;
		so = Flag_orbits->Flag_orbit_node[f].downstep_secondary_orbit;
		lint_vec_copy(Flag_orbits->Pt + f * 3, planes1, 3);
		if (f_v) {
			cout << "classify_trihedral_pairs::upstep initializing planes: ";
			lint_vec_print(cout, planes1, 3);
			cout << endl;
		}
		identify_three_planes(planes1[0], planes1[1], planes1[2],
				type, Elt1 /* int *transporter */, 0 /*verbose_level*/);

		if (f_v) {
			cout << "We found a transporter:" << endl;
			A->element_print_quick(Elt1, cout);
		}

		lint_vec_copy(Planes + po * 3, planes2, 3);
		A->map_a_set_and_reorder(planes2, planes3, 3,
				Elt1, 0 /* verbose_level */);
		if (type == 1) {
			orb = orbits_on_trihedra_type1->trace_set(
				planes3, 3 /* size */, 3 /* level */,
				planes4 /* int *canonical_set */, Elt2, 
				0 /* verbose_level */);
		}
		else if (type == 2) {
			orb = orbits_on_trihedra_type2->trace_set(
				planes3, 3 /* size */, 3 /* level */,
				planes4 /* int *canonical_set */, Elt2, 
				0 /* verbose_level */);
		}
		else {
			cout << "type must be either 1 or 2." << endl;
			exit(1);
		}
		A->element_mult(Elt1, Elt2, Elt3, 0);


		strong_generators *S;
		longinteger_object go;

		S = Flag_orbits->Flag_orbit_node[f].gens->create_copy();
		
		if (type - 1 == po && orb == so) {
			if (f_v) {
				cout << "We found an automorphism "
						"of the trihedral pair:" << endl;
				A->element_print_quick(Elt3, cout);
				cout << endl;
			}
		
			S->add_single_generator(Elt3,
					2 /* group_index */, verbose_level - 2);
		}
		else {
			if (f_v) {
				cout << "We are identifying with po="
						<< type - 1 << " so=" << orb << endl;
			}
			if (type == 1) {
				f2 = orb;
			}
			else {
				f2 = nb_orbits_type1 + orb;
			}
			if (f_v) {
				cout << "We are identifying with po=" << type - 1
						<< " so=" << orb << ", which is "
								"flag orbit " << f2 << endl;
			}
			Flag_orbits->Flag_orbit_node[f2].f_fusion_node = TRUE;
			Flag_orbits->Flag_orbit_node[f2].fusion_with = f;
			Flag_orbits->Flag_orbit_node[f2].fusion_elt =
					NEW_int(A->elt_size_in_int);
			A->element_invert(Elt3,
					Flag_orbits->Flag_orbit_node[f2].fusion_elt, 0);
			f_processed[f2] = TRUE;
			nb_processed++;
		}
		S->group_order(go);
		if (f_v) {
			cout << "the trihedral pair has a stabilizer of order "
					<< go << endl;
		}

		long int Rep[6];

		lint_vec_copy(Planes + po * 3, Rep, 3);
		lint_vec_copy(planes1, Rep + 3, 3);
		Trihedral_pairs->Orbit[Flag_orbits->nb_primary_orbits_upper].init(
			Trihedral_pairs,
			Flag_orbits->nb_primary_orbits_upper, 
			S, Rep, verbose_level);

		
		f_processed[f] = TRUE;
		nb_processed++;
		Flag_orbits->nb_primary_orbits_upper++;
	}
	if (nb_processed != nb_orbits_ordered_total) {
		cout << "nb_processed != nb_orbits_ordered_total" << endl;
		cout << "nb_processed = " << nb_processed << endl;
		cout << "nb_orbits_ordered_total = "
				<< nb_orbits_ordered_total << endl;
		exit(1);
	}

	Trihedral_pairs->nb_orbits = Flag_orbits->nb_primary_orbits_upper;
	
	if (f_v) {
		cout << "We found " << Flag_orbits->nb_primary_orbits_upper
				<< " orbits of trihedral pairs" << endl;
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(f_processed);

	if (f_v) {
		cout << "classify_trihedral_pairs::upstep done" << endl;
	}
}

void classify_trihedral_pairs::print_trihedral_pairs(ostream &ost, 
	int f_with_stabilizers)
{
	Trihedral_pairs->print_latex(ost, 
		"Classification of Double Triplets", f_with_stabilizers,
		FALSE, NULL, NULL);
}


strong_generators
*classify_trihedral_pairs::identify_trihedral_pair_and_get_stabilizer(
	long int *planes6, int *transporter, int &orbit_index,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair_and_get_stabilizer" << endl;
	}

	if (f_v) {
		cout << "planes6: ";
		lint_vec_print(cout, planes6, 6);
		cout << endl;
	}


	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair_"
				"and_get_stabilizer before identify_trihedral_pair" << endl;
	}
	identify_trihedral_pair(planes6, 
		transporter, orbit_index, verbose_level);
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair_"
				"and_get_stabilizer after identify_trihedral_pair" << endl;
	}
	if (f_v) {
		cout << "orbit_index=" << orbit_index << endl;
	}

	strong_generators *gens;
	gens = NEW_OBJECT(strong_generators);

	
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair_"
				"and_get_stabilizer before gens->init_generators_for_"
				"the_conjugate_group_aGav" << endl;
	}
	gens->init_generators_for_the_conjugate_group_aGav(
			Trihedral_pairs->Orbit[orbit_index].gens,
		transporter, verbose_level);
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair_"
				"and_get_stabilizer after gens->init_generators_for_"
				"the_conjugate_group_aGav" << endl;
	}


	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair_and_get_stabilizer done" << endl;
	}

	return gens;
}



void classify_trihedral_pairs::identify_trihedral_pair(long int *planes6,
	int *transporter, int &orbit_index, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int type, orb, f, f2;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	int *Elt4;
	long int planes1[3];
	long int planes2[3];

	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair" << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair "
				"identifying the first trihedron" << endl;
	}
	identify_three_planes(planes6[0], planes6[1], planes6[2],
			type, Elt1 /* int *transporter */, 0 /*verbose_level*/);

	if (f_vv) {
		cout << "Elt1=" << endl;
		A->element_print_quick(Elt1, cout);
	}

	A->map_a_set_and_reorder(
			planes6 + 3, planes1, 3, Elt1,
			0 /* verbose_level */);
	if (type == 1) {
		orb = orbits_on_trihedra_type1->trace_set(
				planes1, 3 /* size */, 3 /* level */,
			planes2 /* int *canonical_set */, Elt2, 
			0 /* verbose_level */);
	}
	else if (type == 2) {
		orb = orbits_on_trihedra_type2->trace_set(
				planes1, 3 /* size */, 3 /* level */,
			planes2 /* int *canonical_set */, Elt2, 
			0 /* verbose_level */);
	}
	else {
		cout << "type must be either 1 or 2." << endl;
		exit(1);
	}
	A->element_mult(Elt1, Elt2, Elt3, 0);
	
	if (type == 1) {
		f = orb;
	}
	else {
		f = nb_orbits_type1 + orb;
	}
	if (Flag_orbits->Flag_orbit_node[f].f_fusion_node) {
		A->element_mult(Elt3,
				Flag_orbits->Flag_orbit_node[f].fusion_elt, Elt4, 0);
		f2 = Flag_orbits->Flag_orbit_node[f].fusion_with;
		orbit_index = Flag_orbits->Flag_orbit_node[f2].upstep_primary_orbit;
	}
	else {
		f2 = -1;
		A->element_move(Elt3, Elt4, 0);
		orbit_index = Flag_orbits->Flag_orbit_node[f].upstep_primary_orbit;
	}
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair "
				"type=" << type << " orb=" << orb << " f=" << f
				<< " f2=" << f2 << " orbit_index=" << orbit_index << endl;
	}
	A->element_move(Elt4, transporter, 0);
	if (f_vv) {
		cout << "transporter=" << endl;
		A->element_print_quick(transporter, cout);
	}
	
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	FREE_int(Elt4);
	if (f_v) {
		cout << "classify_trihedral_pairs::identify_trihedral_pair "
				"done" << endl;
	}
}

// #############################################################################
// global functions:
// #############################################################################

void classify_trihedral_pairs_early_test_function_type1(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	classify_trihedral_pairs *CT = (classify_trihedral_pairs *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "classify_trihedral_pairs_early_test_function_type1 "
				"for set ";
		print_set(cout, len, S);
		cout << endl;
	}
	CT->early_test_func_type1(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "classify_trihedral_pairs_early_test_function_type1 "
				"done" << endl;
	}
}

void classify_trihedral_pairs_early_test_function_type2(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{
	classify_trihedral_pairs *CT = (classify_trihedral_pairs *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "classify_trihedral_pairs_early_test_function_type2 "
				"for set ";
		print_set(cout, len, S);
		cout << endl;
	}
	CT->early_test_func_type2(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "classify_trihedral_pairs_early_test_function_type2 "
				"done" << endl;
	}
}

}}

