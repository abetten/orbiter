/*
 * surface_study.cpp
 *
 *  Created on: Nov 6, 2019
 *      Author: anton
 *
 *  originally created on  September 12, 2016
 */




#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace top_level {


void surface_study::init(finite_field *F, int nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_study::init" << endl;
	}
	q = F->q;
	cout << "q=" << q << endl;
	cout << "nb=" << nb << endl;


	int i;
	number_theory_domain NT;
	knowledge_base K;


	char str_q[1000];
	char str_nb[1000];

	sprintf(str_q, "%d", q);
	sprintf(str_nb, "%d", nb);

	prefix.assign("surface_q");
	prefix.append(str_q);
	prefix.append("_nb");
	prefix.append(str_nb);




	Surf = NEW_OBJECT(surface_domain);

	if (f_v) {
		cout << "surface_study::init initializing surface" << endl;
		}
	Surf->init(F, verbose_level);
	if (f_v) {
		cout << "surface_study::init initializing surface done" << endl;
		}

	nb_lines_PG_3 = Surf->Gr->nCkq->as_int();
	cout << "surface_study::init nb_lines_PG_3 = " << nb_lines_PG_3 << endl;



	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}

	cout << "creating linear group" << endl;
	vector_ge *nice_gens;

	A = NEW_OBJECT(action);

	A->init_linear_group(
		F, 4,
		TRUE /*f_projective*/,
		FALSE /* f_general*/,
		FALSE /* f_affine */,
		f_semilinear, FALSE /* f_special */,
		nice_gens,
		0 /*verbose_level*/);

	S = A->Strong_gens->create_sims(verbose_level - 2);
	FREE_OBJECT(nice_gens);
	cout << "creating linear group done" << endl;

	cout << "creating action on lines" << endl;
	A2 = A->induced_action_on_grassmannian(2, verbose_level);
	cout << "creating action on lines done" << endl;

	coeff = NEW_int(20);

	//nb_reps = cubic_surface_nb_reps(q);
	rep = K.cubic_surface_representative(q, nb);
		// rep is the vector of 20 coefficients

	//six = cubic_surface_single_six(q, nb);
	K.cubic_surface_stab_gens(q, nb,
			data, nb_gens, data_size, stab_order);

	Lines = K.cubic_surface_Lines(q, nb);

	cout << "The lines are: ";
	Orbiter->Lint_vec->print(cout, Lines, 27);
	cout << endl;


	cout << "q=" << q << " nb=" << nb; // << " six=";
	//int_vec_print(cout, six, 6);
	cout << " coeff=";
	Orbiter->Int_vec->print(cout, rep, 20);
	cout << endl;

	cout << "stab_gens for a group of order " << stab_order << ":" << endl;
	for (i = 0; i < nb_gens; i++) {
		Orbiter->Int_vec->print(cout, data + i * data_size, data_size);
		cout << endl;
		}

	SaS = NEW_OBJECT(set_and_stabilizer);


	SaS->init(A, A2, verbose_level);

	SaS->init_data(Lines, 27, verbose_level);

	SaS->init_stab_from_data(data,
		data_size, nb_gens,
		stab_order,
		verbose_level);


	if (q == 11) {


		cout << "q=11:" << endl;


		//int *Elt;
		int data1[] = {3,6,7,4,9,7,4,7,8,2,4,8,7,5,4,6};
		int data2[] = {5,0,0,0,10,10,10,0,0,10,0,0,2,4,3,4};
		int data3[] = { 3,  0,  0,  0,  0,  9,  0,  0,  0,  0,  3,  0,  0,  0,  0,  7};
		int data4[] = { 7, 0, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 3, 7, 2, 3};


		cout << "applying first transformation:" << endl;
		SaS->apply_to_self_element_raw(data1, verbose_level);

		cout << "applying second transformation:" << endl;
		SaS->apply_to_self_inverse_element_raw(data2, verbose_level);

		cout << "applying third transformation:" << endl;
		SaS->apply_to_self_element_raw(data3, verbose_level);

		cout << "applying fourth transformation:" << endl;
		SaS->apply_to_self_element_raw(data4, verbose_level);


		//equation: W^3 + 10X^2W + 10Y^2W + 10Z^2W + 7XYZ
		// a=4, b=1, \alpha=10, \beta=7

		}

	if (q == 13 && nb == 0) {

		cout << "q=13, nb=0:" << endl;

		//int *Elt;
		int data1[] = { 6,4,2,10,1,7,9,6,8,6,9,10,8,1,8,6};
		int data2[] = { 1,0,0,0,0,1,0,0,0,0,4,0,2,6,7,3};
		int data3[] = { 12,0,0,0,0,12,0,0,0,0,12,0,5,0,10,8};


		cout << "applying first transformation:" << endl;
		SaS->apply_to_self_element_raw(data1, verbose_level);

		cout << "applying second transformation:" << endl;
		SaS->apply_to_self_element_raw(data2, verbose_level);

		cout << "applying third transformation:" << endl;
		SaS->apply_to_self_element_raw(data3, verbose_level);

		}

	if (q == 13 && nb == 1) {

		cout << "q=13, nb=1:" << endl;

		//int *Elt;
		int data1[] = {2, 9, 3, 7, 10, 8, 12, 4, 10, 1, 11, 10, 3, 10, 12, 5 };
		int data2[] = {6, 0, 0, 0, 0, 6, 0, 0, 8, 8, 8, 0, 9, 6, 8, 8};
		int data3[] = { 6,  0,  0,  0,  0,  3,  0,  0,  0,  0, 12,  0,  0,  0,  0, 11};
		int data4[] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 12, 0, 11, 12, 9, 6};

#if 1
		cout << "applying first transformation:" << endl;
		SaS->apply_to_self_element_raw(data1, verbose_level);

		cout << "applying second transformation:" << endl;
		SaS->apply_to_self_inverse_element_raw(data2, verbose_level);

		cout << "applying third transformation:" << endl;
		SaS->apply_to_self_element_raw(data3, verbose_level);

		cout << "applying fourth transformation:" << endl;
		SaS->apply_to_self_element_raw(data4, verbose_level);

		//equation: W^3 + 12X^2W + 12Y^2W + 12Z^2W + XYZ
#endif
		}


	if (q == 17 && nb == 0) {

		cout << "q=17, nb=0:" << endl;

		//int *Elt;
		int data1[] = {9, 3, 13, 5, 11, 1, 1, 0, 11, 1, 6, 15, 16, 14, 5, 4};
		int data2[] = {16, 0, 0, 0, 0, 15, 0, 0, 14, 14, 14, 0, 8, 7, 9, 8};
		int data3[] = { 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 8, 0, 0, 4, 0, 1};
		int data4[] = { 0, 6, 0, 0, 6, 0, 0, 0, 0, 0, 11, 0, 1, 4, 0, 6};


		cout << "applying first transformation:" << endl;
		SaS->apply_to_self_element_raw(data1, verbose_level);

		cout << "applying second transformation:" << endl;
		SaS->apply_to_self_inverse_element_raw(data2, verbose_level);

		cout << "applying third transformation:" << endl;
		SaS->apply_to_self_element_raw(data3, verbose_level);

		cout << "applying fourth transformation:" << endl;
		SaS->apply_to_self_element_raw(data4, verbose_level);

		//equation: W^3 + 16X^2W + 16Y^2W + 16Z^2W + 5XYZ

		}
	if (q == 17 && nb == 2) {

		cout << "q=17, nb=2:" << endl;

		//int *Elt;
		int data1[] = {14, 11, 14, 15, 7, 12, 8, 14, 9, 1, 11, 16, 13, 13, 1, 6};
		int data2[] = {16, 0, 0, 0, 0, 15, 0, 0, 14, 14, 14, 0, 8, 7, 9, 8 };
		int data3[] = { 6,  0,  0,  0,  0, 15,  0,  0,  0,  0,  7,  0,  0,  0,  0,  8};
		int data4[] = { 16, 0, 0, 0, 0, 1, 0, 0, 0, 0, 16, 0, 12, 6, 14, 15};


		cout << "applying first transformation:" << endl;
		SaS->apply_to_self_element_raw(data1, verbose_level);

		cout << "applying second transformation:" << endl;
		SaS->apply_to_self_inverse_element_raw(data2, verbose_level);

		cout << "applying third transformation:" << endl;
		SaS->apply_to_self_element_raw(data3, verbose_level);

		cout << "applying fourth transformation:" << endl;
		SaS->apply_to_self_element_raw(data4, verbose_level);

		//equation: W^3 + 8X^2W + 8Y^2W + 8Z^2W + 12XYZ

		}
	if (q == 17 && nb == 6) {

		cout << "q=17, nb=6:" << endl;

		int data1[] = {16, 10, 13, 9, 4, 13, 2, 1, 2, 3, 12, 7, 4, 6, 3, 9};
		int data2[] = {16, 0, 0, 0, 0, 15, 0, 0, 14, 14, 14, 0, 8, 7, 9, 8};
		int data3[] = { 7,  0,  0,  0,  0,  6,  0,  0,  0,  0,  7,  0,  0,  0,  0,  9};
		int data4[] = { 0, 0, 16, 0, 16, 0, 0, 0, 0, 1, 0, 0, 7, 8, 1, 14};

		cout << "applying first transformation:" << endl;
		SaS->apply_to_self_element_raw(data1, verbose_level);

		cout << "applying second transformation:" << endl;
		SaS->apply_to_self_inverse_element_raw(data2, verbose_level);

		cout << "applying third transformation:" << endl;
		SaS->apply_to_self_element_raw(data3, verbose_level);


		cout << "applying fourth transformation:" << endl;
		SaS->apply_to_self_element_raw(data4, verbose_level);


		//equation: W^3 + 16X^2W + 16Y^2W + 16Z^2W + 11XYZ

		}




	Surf->build_cubic_surface_from_lines(
			SaS->sz, SaS->data, coeff,
			0 /* verbose_level */);
	F->PG_element_normalize_from_front(coeff, 1, 20);
	cout << "coefficient vector of the surface: ";
	Orbiter->Int_vec->print(cout, coeff, 20);
	cout << endl;
	cout << "equation: ";
	Surf->print_equation(cout, coeff);
	cout << endl;


	Surf->rearrange_lines_according_to_double_six(
			SaS->data /* Lines */,
			0 /* verbose_level */);
}


void surface_study::study_intersection_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "surface_study::study_intersection_points" << endl;
		}


	int *Adj;
	//int *R;
	long int *Intersection_pt;
	int line_labels[27];
	int fst[1];
	int len[1];
	int i;
	fst[0] = 0;
	len[0] = 27;

	for (i = 0; i < 27; i++) {
		line_labels[i] = i;
		}

	Surf->compute_adjacency_matrix_of_line_intersection_graph(
			Adj, SaS->data, SaS->sz, verbose_level);
	cout << "The adjacency matrix is:" << endl;
	Orbiter->Int_vec->matrix_print(Adj, SaS->sz, SaS->sz);

	Surf->compute_intersection_points(Adj,
		SaS->data, SaS->sz, Intersection_pt,
		verbose_level);

	cout << "The intersection points are:" << endl;
	Orbiter->Lint_vec->matrix_print(Intersection_pt, SaS->sz, SaS->sz);


	string fname_intersection_pts;
	string fname_intersection_pts_tex;

	fname_intersection_pts.assign(prefix);
	fname_intersection_pts.append("_intersection_points0.csv");

	fname_intersection_pts_tex.assign(prefix);
	fname_intersection_pts_tex.append("_intersection_points0.tex");


	Fio.lint_matrix_write_csv(fname_intersection_pts,
			Intersection_pt, SaS->sz, SaS->sz);
	cout << "Written file " << fname_intersection_pts
			<< " of size " << Fio.file_size(fname_intersection_pts) << endl;

	{
	ofstream fp(fname_intersection_pts_tex);
	latex_interface L;

	L.head_easy(fp);
	//latex_head_easy_sideways(fp);
	fp << "{\\tiny \\arraycolsep=1pt" << endl;
	fp << "$$" << endl;
	L.lint_matrix_print_with_labels_and_partition(fp,
		Intersection_pt, SaS->sz, SaS->sz,
		line_labels, line_labels,
		fst, len, 1,
		fst, len, 1,
		matrix_entry_print, (void *) Surf,
		TRUE /* f_tex */);
	fp << "$$}" << endl;
	L.foot(fp);
	}
	cout << "Written file " << fname_intersection_pts_tex
			<< " of size " << Fio.file_size(fname_intersection_pts_tex) << endl;
	FREE_int(Adj);
	//FREE_int(R);
	FREE_lint(Intersection_pt);

	if (f_v) {
		cout << "surface_study::study_intersection_points done" << endl;
		}
}







#if 0
	int *Pts;
	int nb_pts;
	int a, b;

	nb_pts = SaS->sz;
	Pts = NEW_int(nb_pts);
	for (i = 0; i < nb_pts; i++) {
		a = SaS->data[i];
		b = Klein->Line_to_point_on_quadric[a];
		Pts[i] = b;
		}
	//compute_decomposition(O, F, Pts, nb_pts, verbose_level);
	FREE_int(Pts);
#endif



void surface_study::study_line_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_study::study_line_orbits" << endl;
		}


	cout << "rearranging by orbits:" << endl;
	SaS->rearrange_by_orbits(orbit_first, orbit_length, orbit,
		nb_orbits, 0 /* verbose_level*/);


	cout << "after rearranging, the set is:" << endl;
	Orbiter->Lint_vec->print(cout, SaS->data, SaS->sz);
	cout << endl;

	cout << "orbit_length: ";
	Orbiter->Int_vec->print(cout, orbit_length, nb_orbits);
	cout << endl;



	if (f_v) {
		cout << "surface_study::study_line_orbits" << endl;
		}
}



void surface_study::study_group(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	file_io Fio;

	if (f_v) {
		cout << "surface_study::study_group" << endl;
		}
	cout << "The elements are :" << endl;
	//SaS->Stab->print_all_group_elements();

	int *Group_elts;
	int group_order;
	int elt_sz;
	longinteger_object go;
	int i;

	SaS->Stab->table_of_group_elements_in_data_form(
			Group_elts, group_order, elt_sz,
			verbose_level);
	SaS->Stab->group_order(go);

	cout << "Group_elts:" << endl;
	Orbiter->Int_vec->matrix_print(Group_elts, group_order, elt_sz);
	for (i = 0; i < group_order; i++) {
		F->PG_element_normalize(Group_elts + i * elt_sz, 1, elt_sz);
		}
	cout << "group elements:" << endl;
	for (i = 0; i < group_order; i++) {
		A->print_for_make_element(cout,
				Group_elts + i * elt_sz);
		cout << endl;
		}
	cout << "Group_elts normalized from the back:" << endl;
	for (i = 0; i < group_order; i++) {
		F->PG_element_normalize(Group_elts + i * elt_sz, 1, elt_sz);
		cout << "element " << i << " / " << group_order << ":" << endl;
		Orbiter->Int_vec->matrix_print(Group_elts + i * elt_sz, 4, 4);
		}
	//int_matrix_print(Group_elts, group_order, elt_sz);


	if (group_order < 1000) {
		int *Table;
		long int n;

		cout << "creating the group table:..." << endl;
		SaS->Stab->create_group_table(Table, n, 0 /* verbose_level */);
		//cout << "The group table is:" << endl;
		//int_matrix_print(Table, n, n);

		string fname_out;
		char str[1000];

		fname_out.assign(prefix);
		sprintf(str, "_table_%d_%d.csv", q, nb);
		fname_out.append(str);

		Fio.int_matrix_write_csv(fname_out, Table, n, n);
		cout << "Written file " << fname_out
				<< " of size " << Fio.file_size(fname_out) << endl;
		SaS->Stab->write_as_magma_permutation_group(
				prefix, SaS->Strong_gens->gens, verbose_level);
		}
	else {
		cout << "We won't create the group table because "
				"the group order is too big" << endl;
		}



	FREE_int(Group_elts);
	if (f_v) {
		cout << "surface_study::study_group done" << endl;
		}
}


#if 0
	int *Mtx3;
	action *A3;
	int h;
	sims *PGL3;
	vector_ge *gens;
	Mtx3 = NEW_int(9);
	strong_generators *Strong_gens;

	cout << "creating linear group A3" << endl;
	create_linear_group(PGL3, A3,
		F, 3,
		TRUE /*f_projective*/, FALSE /* f_general*/, FALSE /* f_affine */,
		FALSE /* f_semilinear */, FALSE /* f_special */,
		0 /*verbose_level*/);
	cout << "creating linear group A3 done" << endl;
	gens = NEW_OBJECT(vector_ge);
	gens->init(A3);
	gens->allocate(group_order);
	for (h = 0; h < group_order; h++) {
		for (i = 0; i < 3; i++) {
			for (j = 0; j < 3; j++) {
				a = Group_elts[h * elt_sz + i * 4 + j];
				Mtx3[i * 3 + j] = a;
				}
			}
		A3->make_element(gens->ith(h), Mtx3, 0 /* verbose_level */);
		}
	generators_to_strong_generators(A3,
		TRUE /* f_target_go */, go,
		gens, Strong_gens,
		verbose_level);
	cout << "Strong generators:" << endl;
	Strong_gens->print_generators();
	sims *S3;

	S3 = Strong_gens->create_sims(verbose_level);
	S3->print_all_group_elements();
	FREE_int(Group_elts);
	S3->table_of_group_elements_in_data_form(
			Group_elts, group_order, elt_sz, verbose_level);
	cout << "Group_elts:" << endl;
	int_matrix_print(Group_elts, group_order, elt_sz);
	for (i = 0; i < group_order; i++) {
		PG_element_normalize(*F, Group_elts + i * elt_sz, 1, elt_sz);
		}
	cout << "Group_elts normalized from the back:" << endl;
	int_matrix_print(Group_elts, group_order, elt_sz);

	{
	int *Table;
	int n;

	S3->create_group_table(Table, n, 0 /* verbose_level */);
	cout << "The group table is:" << endl;
	int_matrix_print(Table, n, n);
	int_matrix_print_tex(cout, Table, n, n);
	FREE_int(Table);
	}
#endif

	//exit(1);




void surface_study::study_orbits_on_lines(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_study::study_orbits_on_lines" << endl;
		}


	cout << "creating restricted action on the set of lines" << endl;
	A_on_lines = A2->restricted_action(
			SaS->data, 27, verbose_level);
	cout << "creating restricted action on the set of lines done" << endl;


	cout << "computing orbits on lines:" << endl;
	Orb = SaS->Strong_gens->orbits_on_points_schreier(
			A_on_lines, verbose_level);

	cout << "orbits on lines:" << endl;
	Orb->print_and_list_orbits(cout);

	int f, l, b, ii, i, a;
	int Basis[8];

	if (Orb->find_shortest_orbit_if_unique(shortest_line_orbit_idx)) {
		f = Orb->orbit_first[shortest_line_orbit_idx];
		l = Orb->orbit_len[shortest_line_orbit_idx];
		cout << "The unique shortest orbit has length " << l << ":" << endl;
		for (i = 0; i < l; i++) {
			a = Orb->orbit[f + i];
			b = SaS->data[a];
			cout << i << " : " << a << " : " << b << endl;

			Surf->Gr->unrank_lint_here(Basis, b, 0);
			Orbiter->Int_vec->matrix_print(Basis, 2, 4);
			}

		for (ii = 0; ii < Orb->nb_orbits; ii++) {
			f = Orb->orbit_first[ii];
			l = Orb->orbit_len[ii];
			cout << "Orbit " << ii << " has length " << l << ":" << endl;
			for (i = 0; i < l; i++) {
				a = Orb->orbit[f + i];
				b = SaS->data[a];
				cout << i << " : " << a << " : " << b << endl;

				Surf->Gr->unrank_lint_here(Basis, b, 0);
				Orbiter->Int_vec->matrix_print(Basis, 2, 4);
				}
			}
		}
	cout << "The unique shortest orbit on lines is orbit "
			<< shortest_line_orbit_idx << endl;
	if (f_v) {
		cout << "surface_study::study_orbits_on_lines" << endl;
		}
}

void surface_study::study_find_eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::sorting Sorting;
	graph_theory_domain Graph;
	file_io Fio;

	if (f_v) {
		cout << "surface_study::study_find_eckardt_points" << endl;
		}



	Surf->compute_adjacency_matrix_of_line_intersection_graph(
			Adj, SaS->data, SaS->sz, verbose_level);
	cout << "The adjacency matrix is:" << endl;
	Orbiter->Int_vec->matrix_print(Adj, SaS->sz, SaS->sz);



	Graph.compute_decomposition_of_graph_wrt_partition(Adj, SaS->sz,
		orbit_first, orbit_length, nb_orbits, R, verbose_level);


	cout << "The tactical decomposition scheme is:" << endl;
	Orbiter->Int_vec->matrix_print(R, nb_orbits, nb_orbits);

	Surf->compute_intersection_points(Adj,
		SaS->data, SaS->sz, Intersection_pt,
		verbose_level);

	cout << "The intersection points are:" << endl;
	Orbiter->Lint_vec->matrix_print(Intersection_pt, SaS->sz, SaS->sz);

	{
		string fname_intersection_pts;
		string fname_intersection_pts_tex;


		fname_intersection_pts.append(prefix);
		fname_intersection_pts.append("_intersection_points.csv");

		fname_intersection_pts_tex.append(prefix);
		fname_intersection_pts_tex.append("_intersection_points.tex");

		Fio.lint_matrix_write_csv(fname_intersection_pts,
				Intersection_pt, SaS->sz, SaS->sz);
		cout << "Written file " << fname_intersection_pts
				<< " of size " << Fio.file_size(fname_intersection_pts)
				<< endl;

		{
		ofstream fp(fname_intersection_pts_tex);
		latex_interface L;

		L.head_easy(fp);
		//latex_head_easy_sideways(fp);
		fp << "{\\tiny \\arraycolsep=1pt" << endl;
		fp << "$$" << endl;
		L.lint_matrix_print_with_labels_and_partition(fp,
			Intersection_pt, SaS->sz, SaS->sz,
			orbit, orbit,
			Orb->orbit_first, Orb->orbit_len, Orb->nb_orbits,
			Orb->orbit_first, Orb->orbit_len, Orb->nb_orbits,
			matrix_entry_print, (void *) Surf,
			TRUE /* f_tex */);
		fp << "$$}" << endl;
		L.foot(fp);
		}
		cout << "Written file " << fname_intersection_pts_tex
				<< " of size " << Fio.file_size(fname_intersection_pts_tex)
				<< endl;
	}


	tally C;

	C.init_lint(Intersection_pt, SaS->sz * SaS->sz, FALSE, 0);
	cout << "classification of points by multiplicity:" << endl;
	C.print_naked(TRUE);
	cout << endl;




	C.get_data_by_multiplicity_as_lint(
			Double_pts, nb_double_pts,
			2,
			0 /* verbose_level */);

	Sorting.lint_vec_heapsort(Double_pts, nb_double_pts);

	cout << "We found " << nb_double_pts << " double points" << endl;
	Orbiter->Lint_vec->print(cout, Double_pts, nb_double_pts);
	cout << endl;





	C.get_data_by_multiplicity_as_lint(
			Eckardt_pts, nb_Eckardt_pts,
			6,
			0 /* verbose_level */);
		// Eckardt points appear 6 = {3 \choose 2} times in the table

	Sorting.lint_vec_heapsort(Eckardt_pts, nb_Eckardt_pts);

	cout << "We found " << nb_Eckardt_pts << " Eckardt points" << endl;
	Orbiter->Lint_vec->print(cout, Eckardt_pts, nb_Eckardt_pts);
	cout << endl;

	if (f_v) {
		cout << "surface_study::study_find_eckardt_points done" << endl;
		}
}




void surface_study::study_surface_with_6_eckardt_points(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "surface_study::study_surface_with_6_eckardt_points" << endl;
		}

	int i, j;

	cout << "nb_Eckardt_pts = 6" << endl;

	int Basis1[] = {1,0,0,0,0,1,0,0}; // Z=0
	int Basis2[] = {1,0,0,0,0,0,1,0}; // Y=0
	int Basis3[] = {0,1,0,0,0,0,1,0}; // X=0
	long int desired_lines[3];

	desired_lines[0] = Surf->Gr->rank_lint_here(Basis1, 0 /*verbose_level*/);
	desired_lines[1] = Surf->Gr->rank_lint_here(Basis2, 0 /*verbose_level*/);
	desired_lines[2] = Surf->Gr->rank_lint_here(Basis3, 0 /*verbose_level*/);

	cout << "desired_lines : ";
	Orbiter->Lint_vec->print(cout, desired_lines, 3);
	cout << endl;


	//display_all_PG_elements(3, *F);

#if 0
	action *Ar;
	int special_lines[] = {0,1,2,4, 3,5,6,7, 10,11,8,9,  14,16,18,20, 12,13,22,23, 15,17,19,21,  24,25,26};
	int nb_special_lines = 27;

	Ar = A_on_lines->restricted_action(special_lines,
			nb_special_lines, 0 /* verbose_level */);
	cout << "The elements are :" << endl;
	SaS->Stab->print_all_group_elements_as_permutations_in_special_action(Ar);

#endif



	long int *triangle;
	int nb_pts_triangle;
	long int three_points[3] = {0,1,2};
	cout << "creating projective triangle" << endl;
	Surf->P->points_on_projective_triangle(
			triangle, nb_pts_triangle, three_points,
			verbose_level);




	Orbiter->Lint_vec->take_away(Double_pts, nb_double_pts, triangle, nb_pts_triangle);
	Orbiter->Lint_vec->take_away(Double_pts, nb_double_pts, three_points, 3);

	cout << "After taking away the triangle points, "
			"we still have " << nb_double_pts << " double points" << endl;
	Orbiter->Lint_vec->print_as_table(cout, Double_pts, nb_double_pts, 10);
	cout << endl;


	set_and_stabilizer *Triangle;
	set_and_stabilizer *Eckardt_stab;
	action *A_triangle;
	action *A_on_double_pts;
	int *Elt;
	char fname_stab[1000];

	A_triangle = A->restricted_action(
			triangle, nb_pts_triangle, 0 /* verbose_level */);
	Triangle = NEW_OBJECT(set_and_stabilizer);
	Triangle->init(A, A, 0 /* verbose_level */);
	Triangle->init_data(triangle, nb_pts_triangle,
			0 /* verbose_level */);

	sprintf(fname_stab, "PGL_4_%d_Grassmann_4_2_%d_stab_gens_3_2.txt", q, q);
	Triangle->init_stab_from_file(fname_stab, 0 /* verbose_level */);
	Triangle->Strong_gens->test_if_set_is_invariant_under_given_action(
			A, triangle, nb_pts_triangle,
			0 /* verbose_level */);
	cout << "The trangle is invariant under the given group" << endl;

#if 0
	int Eckardt_pts[] = {7,12,15,24,58,91}; // these were the E-points from before. Not any more.
	int Eckardt_pts2[] = {15,24,5,14,25,124}; // these are the n e w E-points
	int nb_E = 6;
#endif

	Elt = NEW_int(A->elt_size_in_int);
	cout << "before move_point_set" << endl;
	move_point_set(A_triangle, Triangle,
			Eckardt_pts, nb_Eckardt_pts,
			Elt, Eckardt_stab,
			verbose_level - 2);
	cout << "after move_point_set" << endl;

	cout << "The stabilizer of the Eckardt points is:" << endl;
	Eckardt_stab->Strong_gens->print_generators(cout);
	cout << "a group of order " << Eckardt_stab->target_go << endl;


#if 0

	// this is how we find the third transformation:

	set_and_stabilizer_apply(SaS, Elt, FALSE, verbose_level);

	cout << "The transporter is:" << endl;
	A->print_for_make_element(cout, Elt);
	cout << endl;

	cout << "the moved set is:" << endl;
	int_vec_print(cout, SaS->data, SaS->sz);
	cout << endl;

	cout << "this is the third transformation, stop" << endl;
	exit(1);
#endif


#if 0
	action *A_triangle_reduced;
	int *triangle_reduced;
	int nb_pts_triangle_reduced;

	triangle_reduced = NEW_int(nb_pts_triangle);
	int_vec_copy(triangle, triangle_reduced, nb_pts_triangle);
	nb_pts_triangle_reduced = nb_pts_triangle;

	int_vec_take_away(triangle_reduced, nb_pts_triangle_reduced,
			Eckardt_pts2, nb_Eckardt_pts);

	cout << "The triangle without Eckardt points: " << endl;
	int_vec_print(cout, triangle_reduced, nb_pts_triangle_reduced);
	cout << endl;
	cout << "nb_pts_triangle_reduced=" << nb_pts_triangle_reduced << endl;

	cout << "setting up A_triangle_reduced" << endl;
	A_triangle_reduced = A->restricted_action(
			triangle_reduced, nb_pts_triangle_reduced,
			0 /* verbose_level */);
	cout << "setting up A_triangle_reduced done" << endl;


	Eckardt_stab->init_data(
			triangle_reduced,
			nb_pts_triangle_reduced,
			0 /* verbose_level*/);

	cout << "testing if the reduced set is invariant "
			"under the stabilizer of the E-pts:" << endl;
	Eckardt_stab->Strong_gens->test_if_set_is_invariant_under_given_action(
			A, triangle_reduced, nb_pts_triangle_reduced,
			0 /* verbose_level */);

	cout << "Yes!" << endl;
#endif


#if 0
	cout << "testing if the double pts are invariant "
			"under the stabilizer of the E-pts:" << endl;
	Eckardt_stab->Strong_gens->test_if_set_is_invariant_under_given_action(
			A, Double_pts, nb_double_pts,
			0 /* verbose_level */);

	cout << "Yes!" << endl;



#endif


	cout << "Creating restricted action on Double_pts:" << endl;
	A_on_double_pts = A->restricted_action(
			Double_pts, nb_double_pts,
			0 /* verbose_level */);

	schreier *Orb2;

	cout << "computing orbits on double points off the triangle:" << endl;
	Orb2 = SaS->Strong_gens->orbits_on_points_schreier(
			A_on_double_pts, verbose_level);
	cout << "orbits on double points off the triangle:" << endl;
	Orb2->print_and_list_orbits(cout);


	int idx, f, l, a, b;

	if (!Orb2->find_shortest_orbit_if_unique(idx)) {
		cout << "the shortest orbit on double points "
				"is not unique" << endl;
		exit(1);
		}

	f = Orb2->orbit_first[idx];
	l = Orb2->orbit_len[idx];
	cout << "The unique shortest orbit on double points off "
			"the triangle has length " << l << ":" << endl;
	for (i = 0; i < l; i++) {
		a = Orb2->orbit[f + i];
		b = Double_pts[a];
		//cout << i << " : " << a << " : " << b << endl;
		}


	long int *short_orbit;
	int short_orbit_len;

	f = Orb2->orbit_first[idx];
	short_orbit_len = Orb2->orbit_len[idx];
	short_orbit = NEW_lint(short_orbit_len);
	for (i = 0; i < short_orbit_len; i++) {
		a = Orb2->orbit[f + i];
		b = Double_pts[a];
		short_orbit[i] = b;
		}
	Orbiter->Lint_vec->print(cout, short_orbit, short_orbit_len);
	cout << endl;


	orbit_of_sets *OS;

	OS = NEW_OBJECT(orbit_of_sets);
	cout << "before OS->init" << endl;
	OS->init(A, A, short_orbit, short_orbit_len,
			Eckardt_stab->Strong_gens->gens, verbose_level);
	cout << "after OS->init" << endl;

	long int *six_point_orbit;
	int six_point_orbit_length;
	int sz;
	action *A_on_six_point_orbit;

	OS->get_table_of_orbits(six_point_orbit,
			six_point_orbit_length, sz, verbose_level);

	cout << "Creating action on six_point_orbit" << endl;
	A_on_six_point_orbit = A->create_induced_action_on_sets(
			six_point_orbit_length, sz, six_point_orbit,
			verbose_level);
	cout << "Creating action on six_point_orbit done" << endl;

	schreier *Orb_six_points;

	cout << "computing orbits on set of six points:" << endl;
	Orb_six_points = Eckardt_stab->Strong_gens->orbit_of_one_point_schreier(
			A_on_six_point_orbit,
			OS->position_of_original_set,
			verbose_level);
	cout << "orbits on set of six points:" << endl;
	Orb_six_points->print(cout);
	//Orb_six_points->print_and_list_orbits(cout);

	l = Orb_six_points->orbit_len[0];


	int *Coeff;


	cout << "computing Coeff" << endl;

	Coeff = NEW_int(l * 20);
	for (j = 0; j < l; j++) {
		if ((j % 1000) == 0) {
			cout << "coset " << j << " / " << l << ":" << endl;
			}
		Orb_six_points->coset_rep(j, 0 /* verbose_level */);
		A->element_move(Orb_six_points->cosetrep, Elt, 0);

		set_and_stabilizer *SaS2;


		SaS2 = SaS->create_copy(0 /* verbose_level */);
		SaS2->apply_to_self(Elt, 0 /* verbose_level */);

		Surf->build_cubic_surface_from_lines(SaS2->sz,
				SaS2->data, coeff,
				0 /* verbose_level */);
		F->PG_element_normalize_from_front(coeff, 1, 20);

#if 0
		cout << "coefficient vector of the surface: ";
		int_vec_print(cout, coeff, 20);
		cout << endl;
		cout << "equation: ";
		Surf->print_equation(cout, coeff);
		cout << endl;
#endif

		Orbiter->Int_vec->copy(coeff, Coeff + j * 20, 20);

		FREE_OBJECT(SaS2);
		}

#if 1
	cout << "Coeff:" << endl;
	for (i = 0; i < l; i++) {

		int *co;

		co = Coeff + i * 20;
		if ((co[16] == co[17]) && (co[17] == co[18])) {
			cout << i << " / " << l << " : ";
			Orbiter->Int_vec->print(cout, co, 20);
			cout << endl;
			}
		}
	//int_matrix_print(Coeff, l, 20);
#endif



#if 1
	// this is how we find the fourth transformation:

	if (q == 11 && nb == 0) {
		//j = 298;
		j = 4051;
		}
	else if (q == 13 && nb == 1) {
		j = 2963;
		}
	else if (q == 17 && nb == 0) {
		j = 38093;
		}
	else if (q == 17 && nb == 2) {
		j = 3511;
		}
	else if (q == 17 && nb == 6) {
		j = 39271;
		}
	else {
		j = -1;
		}


	if (j >= 0) {
		cout << "coset " << j << " / " << l << ":" << endl;
		Orb_six_points->coset_rep(j, 0 /* verbose_level */);
		A->element_move(Orb_six_points->cosetrep, Elt, 0);

		set_and_stabilizer *SaS2;

		SaS2 = SaS->create_copy(0 /* verbose_level */);

		SaS2->apply_to_self(Elt, verbose_level);

		Surf->build_cubic_surface_from_lines(SaS2->sz,
				SaS2->data, coeff,
				0 /* verbose_level */);
		F->PG_element_normalize_from_front(coeff, 1, 20);
		cout << "coefficient vector of the surface: ";
		Orbiter->Int_vec->print(cout, coeff, 20);
		cout << endl;
		cout << "equation: ";
		Surf->print_equation(cout, coeff);
		cout << endl;

		FREE_OBJECT(SaS2);
		}
#endif


	if (f_v) {
		cout << "surface_study::study_surface_with_6_eckardt_points done" << endl;
		}
}


#if 0

	if (nb_Eckardt_pts == 4) {
		cout << "nb_Eckardt_pts = 4" << endl;


		int Basis1[] = {1,0,0,0,0,0,1,0};
		int Basis2[] = {0,1,0,0,0,0,1,0};
		int Basis3[] = {0,0,1,0,1,1,0,0};
		int desired_lines[3];

		desired_lines[0] = Surf->Gr->rank_int_here(Basis1, 0 /*verbose_level*/);
		desired_lines[1] = Surf->Gr->rank_int_here(Basis2, 0 /*verbose_level*/);
		desired_lines[2] = Surf->Gr->rank_int_here(Basis3, 0 /*verbose_level*/);

		cout << "desired_lines : ";
		int_vec_print(cout, desired_lines, 3);
		cout << endl;


		int *pts_on_three_lines;
		int nb_pts_on_three_lines;
		int three_lines_idx[3] = {169, 30927, 352}; // this is in PG(3,13)
		int a, b, c, idx;

		cout << "creating the three lines" << endl;

		pts_on_three_lines = NEW_int(3 * (q + 1));
		nb_pts_on_three_lines = 0;

		for (i = 0; i < 3; i++) {
			for (j = 0; j < q + 1; j++) {
				a = three_lines_idx[i];
				b = Surf->P->Lines[a * (q + 1) + j];
				if (int_vec_search(pts_on_three_lines,
						nb_pts_on_three_lines, b, idx)) {
					}
				else {
					for (c = nb_pts_on_three_lines; c > idx; c--) {
						pts_on_three_lines[c] = pts_on_three_lines[c - 1];
						}
					pts_on_three_lines[idx] = b;
					nb_pts_on_three_lines++;
					}
				}
			}

		int_vec_take_away(
				pts_on_three_lines, nb_pts_on_three_lines,
				Eckardt_pts, nb_Eckardt_pts);
		cout << "After taking away the Eckardt points, we still have "
				<< nb_pts_on_three_lines
				<< " points on the three lines" << endl;
		int_vec_print_as_table(cout,
				pts_on_three_lines, nb_pts_on_three_lines, 10);
		cout << endl;


#if 1
		int_vec_take_away(Double_pts, nb_double_pts,
				pts_on_three_lines, nb_pts_on_three_lines);
		//int_vec_take_away(Double_pts, nb_double_pts, three_points, 3);

		cout << "After taking away the triangle points, "
				"we still have " << nb_double_pts
				<< " double points" << endl;
		int_vec_print_as_table(cout,
				Double_pts, nb_double_pts, 10);
		cout << endl;
#endif


		set_and_stabilizer *Three_lines;
		//set_and_stabilizer *Eckardt_stab;
		action *A_on_pts_on_three_lines;
		//int *Elt;
		char fname_stab[1000];

		A_on_pts_on_three_lines = A->restricted_action(
				pts_on_three_lines, nb_pts_on_three_lines,
				0 /* verbose_level */);
		Three_lines = NEW_OBJECT(set_and_stabilizer);
		Three_lines->init(A, A, 0 /* verbose_level */);
		Three_lines->init_data(
				pts_on_three_lines, nb_pts_on_three_lines,
				0 /* verbose_level */);

		sprintf(fname_stab, "PGL_4_%d_stab_gens_4_1.txt", q);
		cout << "Reading group from file " << fname_stab << endl;
		Three_lines->init_stab_from_file(fname_stab, verbose_level);
		Three_lines->Strong_gens->test_if_set_is_invariant_under_given_action(
			A,
			pts_on_three_lines, nb_pts_on_three_lines,
			0 /* verbose_level */);
		cout << "The points on the three lines "
				"are invariant under the given group" << endl;

		//int interesting_points[6] = {25,20,133,68,72,142};
		int interesting_points[6] = {17,28,29,172,30,184};
		int interesting_pt_idx[6];

		for (i = 0; i < 6; i++) {
			if (!int_vec_search(
					pts_on_three_lines, nb_pts_on_three_lines,
					interesting_points[i], idx)) {
				cout << "could not find interesting point" << endl;
				exit(1);
				}
			interesting_pt_idx[i] = idx;
			}
		cout << "interesting_points: ";
		int_vec_print(cout, interesting_points, 6);
		cout << endl;
		cout << "interesting_pt_idx: ";
		int_vec_print(cout, interesting_pt_idx, 6);
		cout << endl;

		set_and_stabilizer *Interesting_pts_stab;
		int *Elt;
		char fname_stabilizer_stage_two[1000];
		char fname_stabilizer_stage_three[1000];

		sprintf(fname_stabilizer_stage_two, "%s_stage2_stab.txt", prefix);
		sprintf(fname_stabilizer_stage_three, "%s_stage3_stab.txt", prefix);
		Elt = NEW_int(A->elt_size_in_int);


#if 0
		move_point_set(A_on_pts_on_three_lines, Three_lines,
			interesting_points, 6, Elt, Interesting_pts_stab, verbose_level);

		cout << "Writing generators to file "
				<< fname_stabilizer_stage_two << endl;
		Interesting_pts_stab->Strong_gens->print_generators_in_source_code_to_file(
				fname_stabilizer_stage_two);
		cout << "Written file " << fname_stabilizer_stage_two
				<< " of size " << file_size(fname_stabilizer_stage_two) << endl;
		exit(1);
#else
		cout << "Reading generators from file "
				<< fname_stabilizer_stage_two << endl;
		Interesting_pts_stab = NEW_OBJECT(set_and_stabilizer);
		Interesting_pts_stab->init(A, A, 0 /* verbose_level*/);
		Interesting_pts_stab->init_stab_from_file(
				fname_stabilizer_stage_two, verbose_level);
		cout << "Generators are:" << endl;
		Interesting_pts_stab->Strong_gens->print_generators_tex();
#endif




		//int stage3_points[3] = {1878,1883,1948};
		int stage3_points[3] = {3,185,198};

		schreier *Orb3;

		Orb3 = Interesting_pts_stab->Strong_gens->orbits_on_points_schreier(
				A, verbose_level);
		cout << "orbits in stage3:" << endl;
		Orb3->print_and_list_orbits(cout);


		int *set;
		int sz;
		int orbit_idx;

		set = NEW_int(A->degree);
		orbit_idx = Orb3->orbit_number(stage3_points[0]);
		Orb3->get_orbit(orbit_idx, set, sz, 0 /* verbose_level */);

		int_vec_heapsort(set, sz);

		cout << "We will consider the set of size " << sz << ":" << endl;
		int_vec_print_as_table(cout, set, sz, 25);
		cout << endl;

		action *A_on_stage3_set;

		cout << "Creating restricted action on Stage 3 set:" << endl;
		A_on_stage3_set = A->restricted_action(
				set, sz, 0 /* verbose_level */);

		set_and_stabilizer *Stage3_set;

		Stage3_set = NEW_OBJECT(set_and_stabilizer);
		Stage3_set->init(A, A, 0 /* verbose_level */);
		Stage3_set->init_data(set, sz, 0 /* verbose_level */);

		cout << "Reading group from file "
				<< fname_stabilizer_stage_two << endl;
		Stage3_set->init_stab_from_file(
				fname_stabilizer_stage_two, verbose_level);
		Stage3_set->Strong_gens->test_if_set_is_invariant_under_given_action(
			A,
			set, sz, 0 /* verbose_level */);
		cout << "The points on the three lines are invariant "
				"under the given group" << endl;



		set_and_stabilizer *Stage3_Interesting_pts_stab;


		cout << "before move_point_set" << endl;
		move_point_set(A_on_stage3_set, Stage3_set,
			stage3_points, 3, Elt, Stage3_Interesting_pts_stab,
			verbose_level - 2);
		cout << "after move_point_set" << endl;

		cout << "Writing generators to file "
				<< fname_stabilizer_stage_three << endl;
		Stage3_Interesting_pts_stab->Strong_gens->print_generators_in_source_code_to_file(
				fname_stabilizer_stage_three);
		cout << "Written file " << fname_stabilizer_stage_three
				<< " of size " << file_size(fname_stabilizer_stage_three)
				<< endl;

		cout << "Generators are:" << endl;
		Stage3_Interesting_pts_stab->Strong_gens->print_generators_tex();

		{
		char fname_intersection_pts[1000];
		char fname_intersection_pts_tex[1000];
		sprintf(fname_intersection_pts,
				"%s_intersection_points3.csv", prefix);
		sprintf(fname_intersection_pts_tex,
				"%s_intersection_points3.tex", prefix);

		int_matrix_write_csv(fname_intersection_pts,
				Intersection_pt, SaS->sz, SaS->sz);
		cout << "Written file " << fname_intersection_pts
				<< " of size " << file_size(fname_intersection_pts) << endl;
		{
		ofstream fp(fname_intersection_pts_tex);
		latex_head_easy(fp);
		//latex_head_easy_sideways(fp);
		fp << "{\\tiny \\arraycolsep=1pt" << endl;
		fp << "$$" << endl;
		int_matrix_print_with_labels_and_partition(fp,
			Intersection_pt, SaS->sz, SaS->sz,
			orbit, orbit,
			Orb->orbit_first, Orb->orbit_len, Orb->nb_orbits,
			Orb->orbit_first, Orb->orbit_len, Orb->nb_orbits,
			matrix_entry_print, (void *) Surf,
			TRUE /* f_tex */);
		fp << "$$}" << endl;
		latex_foot(fp);
		}
		cout << "Written file " << fname_intersection_pts_tex
				<< " of size "
				<< file_size(fname_intersection_pts_tex) << endl;
		}


		}

	else if (nb_Eckardt_pts == 6) {


		} // nb_E = 6



	//FREE_int(Elt);

	FREE_int(Eckardt_pts);
	FREE_int(Double_pts);
	FREE_int(R);
	FREE_int(Adj);
	FREE_int(orbit_first);
	FREE_int(orbit_length);
	FREE_OBJECT(SaS);
	FREE_OBJECT(A_on_lines);

	FREE_int(coeff);
	FREE_OBJECT(S);
	FREE_OBJECT(A2);
	FREE_OBJECT(A);

	FREE_OBJECT(Surf);
	FREE_OBJECT(F);
}



void compute_decomposition(orthogonal *O,
		finite_field *F, int *Pts, int nb_pts,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "compute_decomposition" << endl;
		}
	int N, ht0, depth;

	partitionstack S;

	N = O->nb_points + O->nb_lines;

	//O.make_initial_partition(S, verbose_level);
	S.allocate(N, FALSE);
	// split off the column class:
	S.subset_continguous(O->nb_points, O->nb_lines);
	S.split_cell(FALSE);

	ht0 = S.ht;

	cout << "ht = " << S.ht << endl;
	cout << "before S.refine_arbitrary_set" << endl;
	S.refine_arbitrary_set(nb_pts, Pts, verbose_level - 1);
	cout << "after S.refine_arbitrary_set" << endl;

	if (f_v) {
		cout << "compute_decomposition before S.compute_TDO" << endl;
		}

	depth = 1;
	S.compute_TDO(*O, ht0, -1, -1, depth, verbose_level - 2);
	if (f_v) {
		cout << "compute_decomposition after S.compute_TDO" << endl;
		}
	if (EVEN(depth)) {
		S.get_and_print_row_decomposition_scheme(*O, -1, -1);
		}
	else {
		S.get_and_print_col_decomposition_scheme(*O, -1, -1);
		}


	if (f_v) {
		cout << "compute_decomposition done" << endl;
		}
}
#endif

void move_point_set(action *A2,
	set_and_stabilizer *Universe,
	long int *Pts, int nb_pts,
	int *Elt,
	set_and_stabilizer *&new_stab,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "move_point_set" << endl;
		}
	poset_classification_control *Control;
	poset_with_group_action *Poset;
	poset_classification *gen;
	//char prefix[1000];
	//int f_W = FALSE;
	//int f_w = FALSE;
	long int *data_in;
	long int *data_out;
	long int *data2;
	int idx, i;
	data_structures::sorting Sorting;

	//prefix[0] = 0;


	if (f_v) {
		cout << "move_point_set computing orbits "
				"on subsets of size " << nb_pts << endl;
		}

	Control = NEW_OBJECT(poset_classification_control);
	Poset = NEW_OBJECT(poset_with_group_action);
	Poset->init_subset_lattice(
			Universe->A, A2,
			Universe->Strong_gens,
			verbose_level);



	gen = NEW_OBJECT(poset_classification);

	gen->compute_orbits_on_subsets(
		nb_pts,
		//prefix,
		//f_W, f_w,
		Control,
		Poset,
		verbose_level - 2);

	data_in = NEW_lint(nb_pts);
	data_out = NEW_lint(nb_pts);
	data2 = NEW_lint(nb_pts);

	for (i = 0; i < nb_pts; i++) {

		data_in[i] = Universe->find(Pts[i]);

		}

	if (f_v) {
		cout << "identifying " << nb_pts << " points:" << endl;
		}
	idx = gen->trace_set(data_in, nb_pts, nb_pts,
		data_out, Elt,
		0 /* verbose_level */);

	if (f_v) {
		cout << "idx = " << idx << " data_out = ";
		Orbiter->Lint_vec->print(cout, data_out, nb_pts);
		cout << endl;
		}

	for (i = 0; i < nb_pts; i++) {
		data2[i] = Universe->data[data_out[i]];
		}

	if (f_v) {
		cout << "data2 = ";
		Orbiter->Lint_vec->print(cout, data2, nb_pts);
		cout << endl;
		}

	if (f_v) {
		cout << "transporter:" << endl;
		Universe->A->element_print_quick(Elt, cout);
		cout << endl;
		}



	new_stab = gen->get_set_and_stabilizer(
			nb_pts,
			idx /* orbit_at_level */,
			verbose_level);

	if (f_v) {
		cout << "pulled out set and stabilizer at level "
				<< nb_pts << " for orbit " << idx << ":" << endl;
		cout << "a set with a group of order "
				<< new_stab->target_go << endl;
		}


	FREE_lint(data_in);
	FREE_lint(data_out);
	FREE_lint(data2);
	FREE_OBJECT(gen);
	FREE_OBJECT(Poset);
	FREE_OBJECT(Control);

	if (f_v) {
		cout << "move_point_set done" << endl;
		}
}

void matrix_entry_print(long int *p,
		int m, int n, int i, int j, int val,
		std::string &output, void *data)
{
	surface_domain *Surf;
	Surf = (surface_domain *) data;
	char str[1000];

	if (i == -1) {
		strcpy(str, Surf->Schlaefli->Labels->Line_label_tex[val].c_str());
		}
	else if (j == -1) {
		strcpy(str, Surf->Schlaefli->Labels->Line_label_tex[val].c_str());
		}
	else {
		if (val == -1) {
			strcpy(str, ".");
			}
		else {
			sprintf(str, "%d", val);
			}
		}
	output.assign(str);
}





}}

