/*
 * orbits_on_polynomials.cpp
 *
 *  Created on: Nov 28, 2020
 *      Author: betten
 */






#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {


orbits_on_polynomials::orbits_on_polynomials()
{
	LG = NULL;
	degree_of_poly = 0;

	F = NULL;
	A = NULL;
	n = 0;
	// go;
	HPD = NULL;
	A2 = NULL;
	Elt1 = Elt2 = Elt3 = NULL;
	Sch = NULL;
	// full_go
	//fname_base
	//fname_csv
	//fname_report
	T = NULL;
	Nb_pts = NULL;

}

orbits_on_polynomials::~orbits_on_polynomials()
{
}

void orbits_on_polynomials::init(
		linear_group *LG,
		int degree_of_poly,
		int f_recognize, std::string &recognize_text,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::init" << endl;
	}

	orbits_on_polynomials::LG = LG;
	orbits_on_polynomials::degree_of_poly = degree_of_poly;



	A = LG->A_linear;
	F = A->matrix_group_finite_field();
	A->group_order(go);

	n = A->matrix_group_dimension();

	cout << "n = " << n << endl;

	cout << "strong generators:" << endl;
	//A->Strong_gens->print_generators();
	A->Strong_gens->print_generators_tex();

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);


	monomial_ordering_type Monomial_ordering_type = t_PART;


	HPD->init(F, n /* nb_var */, degree_of_poly,
			TRUE /* f_init_incidence_structure */,
			Monomial_ordering_type,
			verbose_level);

	A2 = NEW_OBJECT(action);
	A2->induced_action_on_homogeneous_polynomials(A,
		HPD,
		FALSE /* f_induce_action */, NULL,
		verbose_level);

	cout << "created action A2" << endl;
	A2->print_info();


	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	char str[1000];

	sprintf(str, "poly_orbits_d%d_n%d_q%d", degree_of_poly, n - 1, F->q);
	fname_base.assign(str);
	fname_csv.assign(fname_base);
	fname_csv.append(".csv");


	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);


	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before A->Strong_gens->orbits_on_points_schreier" << endl;
	}


	Sch = A->Strong_gens->orbits_on_points_schreier(A2, verbose_level);

	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after A->Strong_gens->orbits_on_points_schreier" << endl;
	}



	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before Sch->write_orbit_summary" << endl;
	}
	Sch->write_orbit_summary(fname_csv,
			A /*default_action*/,
			go,
			verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after Sch->write_orbit_summary" << endl;
	}



	A->group_order(full_go);
	T = NEW_OBJECT(orbit_transversal);

	if (f_v) {
		cout << "orbits_on_polynomials::init before T->init_from_schreier" << endl;
	}

	T->init_from_schreier(
			Sch,
			A,
			full_go,
			verbose_level);

	if (f_v) {
		cout << "orbits_on_polynomials::init after T->init_from_schreier" << endl;
	}



	Sch->print_orbit_reps(cout);


	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"before compute_points" << endl;
	}
	compute_points(verbose_level);
	if (f_v) {
		cout << "orbits_on_polynomials::init "
				"after compute_points" << endl;
	}



	if (f_recognize) {
		long int *Rank;
		int len;
		int i;

		int *Idx;


		cout << "orbits_on_polynomials::init recognition:" << endl;
		lint_vec_scan(recognize_text, Rank, len);

		Idx = NEW_int(len);

		for (i = 0; i < len; i++) {
			//cout << "recognizing object " << i << " / " << len << " which is " << Rank[i] << endl;
			int orbit_idx;
			orbit_idx = Sch->orbit_number(Rank[i]);
			Idx[i] = orbit_idx;
			cout << "recognizing object " << i << " / " << len << ", point " << Rank[i] << " lies in orbit " << orbit_idx << endl;
		}
		file_io Fio;
		std::string fname;

		fname.assign(fname_base);
		fname.append("_recognition.csv");

		Fio.int_vec_write_csv(Idx, len, fname, "Idx");

		FREE_lint(Rank);

	}


#if 0
	int f, l, a, r;
	longinteger_object go, go1;
	longinteger_domain D;
	int *coeff;
	int *line_type;
	long int *Pts;
	int nb_pts;
	int *Kernel;
	//int h, pt, orbit_idx;
	sorting Sorting;
	int i;


	Pts = NEW_lint(HPD->get_P()->N_points);
	coeff = NEW_int(HPD->get_nb_monomials());
	line_type = NEW_int(HPD->get_P()->N_lines);
	Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());


	A->group_order(go);

	for (i = 0; i < Sch->nb_orbits; i++) {
		cout << "Orbit " << i << " / " << Sch->nb_orbits << ":" << endl;
		f = Sch->orbit_first[i];
		l = Sch->orbit_len[i];

		D.integral_division_by_int(go, l, go1, r);

		cout << "stab order " << go1 << endl;
		a = Sch->orbit[f];
		cout << "orbit length = " << l << endl;
		cout << "orbit rep = " << a << endl;

		HPD->unrank_coeff_vector(coeff, a);
		int_vec_print(cout, coeff, HPD->get_nb_monomials());
		cout << " = ";
		HPD->print_equation(cout, coeff);
		cout << endl;


		vector<long int> Points;

		HPD->enumerate_points(coeff, Points, verbose_level);

		nb_pts = Points.size();
		Pts = NEW_lint(nb_pts);
		for (int u = 0; u < nb_pts; u++) {
			Pts[u] = Points[u];
		}


		cout << "We found " << nb_pts << " points on the curve" << endl;
		cout << "They are : ";
		lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);

		HPD->get_P()->line_intersection_type(Pts, nb_pts,
				line_type, 0 /* verbose_level */);

		cout << "The line type is: ";
		int_vec_print_classified(line_type, HPD->get_P()->N_lines);
		cout << "after int_vec_print_classified" << endl;

		cout << "The stabilizer is generated by:" << endl;
		T->Reps[i].Strong_gens->print_generators_tex(cout);


#if 0
		HPD->vanishing_ideal(Pts, nb_pts, r, Kernel, 0 /*verbose_level */);

		cout << "The system has rank " << r << endl;
		cout << "The ideal has dimension " << HPD->nb_monomials - r << endl;
		cout << "and is generated by:" << endl;
		int_matrix_print(Kernel, HPD->nb_monomials - r, HPD->nb_monomials);
		cout << "corresponding to the following basis of polynomials:" << endl;
		for (h = 0; h < HPD->nb_monomials - r; h++) {
			HPD->print_equation(cout, Kernel + h * HPD->nb_monomials);
			cout << endl;
			}
#endif



		if (f_stabilizer) {
			strong_generators *SG;
			longinteger_object stab_go;
			int canonical_pt;

			cout << "before set_stabilizer_in_projective_space" << endl;
			SG = A->set_stabilizer_in_projective_space(
				HPD->get_P(),
				Pts, nb_pts,
				canonical_pt, NULL,
				//FALSE, NULL,
				0 /*verbose_level */);




			SG->group_order(stab_go);

			cout << "stabilizer order = " << stab_go << endl;
			cout << "stabilizer generators:" << endl;
			SG->print_generators_tex();
			FREE_OBJECT(SG);
			}


		if (f_draw_tree) {
			{

			char label[1000];
			char fname[1000];
			int xmax = 1000000;
			int ymax =  500000;
#if 0
			int f_circletext = TRUE;
			int rad = 30000;
#else
			int f_circletext = FALSE;
			int rad = 2000;
#endif

			cout << "before Sch->draw_tree" << endl;
			sprintf(label, "orbit_tree_q%d_d%d_%d", F->q, n, i);
			sprintf(fname, "orbit_tree_q%d_d%d_%d_tables.tex", F->q, n, i);
			Sch->draw_tree(label,
				i, xmax, ymax,
				f_circletext, rad,
				TRUE /* f_embedded */, FALSE /* f_sideways */,
				0.3 /* scale */, 1. /* line_width */,
				FALSE, NULL,
				0 /*verbose_level */);

			{
			ofstream fp(fname);
			latex_interface L;

			L.head_easy(fp);


			A->Strong_gens->print_generators_tex(fp);

			Sch->print_and_list_orbits_tex(fp);

			Sch->print_tables_latex(fp, TRUE /* f_with_cosetrep */);

			L.foot(fp);
			}
			cout << "Written file " << fname << " of size "
					<< Fio.file_size(fname) << endl;

			}
			}



		}
#endif

#if 0
	if (nb_identify) {
		cout << "starting the identification:" << endl;
		for (h = 0; h < nb_identify; h++) {



			cout << "Trying to identify " << h << " / "
				<< nb_identify << endl;

			cout << "which is "
				<< Identify_label[h] << " : "
				<< Identify_coeff_text[h] << endl;

			int *coeff_list;
			int nb_coeffs;
			int a, b;

			int_vec_scan(Identify_coeff_text[h],
					coeff_list, nb_coeffs);
			if (ODD(nb_coeffs)) {
				cout << "number of coefficients must be even" << endl;
				exit(1);
				}

			int_vec_zero(coeff, HPD->nb_monomials);
			for (i = 0; i < nb_coeffs >> 1; i++) {
				a = coeff_list[2 * i + 0];
				b = coeff_list[2 * i + 1];
				if (b >= HPD->nb_monomials) {
					cout << "b >= HPD->nb_monomials" << endl;
					exit(1);
					}
				coeff[b] = a;
				}

			cout << "The equation of the input surface is:" << endl;
			int_vec_print(cout, coeff, HPD->nb_monomials);
			cout << endl;


			pt = HPD->rank_coeff_vector(coeff);
			cout << "pt=" << pt << endl;


			HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);
			cout << "We found " << nb_pts << " points on the curve" << endl;
			cout << "They are : ";
			lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			HPD->P->print_set_numerical(cout, Pts, nb_pts);

			char fname[1000];

			sprintf(fname, "point_set_identify_%d", h);
			//draw_point_set_in_plane(HPD, Pts, nb_pts,
			//fname, TRUE /* f_with_points */);

			cout << "before HPD->P->draw_point_set_in_plane" << endl;
			HPD->P->draw_point_set_in_plane(fname,
					Pts, nb_pts,
					TRUE /* f_with_points */,
					FALSE /* f_point_labels */,
					TRUE /* f_embedded */,
					FALSE /* f_sideways */,
					17000 /* rad */,
					verbose_level);
			cout << "after HPD->P->draw_point_set_in_plane" << endl;

			strong_generators *SG;
			longinteger_object ago;
			cout << "before Sch->transporter_from_point_to_orbit_rep" << endl;
			Sch->transporter_from_point_to_orbit_rep(pt,
					orbit_idx, Elt1, 5 /* verbose_level */);
			cout << "after Sch->transporter_from_point_to_orbit_rep" << endl;



			SG = NEW_OBJECT(strong_generators);
			SG->init(A);
			cout << "before SG->init_point_stabilizer_of_arbitrary_"
					"point_through_schreier" << endl;
			SG->init_point_stabilizer_of_arbitrary_point_through_schreier(
				Sch,
				pt, orbit_idx, go /*full_group_order */,
				0 /* verbose_level */);
			cout << "The given pt " << pt << " lies in orbit " << orbit_idx << endl;
			cout << "orbit has length " << Sch->orbit_len[orbit_idx] << endl;
			cout << "A transporter from the point to the orbit rep is:" << endl;
			A->element_print(Elt1, cout);

			Sch->print_generators();

			char label[1000];
			int xmax = 1000000;
			int ymax =  500000;
			int f_circletext = FALSE;
			int rad = 2000;

			sprintf(label, "orbit_tree_q%d_d%d_%d", q, degree, orbit_idx);
			cout << "before Sch->draw_tree" << endl;
			Sch->draw_tree(label,
				orbit_idx, xmax, ymax, f_circletext, rad,
				TRUE /* f_embedded */, FALSE /* f_sideways */,
				0.3 /* scale */, 1. /* line_width */,
				FALSE, NULL,
				0 /*verbose_level */);


			SG->group_order(ago);
			cout << "The stabilizer is a group of order " << ago << endl;

			cout << "generators for the stabilizer of "
					"pt = " << pt << " are:" << endl;
			SG->print_generators();


			cout << "before set_stabilizer_in_projective_space" << endl;
			strong_generators *SG2;
			longinteger_object stab_go;
			int canonical_pt;
			SG2 = A->set_stabilizer_in_projective_space(
				HPD->P,
				Pts, nb_pts, canonical_pt, NULL,
				FALSE, NULL,
				verbose_level + 3);

			SG2->group_order(stab_go);

			cout << "stabilizer order = " << stab_go << endl;
			cout << "stabilizer generators:" << endl;
			SG2->print_generators_tex();
			FREE_OBJECT(SG2);

			//sims *Stab;
			//action *AR;
			int *perm;
			int *Elt;

			Elt = NEW_int(A->elt_size_in_int);
			perm = NEW_int(nb_pts);
			//Stab = SG->create_sims(0 /*verbose_level*/);
			cout << "creating restricted action on the curve:" << endl;
			//AR = A->restricted_action(Pts, nb_pts, 0 /* verbose_level */);


			int a1, a2, a3, a4, a6;
			//int A6[6];

			if (!HPD->test_weierstrass_form(pt,
				a1, a2, a3, a4, a6, 0 /* verbose_level */)) {
				cout << "Not in Weierstrass form" << endl;
				exit(1);
				}
			cout << "The curve is in Weierstrass form: "
					"a1=" << a1
					<< " a2=" << a2
					<< " a3=" << a3
					<< " a4=" << a4
					<< " a6=" << a6 << endl;
#if 0
			A6[0] = a1;
			A6[1] = a2;
			A6[2] = a3;
			A6[3] = a4;
			A6[4] = 0;
			A6[5] = a6;
#endif

			//c = HPD->P->elliptic_curve_addition(A6,
			//Pts[0], Pts[1], 2 /*verbose_level*/);
			//cout << "c=" << c << endl;

#if 0
			int *Table;
			cout << "before HPD->P->elliptic_curve_addition_table" << endl;
			HPD->P->elliptic_curve_addition_table(A6,
					Pts, nb_pts, Table, verbose_level);
			cout << "The addition table:" << endl;
			int_matrix_print(Table, nb_pts, nb_pts);
			int_matrix_print_tex(cout, Table, nb_pts, nb_pts);


			cout << "The group elements acting on the curve:" << endl;
			Stab->print_all_group_elements_as_permutations_in_special_action(AR);

			int order, k, idx;
			int *Idx;

			Idx = NEW_int(nb_pts);
			for (k = 0; k < nb_pts; k++) {
				Idx[k] = -1;
				}

			order = ago.as_int();
			for (i = 0; i < order; i++) {
				cout << "elt " << i << " / " << order << " : ";
				Stab->element_as_permutation(AR, i, perm, 0 /*verbose_level */);
				for (j = 0; j < nb_pts; j++) {
					cout << perm[j] << ", ";
					}
				cout << " ; ";
				for (k = 0; k < nb_pts; k++) {
					if (int_vec_compare(perm, Table + k * nb_pts, nb_pts) == 0) {
						cout << k << " ";
						Idx[k] = i;
						}
					}
				cout << endl;
				}

			for (i = 0; i < nb_pts; i++) {
				idx = Idx[i];
				cout << "point " << i << " / " << nb_pts
						<< " corresponds to element " << idx << ":" << endl;
				Stab->element_unrank_int(idx, Elt);
				cout << "which is:" << endl;
				A->element_print(Elt, cout);
				cout << endl;
				}
#endif

			FREE_OBJECT(SG);
			FREE_int(perm);
			FREE_int(Elt);

			}
	}

	if (f_test_orbit) {
		cout << "test_orbit_idx = " << test_orbit_idx << endl;

		f = Sch->orbit_first[test_orbit_idx];
		l = Sch->orbit_len[test_orbit_idx];

		D.integral_division_by_int(go, l, go1, r);

		cout << "stab order " << go1 << endl;
		a = Sch->orbit[f];
		cout << "orbit length = " << l << endl;
		cout << "orbit rep = " << a << endl;

		HPD->unrank_coeff_vector(coeff, a);

		cout << "orbit rep is " << a << " which is ";
		int_vec_print(cout, coeff, HPD->nb_monomials);
		cout << " = ";
		HPD->print_equation(cout, coeff);
		cout << endl;

		long int *Pts2;
		long int *Pts3;
		int nb_pts2;
		int *coeff2;
		int r, b, orbit_idx;

		Pts2 = NEW_lint(HPD->P->N_points);
		Pts3 = NEW_lint(HPD->P->N_points);
		coeff2 = NEW_int(HPD->nb_monomials);

		HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);
		cout << "We found " << nb_pts << " points on the curve" << endl;

		Sorting.lint_vec_heapsort(Pts, nb_pts);

		cout << "They are : ";
		lint_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->P->print_set_numerical(cout, Pts, nb_pts);

		//r = random_integer(l);
		//cout << "Picking random integer " << r << endl;


		for (r = 0; r < l; r++) {
			cout << "orbit element " << r << " / " << l << endl;

			b = Sch->orbit[f + r];
			cout << "b=" << b << endl;

			HPD->unrank_coeff_vector(coeff2, b);


			cout << "orbit element " << b << " is ";
			int_vec_print(cout, coeff2, HPD->nb_monomials);
			cout << " = ";
			HPD->print_equation(cout, coeff2);
			cout << endl;



			HPD->enumerate_points(coeff2, Pts2, nb_pts2, verbose_level);
			cout << "We found " << nb_pts2
					<< " points on the curve" << endl;

			Sorting.lint_vec_heapsort(Pts2, nb_pts);

			cout << "They are : ";
			lint_vec_print(cout, Pts2, nb_pts2);
			cout << endl;
			HPD->P->print_set_numerical(cout, Pts2, nb_pts2);
			if (nb_pts2 != nb_pts) {
				cout << "nb_pts2 != nb_pts" << endl;
				exit(1);
				}


			Sch->transporter_from_orbit_rep_to_point(b /* pt */,
					orbit_idx, Elt1, verbose_level);
			cout << "transporter = " << endl;
			A->element_print_quick(Elt1, cout);

			A->element_invert(Elt1, Elt2, 0);

			cout << "transporter inv = " << endl;
			A->element_print_quick(Elt2, cout);

			A->map_a_set_and_reorder(Pts, Pts3, nb_pts, Elt1,
					verbose_level);
			Sorting.lint_vec_heapsort(Pts3, nb_pts);

			cout << "after apply : ";
			lint_vec_print(cout, Pts3, nb_pts);
			cout << endl;


			if (lint_vec_compare(Pts3, Pts2, nb_pts)) {
				cout << "The sets are different" << endl;
				exit(1);
				}

			cout << "The element maps the orbit representative "
					"to the new set, which is good" << endl;
			}

		FREE_lint(Pts2);
		FREE_lint(Pts3);
		FREE_int(coeff2);
		}
#endif


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

#if 0
	FREE_lint(Pts);
	FREE_int(coeff);
	FREE_int(line_type);
	FREE_int(Kernel);
#endif


	if (f_v) {
		cout << "orbits_on_polynomials::init done" << endl;
	}
}

void orbits_on_polynomials::compute_points(int verbose_level)
{
	int *coeff;
	int i;

	coeff = NEW_int(HPD->get_nb_monomials());
	Nb_pts = NEW_int(T->nb_orbits);


	for (i = 0; i < T->nb_orbits; i++) {

		longinteger_object go;
		T->Reps[i].Strong_gens->group_order(go);

		cout << i << " : ";
		lint_vec_print(cout, T->Reps[i].data, T->Reps[i].sz);
		cout << " : ";
		cout << go;

		cout << " : ";

		HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

		std::vector<long int> Pts;

		HPD->enumerate_points(coeff, Pts, verbose_level);

		Points.push_back(Pts);
		Nb_pts[i] = Pts.size();
	}
	FREE_int(coeff);

}

void orbits_on_polynomials::report(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::report" << endl;
	}
	cout << "orbit reps:" << endl;

	char title[1000];
	char author[1000];
	char str[1000];

	sprintf(str, "poly_orbits_d%d_n%d_q%d.tex", degree_of_poly, n - 1, F->q);
	fname_report.assign(str);

	sprintf(title, "Varieties of degree %d in PG(%d,%d)", degree_of_poly, n - 1, F->q);
	sprintf(author, "Orbiter");
	{
		ofstream ost(fname_report);
		latex_interface L;

		L.head(ost,
				FALSE /* f_book*/,
				TRUE /* f_title */,
				title, author,
				FALSE /* f_toc */,
				FALSE /* f_landscape */,
				TRUE /* f_12pt */,
				TRUE /* f_enlarged_page */,
				TRUE /* f_pagenumbers */,
				NULL /* extra_praeamble */);

		ost << "\\small" << endl;
		ost << "\\arraycolsep=2pt" << endl;
		ost << "\\parindent=0pt" << endl;
		ost << "$q = " << F->q << "$\\\\" << endl;
		ost << "$n = " << n - 1 << "$\\\\" << endl;
		ost << "degree of poly $ = " << degree_of_poly << "$\\\\" << endl;

		ost << "\\clearpage" << endl << endl;


		// summary table:

		ost << "\\section{The Varieties of degree $" << degree_of_poly
				<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, summary}" << endl;

#if 0
		T->print_table_latex(
				f,
				TRUE /* f_has_callback */,
				polynomial_orbits_callback_print_function2,
				HPD /* callback_data */,
				TRUE /* f_has_callback */,
				polynomial_orbits_callback_print_function,
				HPD /* callback_data */,
				verbose_level);
#else
		int *coeff;
		int i;

		coeff = NEW_int(HPD->get_nb_monomials());
		//Nb_pts = NEW_int(T->nb_orbits);


		// compute the group of the surface:
		projective_space_with_action *PA;
		int f_semilinear;
		number_theory_domain NT;

		if (NT.is_prime(F->q)) {
			f_semilinear = FALSE;
		}
		else {
			f_semilinear = TRUE;
		}

		PA = NEW_OBJECT(projective_space_with_action);

		if (f_v) {
			cout << "group_theoretic_activity::do_cubic_surface_properties before PA->init" << endl;
		}
		PA->init(
			F, n - 1 /*n*/, f_semilinear,
			TRUE /* f_init_incidence_structure */,
			verbose_level);
		if (f_v) {
			cout << "group_theoretic_activity::do_cubic_surface_properties after PA->init" << endl;
		}






		tally T_nb_pts;
		int h, j, f, l, a;

		T_nb_pts.init(Nb_pts, T->nb_orbits, FALSE, 0);

		for (h = T_nb_pts.nb_types - 1; h >= 0; h--) {

			f = T_nb_pts.type_first[h];
			l = T_nb_pts.type_len[h];
			a = T_nb_pts.data_sorted[f];

			ost << "\\subsection{Objects with " << a << " Points}" << endl;

			ost << "There are " << l << " objects with " << a << " Points: \\\\" << endl;

			int *Idx;
			int len;

			T_nb_pts.get_class_by_value(Idx, len, a, 0 /*verbose_level*/);


			sorting Sorting;

			Sorting.int_vec_heapsort(Idx, l);

			ost << "orbit : rep : go : poly : Pts\\\\" << endl;
			for (j = 0; j < l; j++) {

				//i = T_nb_pts.sorting_perm_inv[f + j];

				i = Idx[j];

				longinteger_object go;
				T->Reps[i].Strong_gens->group_order(go);

				ost << i << " : ";
				lint_vec_print(ost, T->Reps[i].data, T->Reps[i].sz);
				ost << " : ";
				ost << go;

				ost << " : ";

				HPD->unrank_coeff_vector(coeff, T->Reps[i].data[0]);

				int nb_pts;

				nb_pts = Nb_pts[i];

				//ost << nb_pts;
				//ost << " : ";

				ost << T->Reps[i].data[0] << "=$";
				HPD->print_equation_tex(ost, coeff);
				//int_vec_print(f, coeff, HPD->get_nb_monomials());
				//cout << " = ";
				//HPD->print_equation_str(ost, coeff);

				//f << " & ";
				//Reps[i].Strong_gens->print_generators_tex(f);
				ost << "$";

				ost << " : ";

				int u;
				long int *set;
				strong_generators *Sg;

				set = NEW_lint(nb_pts);
				for (u = 0; u < nb_pts; u++) {
					set[u] = Points[i][u];
				}

				for (u = 0; u < nb_pts; u++) {
					ost << set[u];
					if (u < nb_pts - 1) {
						ost << ",";
					}
				}

				PA->compute_group_of_set(set, nb_pts,
						Sg,
						verbose_level);

				ost << " : go=";
				longinteger_object go1;
				Sg->group_order(go1);
				ost << go1;
				ost << "\\\\" << endl;

				FREE_lint(set);
			}

			FREE_int(Idx);

		}
		FREE_OBJECT(PA);

#endif

		FREE_int(coeff);



		L.foot(ost);

	}
	file_io Fio;

	cout << "Written file " << fname_report << " of size " << Fio.file_size(fname_report) << endl;
	if (f_v) {
		cout << "orbits_on_polynomials::report done" << endl;
	}

}

void orbits_on_polynomials::report_detailed_list(std::ostream &ost,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "orbits_on_polynomials::report_detailed_list" << endl;
	}
	// detailed listing:



	tally T1;

	T1.init(Nb_pts, T->nb_orbits, FALSE, 0);
	ost << "Distribution of the number of points: $";
	T1.print_naked_tex(ost, TRUE);
	ost << "$\\\\" << endl;

	ost << "\\section{The Varieties of degree $" << degree_of_poly
			<< "$ in $PG(" << n - 1 << ", " << F->q << ")$, "
					"detailed listing}" << endl;
	{
		int fst, l, a, r;
		longinteger_object go, go1;
		longinteger_domain D;
		int *coeff;
		int *line_type;
		long int *Pts;
		int nb_pts;
		int *Kernel;
		int *v;
		int i;
		//int h, pt, orbit_idx;

		A->group_order(go);
		Pts = NEW_lint(HPD->get_P()->N_points);
		coeff = NEW_int(HPD->get_nb_monomials());
		line_type = NEW_int(HPD->get_P()->N_lines);
		Kernel = NEW_int(HPD->get_nb_monomials() * HPD->get_nb_monomials());
		v = NEW_int(n);

		for (i = 0; i < Sch->nb_orbits; i++) {
			ost << "\\subsection*{Orbit " << i << " / "
					<< Sch->nb_orbits << "}" << endl;
			fst = Sch->orbit_first[i];
			l = Sch->orbit_len[i];

			D.integral_division_by_int(go, l, go1, r);
			a = Sch->orbit[fst];
			HPD->unrank_coeff_vector(coeff, a);


			vector<long int> Points;

			HPD->enumerate_points(coeff, Points, verbose_level);

			nb_pts = Points.size();
			Pts = NEW_lint(nb_pts);
			for (int u = 0; u < nb_pts; u++) {
				Pts[u] = Points[u];
			}

			ost << "stab order " << go1 << "\\\\" << endl;
			ost << "orbit length = " << l << "\\\\" << endl;
			ost << "orbit rep = " << a << "\\\\" << endl;
			ost << "number of points = " << nb_pts << "\\\\" << endl;

			ost << "$";
			int_vec_print(ost, coeff, HPD->get_nb_monomials());
			ost << " = ";
			HPD->print_equation(ost, coeff);
			ost << "$\\\\" << endl;


			cout << "We found " << nb_pts << " points in the variety" << endl;
			cout << "They are : ";
			lint_vec_print(cout, Pts, nb_pts);
			cout << endl;
			HPD->get_P()->print_set_numerical(cout, Pts, nb_pts);

			F->display_table_of_projective_points(
					ost, Pts, nb_pts, n);

			HPD->get_P()->line_intersection_type(Pts, nb_pts,
					line_type, 0 /* verbose_level */);

			ost << "The line type is: ";

			stringstream sstr;
			int_vec_print_classified_str(sstr,
					line_type, HPD->get_P()->N_lines, TRUE /* f_backwards*/);
			string s = sstr.str();
			ost << "$" << s << "$\\\\" << endl;
			//int_vec_print_classified(line_type, HPD->P->N_lines);
			//cout << "after int_vec_print_classified" << endl;

			ost << "The stabilizer is generated by:" << endl;
			T->Reps[i].Strong_gens->print_generators_tex(ost);
		} // next i

		FREE_lint(Pts);
		FREE_int(coeff);
		FREE_int(line_type);
		FREE_int(Kernel);
		FREE_int(v);
		}

	if (f_v) {
		cout << "orbits_on_polynomials::report_detailed_list done" << endl;
	}
}

}}
