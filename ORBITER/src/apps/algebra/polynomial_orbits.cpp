// polynomial_orbits.C
// 
// Anton Betten
// September 10, 2016
//
//
// 
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);

int main(int argc, const char **argv)
{
	t0 = os_ticks();
	
	
	{
	finite_field *F;
	linear_group_description *Descr;
	linear_group *LG;


	int verbose_level = 0;
	int f_linear = FALSE;
	int q;
	int f_degree = FALSE;
	int degree = 0;
	int nb_identify = 0;
	const char *Identify_label[1000];
	const char *Identify_coeff_text[1000];
	int f_stabilizer = FALSE;
	int f_draw_tree = FALSE;
	int f_test_orbit = FALSE;
	int test_orbit_idx = 0;


	int i, j;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-degree") == 0) {
			f_degree = TRUE;
			degree = atoi(argv[++i]);
			cout << "-degree " << degree << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr = new linear_group_description;
			i += Descr->read_arguments(argc - (i - 1),
					argv + i, verbose_level);

			cout << "after Descr->read_arguments" << endl;
			}
		else if (strcmp(argv[i], "-identify") == 0) {
			Identify_label[nb_identify] = argv[++i];
			Identify_coeff_text[nb_identify] = argv[++i];
			cout << "-identify " << Identify_label[nb_identify]
				<< " " << Identify_coeff_text[nb_identify] << endl;
			nb_identify++;
			
			}
		else if (strcmp(argv[i], "-stabilizer") == 0) {
			f_stabilizer = TRUE;
			cout << "-stabilizer " << endl;
			}
		else if (strcmp(argv[i], "-draw_tree") == 0) {
			f_draw_tree = TRUE;
			cout << "-draw_tree " << endl;
			}
		else if (strcmp(argv[i], "-test_orbit") == 0) {
			f_test_orbit = TRUE;
			test_orbit_idx = atoi(argv[++i]);
			cout << "-test_orbit " << test_orbit_idx << endl;
			}
		}

	if (!f_linear) {
		cout << "please use option -linear ..." << endl;
		exit(1);
		}
	if (!f_degree) {
		cout << "please use option -degree <degree>" << endl;
		exit(1);
		}

	int f_v = (verbose_level >= 1);
	

	F = NEW_OBJECT(finite_field);
	F->init(Descr->input_q, 0);

	Descr->F = F;
	q = Descr->input_q;
	


	LG = NEW_OBJECT(linear_group);
	if (f_v) {
		cout << "before LG->init, creating the group" << endl;
		}

	LG->init(Descr, verbose_level);
	
	cout << "after LG->init" << endl;

	action *A;
	matrix_group *M;
	int n;
	
	A = LG->A_linear;

	M = A->G.matrix_grp;
	n = M->n;

	cout << "n = " << n << endl;
	cout << "degree = " << degree << endl;

	cout << "strong generators:" << endl;
	A->Strong_gens->print_generators();
	A->Strong_gens->print_generators_tex();

	homogeneous_polynomial_domain *HPD;

	HPD = NEW_OBJECT(homogeneous_polynomial_domain);

	HPD->init(F, n, degree,
			TRUE /* f_init_incidence_structure */,
			verbose_level);

	action *A2;

	A2 = NEW_OBJECT(action);
	A2->induced_action_on_homogeneous_polynomials(A, 
		HPD, 
		FALSE /* f_induce_action */, NULL, 
		verbose_level);

	cout << "created action A2" << endl;
	A2->print_info();


	int *Elt1;
	int *Elt2;
	int *Elt3;

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);


	
#if 0
	action_on_homogeneous_polynomials *OnHP;

	OnHP = A2->G.OnHP;
#endif
	schreier *Sch;

	//Sch = new schreier;
	//A2->all_point_orbits(*Sch, verbose_level);
	
	cout << "computing orbits:" << endl;
	
	Sch = A->Strong_gens->orbits_on_points_schreier(A2, verbose_level);
	Sch->print_orbit_reps(cout);

	int f, l, a, r;
	longinteger_object go, go1;
	longinteger_domain D;
	int *coeff;
	int *line_type;
	int *Pts;
	int nb_pts;
	int *Kernel;
	int h, pt, orbit_idx;


	Pts = NEW_int(HPD->P->N_points);
	coeff = NEW_int(HPD->nb_monomials);
	line_type = NEW_int(HPD->P->N_lines);
	Kernel = NEW_int(HPD->nb_monomials * HPD->nb_monomials);


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
		int_vec_print(cout, coeff, HPD->nb_monomials);
		cout << " = ";
		HPD->print_equation(cout, coeff);
		cout << endl;


		
		HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);
		cout << "We found " << nb_pts << " points on the curve" << endl;
		cout << "They are : ";
		int_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->P->print_set_numerical(Pts, nb_pts);

		HPD->P->line_intersection_type(Pts, nb_pts,
				line_type, 0 /* verbose_level */);

		cout << "The line type is: ";
		int_vec_print_classified(line_type, HPD->P->N_lines);

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
			sims *Stab;
			longinteger_object stab_go;
			int canonical_pt;

			Stab = set_stabilizer_in_projective_space(
				A, HPD->P, 
				Pts, nb_pts,
				canonical_pt, NULL,
				FALSE, NULL, 
				0 /*verbose_level */);
			Stab->group_order(stab_go);

			cout << "stabilizer order = " << stab_go << endl;
			FREE_OBJECT(Stab);
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

			sprintf(label, "orbit_tree_q%d_d%d_%d", q, degree, i);
			sprintf(fname, "orbit_tree_q%d_d%d_%d_tables.tex", q, degree, i);
			Sch->draw_tree(label,
				i, xmax, ymax,
				f_circletext, rad,
				TRUE /* f_embedded */, FALSE /* f_sideways */, 
				0.3 /* scale */, 1. /* line_width */, 
				FALSE, NULL, 
				0 /*verbose_level */);

			{
			ofstream fp(fname);
			
			latex_head_easy(fp);


			A->Strong_gens->print_generators_tex(fp);

			Sch->print_and_list_orbits_tex(fp);

			Sch->print_tables_latex(fp, TRUE /* f_with_cosetrep */);

			latex_foot(fp);
			}
			cout << "Written file " << fname << " of size "
					<< file_size(fname) << endl;
			
			}
			}



		}

	
	for (h = 0; h < nb_identify; h++) {


		
		cout << "Trying to identify " << h << " / "
			<< nb_identify << " which is "
			<< Identify_label[h] << " : "
			<< Identify_coeff_text[nb_identify] << endl;
		
		int *coeff_list;
		int nb_coeffs;
		int a, b;
	
		int_vec_scan(Identify_coeff_text[nb_identify],
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
		int_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->P->print_set_numerical(Pts, nb_pts);

		char fname[1000];

		sprintf(fname, "point_set_identify_%d", h);
		//draw_point_set_in_plane(HPD, Pts, nb_pts,
		//fname, TRUE /* f_with_points */);

		HPD->P->draw_point_set_in_plane(fname,
				Pts, nb_pts,
				TRUE /* f_with_points */,
				FALSE /* f_point_labels */,
				TRUE /* f_embedded */,
				FALSE /* f_sideways */,
				17000 /* rad */,
				verbose_level);
		
		strong_generators *SG;
		longinteger_object ago;
		Sch->transporter_from_point_to_orbit_rep(pt,
				orbit_idx, Elt1, 5 /* verbose_level */);


		
		SG = NEW_OBJECT(strong_generators);
		SG->init(A);
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
		sims *Stab2;
		int canonical_pt;
		Stab2 = set_stabilizer_in_projective_space(
			A, HPD->P, 
			Pts, nb_pts, canonical_pt, NULL, 
			FALSE, NULL, 
			verbose_level + 3);

		FREE_OBJECT(Stab2);

		sims *Stab;
		action *AR;
		int *perm;
		int *Elt;

		Elt = NEW_int(A->elt_size_in_int);
		perm = NEW_int(nb_pts);
		Stab = SG->create_sims(0 /*verbose_level*/);
		cout << "creating restricted action on the curve:" << endl;
		AR = A->restricted_action(Pts, nb_pts, 0 /* verbose_level */);


		int a1, a2, a3, a4, a6;
		int A6[6];

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
		A6[0] = a1;
		A6[1] = a2;
		A6[2] = a3;
		A6[3] = a4;
		A6[4] = 0;
		A6[5] = a6;

		//c = HPD->P->elliptic_curve_addition(A6,
		//Pts[0], Pts[1], 2 /*verbose_level*/);
		//cout << "c=" << c << endl;

		int *Table;
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

		FREE_OBJECT(SG);
		FREE_int(perm);
		FREE_int(Elt);

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

		int *Pts2;
		int *Pts3;
		int nb_pts2;
		int *coeff2;
		int r, b, orbit_idx;

		Pts2 = NEW_int(HPD->P->N_points);
		Pts3 = NEW_int(HPD->P->N_points);
		coeff2 = NEW_int(HPD->nb_monomials);
		
		HPD->enumerate_points(coeff, Pts, nb_pts, verbose_level);
		cout << "We found " << nb_pts << " points on the curve" << endl;

		int_vec_heapsort(Pts, nb_pts);

		cout << "They are : ";
		int_vec_print(cout, Pts, nb_pts);
		cout << endl;
		HPD->P->print_set_numerical(Pts, nb_pts);

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

			int_vec_heapsort(Pts2, nb_pts);

			cout << "They are : ";
			int_vec_print(cout, Pts2, nb_pts2);
			cout << endl;
			HPD->P->print_set_numerical(Pts2, nb_pts2);
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
			int_vec_heapsort(Pts3, nb_pts);

			cout << "after apply : ";
			int_vec_print(cout, Pts3, nb_pts);
			cout << endl;

		
			if (int_vec_compare(Pts3, Pts2, nb_pts)) {
				cout << "The sets are different" << endl;
				exit(1);
				}

			cout << "The element maps the orbit representative "
					"to the new set, which is good" << endl;
			}

		FREE_int(Pts2);
		FREE_int(Pts3);
		FREE_int(coeff2);
		}


	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);

	FREE_int(Pts);
	FREE_int(coeff);
	FREE_int(line_type);
	FREE_int(Kernel);
	
	}
}


