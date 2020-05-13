/*
 * intersection.cpp
 *
 *  Created on: Apr 29, 2019
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;


using namespace orbiter;
using namespace orbiter::top_level;


// global data:

int t0; // the system time when the program started

int main(int argc, const char **argv);

int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_description = FALSE;
	surface_create_description *Descr;
	int nb_transform = 0;
	const char *transform_coeffs[1000];
	int f_inverse_transform[1000];
	int f_linear = FALSE;
	linear_group_description *Descr_group;
	os_interface Os;

	t0 = Os.os_ticks();


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-description") == 0) {
			f_description = TRUE;
			Descr = NEW_OBJECT(surface_create_description);
			i += Descr->read_arguments(argc - (i - 1), argv + i,
					verbose_level) - 1;

			cout << "-description" << endl;
			}
		else if (strcmp(argv[i], "-linear") == 0) {
			f_linear = TRUE;
			Descr_group = NEW_OBJECT(linear_group_description);
			i += Descr_group->read_arguments(argc - (i + 1),
				argv + i + 1, verbose_level);

			cout << "-linear" << endl;
			}
		else if (strcmp(argv[i], "-transform") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = FALSE;
			cout << "-transform " << transform_coeffs[nb_transform] << endl;
			nb_transform++;
			}
		else if (strcmp(argv[i], "-transform_inverse") == 0) {
			transform_coeffs[nb_transform] = argv[++i];
			f_inverse_transform[nb_transform] = TRUE;
			cout << "-transform_inverse "
					<< transform_coeffs[nb_transform] << endl;
			nb_transform++;
			}
		}
	if (!f_description) {
		cout << "please use option -description ... to enter a "
				"description of the surface" << endl;
		exit(1);
		}
	if (!f_linear) {
		cout << "please use option -linear ... to enter a "
				"description of the orthogonal group" << endl;
		exit(1);
		}


	//int j;
	int f_v = (verbose_level >= 1);


	int q;
	int f_semilinear;
	finite_field *F;
	surface_domain *Surf;
	surface_with_action *Surf_A;
	number_theory_domain NT;
	sorting Sorting;
	file_io Fio;

	q = Descr->get_q();
	cout << "q=" << q << endl;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
		}
	else {
		f_semilinear = TRUE;
		}


	F = NEW_OBJECT(finite_field);
	F->init(q, 0);


	if (f_v) {
		cout << "create_surface_main before Surf->init" << endl;
		}
	Surf = NEW_OBJECT(surface_domain);
	Surf->init(F, verbose_level - 1);
	if (f_v) {
		cout << "create_surface_main after Surf->init" << endl;
		}


	// create the other group:
	linear_group *LG;

	LG = NEW_OBJECT(linear_group);


	//F = NEW_OBJECT(finite_field);
	//F->init(Descr->input_q, 0);

	Descr_group->F = F;


	if (f_v) {
		cout << "linear_group before LG->init, "
				"creating the group" << endl;
		}

	LG->init(Descr_group, verbose_level - 1);

	if (f_v) {
		cout << "linear_group after LG->init" << endl;
		}

	action *A;

	A = LG->A2;

	cout << "created group " << LG->prefix << endl;




	Surf_A = NEW_OBJECT(surface_with_action);




	if (f_v) {
		cout << "create_surface_main before Surf_A->init" << endl;
		}
	Surf_A->init(Surf, LG, verbose_level);
	if (f_v) {
		cout << "create_surface_main after Surf_A->init" << endl;
		}


	// create the surface:
	surface_create *SC;
	SC = NEW_OBJECT(surface_create);

	cout << "before SC->init" << endl;
	SC->init(Descr, Surf_A, verbose_level);
	cout << "after SC->init" << endl;

	if (nb_transform) {
		cout << "create_surface_main "
				"before SC->apply_transformations" << endl;
		SC->apply_transformations(transform_coeffs,
				f_inverse_transform, nb_transform, verbose_level);
		cout << "create_surface_main "
				"after SC->apply_transformations" << endl;
		}


	if (Descr_group->input_q != q) {
		cout << "the group and the surface "
				"must be over the same field" << endl;
		exit(1);
	}

	if (Descr_group->n != 4) {
		cout << "Descr_group->n != 4" << endl;
		exit(1);
	}


	schreier *Sch;
	Sch = NEW_OBJECT(schreier);

	cout << "Strong generators are:" << endl;
	LG->Strong_gens->print_generators_tex(cout);
	cout << "Strong generators as permutations are:" << endl;
	LG->Strong_gens->print_generators_as_permutations();


	if (LG->f_has_nice_gens) {
		cout << "we have nice generators, they are:" << endl;
		LG->nice_gens->print(cout);
		cout << "$$" << endl;
		for (i = 0; i < LG->nice_gens->len; i++) {
			//cout << "Generator " << i << " / " << gens->len
			// << " is:" << endl;
			A->element_print_latex(LG->nice_gens->ith(i), cout);
			if (i < LG->nice_gens->len - 1) {
				cout << ", " << endl;
			}
			if (((i + 1) % 3) == 0 && i < LG->nice_gens->len - 1) {
				cout << "$$" << endl;
				cout << "$$" << endl;
				}
			}
		cout << "$$" << endl;
		LG->nice_gens->print_as_permutation(cout);
	}



	cout << "The group acts on the points of PG(" << Descr_group->n - 1
			<< "," << Descr_group->input_q << ")" << endl;


	// create the quadric:

	cout << "We are now creating the quadric:" << endl;

	int nb_pts;
	long int *Pts;
	int n = 3;
	int epsilon = 1;
	int c1 = 1, c2 = 0, c3 = 0;
	int j;
	int d = n + 1;
	int *v;
	geometry_global Gg;

	nb_pts = Gg.nb_pts_Qepsilon(epsilon, n, q);

	v = NEW_int(d);
	Pts = NEW_lint(nb_pts);

	if (epsilon == -1) {
		F->choose_anisotropic_form(c1, c2, c3, verbose_level);
		if (f_v) {
			cout << "c1=" << c1 << " c2=" << c2 << " c3=" << c3 << endl;
			}
		}
	if (f_v) {
		cout << "orthogonal rank : point : projective rank" << endl;
		}
	for (i = 0; i < nb_pts; i++) {
		F->Q_epsilon_unrank(v, 1, epsilon, n, c1, c2, c3, i, 0 /* verbose_level */);
		F->PG_element_rank_modified(v, 1, d, j);
		Pts[i] = j;
		if (f_v) {
			cout << setw(4) << i << " : ";
			int_vec_print(cout, v, d);
			cout << " : " << setw(5) << j << endl;
			}
		}


	cout << "The quadric is ";
	lint_vec_print(cout, Pts, nb_pts);
	cout << endl;





	cout << "We are now printing the surface:" << endl;


	int coeffs_out[20];
	action *A_big;
	//int *Elt1;
	int *Elt2;

	A_big = SC->Surf_A->A;

	Elt2 = NEW_int(A_big->elt_size_in_int);

	SC->F->init_symbol_for_print("\\omega");

	if (SC->F->e == 1) {
		SC->F->f_print_as_exponentials = FALSE;
	}

	SC->F->PG_element_normalize(SC->coeffs, 1, 20);

	cout << "create_surface_main "
			"We have created the following surface:" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;


	if (SC->f_has_group) {
		for (i = 0; i < SC->Sg->gens->len; i++) {
			cout << "Testing generator " << i << " / "
					<< SC->Sg->gens->len << endl;
			A_big->element_invert(SC->Sg->gens->ith(i),
					Elt2, 0 /*verbose_level*/);



			matrix_group *M;

			M = A_big->G.matrix_grp;
			M->substitute_surface_equation(Elt2,
					SC->coeffs, coeffs_out, SC->Surf,
					verbose_level - 1);


			if (int_vec_compare(SC->coeffs, coeffs_out, 20)) {
				cout << "error, the transformation does not preserve "
						"the equation of the surface" << endl;
				exit(1);
				}
			cout << "Generator " << i << " / " << SC->Sg->gens->len
					<< " is good" << endl;
			}
		}
	else {
		cout << "We do not have information about "
				"the automorphism group" << endl;
		exit(1);
		}


	cout << "We have created the surface " << SC->label_txt << ":" << endl;
	cout << "$$" << endl;
	SC->Surf->print_equation_tex(cout, SC->coeffs);
	cout << endl;
	cout << "$$" << endl;

	if (SC->f_has_group) {
		cout << "The stabilizer is generated by:" << endl;
		SC->Sg->print_generators_tex(cout);
		}


	// create surface object:


	if (!SC->f_has_lines) {
		cout << "The surface " << SC->label_txt
				<< " does not have lines" << endl;
		exit(1);
	}
	cout << "The lines are:" << endl;
	SC->Surf->Gr->print_set_tex(cout, SC->Lines, 27);


	surface_object *SO;

	SO = NEW_OBJECT(surface_object);
	if (f_v) {
		cout << "before SO->init" << endl;
		}
	SO->init(SC->Surf, SC->Lines, SC->coeffs,
			FALSE /*f_find_double_six_and_rearrange_lines */, verbose_level);
	if (f_v) {
		cout << "after SO->init" << endl;
		}

	char fname_points[1000];

	sprintf(fname_points, "surface_%s_points.txt", SC->label_txt);
	Fio.write_set_to_file(fname_points,
			SO->Pts, SO->nb_pts, 0 /*verbose_level*/);
	cout << "Written file " << fname_points << " of size "
			<< Fio.file_size(fname_points) << endl;





	if (!SC->f_has_group) {
		cout << "The surface " << SC->label_txt
				<< " does not have a group" << endl;
		exit(1);
	}

	cout << "creating surface_object_with_action object" << endl;

	surface_object_with_action *SoA;

	SoA = NEW_OBJECT(surface_object_with_action);

	if (SC->f_has_lines) {
		cout << "creating surface using the known lines (which are "
				"arranged with respect to a double six):" << endl;
		SoA->init(SC->Surf_A,
			SC->Lines,
			SC->coeffs,
			SC->Sg,
			FALSE /*f_find_double_six_and_rearrange_lines*/,
			SC->f_has_nice_gens, SC->nice_gens,
		verbose_level);
		}
	else {
		cout << "creating surface from equation only "
				"(no lines):" << endl;
		SoA->init_equation(SC->Surf_A,
			SC->coeffs,
			SC->Sg,
			verbose_level);
		}
	cout << "The surface has been created." << endl;


	cout << "We are now computing the orbit of the quadric "
			"under the big group:" << endl;

	// compute the orbit of the quadric under PGGL(4,q):

	long int *the_set = Pts;
	int set_sz = nb_pts;

	orbit_of_sets *OS;

	OS = NEW_OBJECT(orbit_of_sets);

	OS->init(A_big, A_big, the_set, set_sz,
			A_big->Strong_gens->gens, verbose_level);

	//OS->compute(verbose_level);

	cout << "Found an orbit of length " << OS->used_length << endl;

	long int *Table;
	int orbit_length, set_size;

	cout << "before OS->get_table_of_orbits" << endl;
	OS->get_table_of_orbits_and_hash_values(Table,
			orbit_length, set_size, verbose_level);
	cout << "after OS->get_table_of_orbits" << endl;


	char fname[1000];
	sprintf(fname, "orbit_of_quadric_under_%s_with_hash.csv", LG->prefix);
	cout << "Writing table to file " << fname << endl;
	Fio.lint_matrix_write_csv(fname,
			Table, orbit_length, set_size);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	cout << "before OS->get_table_of_orbits" << endl;
	OS->get_table_of_orbits(Table,
			orbit_length, set_size, verbose_level);
	cout << "after OS->get_table_of_orbits" << endl;

	sprintf(fname, "orbit_of_quadric_under_%s.txt", LG->prefix);
	cout << "Writing table to file " << fname << endl;
	{
		ofstream ost(fname);
		for (i = 0; i < orbit_length; i++) {
			ost << set_size;
			for (int j = 0; j < set_size; j++) {
				ost << " " << Table[i * set_size + j];
			}
			ost << endl;
		}
		ost << -1 << " " << orbit_length << endl;
	}
	//Fio.int_matrix_write_csv(fname,
	//		Table, orbit_length, set_size);
	cout << "Written file " << fname << " of size "
			<< Fio.file_size(fname) << endl;


	cout << "We are now computing the orbits of the stabilizer "
			"of the surface on the set of quadrics:" << endl;



	// compute the orbits on the set of quadrics under the group of the surface:

	action *A_on_sets;
	int *transporter_inv;

	cout << "creating action on sets:" << endl;
	A_on_sets = A->create_induced_action_on_sets(orbit_length /* nb_sets */,
			set_size, Table,
			verbose_level);

	transporter_inv = NEW_int(A->elt_size_in_int);

	schreier *Sch_on_sets;
	int first, a, orbit_idx;
	longinteger_object go;

	SC->Sg->group_order(go);

	cout << "computing orbits of the group of order " << go
			<< " on the set system:" << endl;
	A_on_sets->compute_orbits_on_points(Sch_on_sets,
			SC->Sg->gens, verbose_level);

	cout << "The orbit lengths are:" << endl;
	Sch_on_sets->print_orbit_lengths(cout);

	cout << "The quadric is:" << endl;
	Surf_A->Surf->P->print_set(the_set, set_sz);

	cout << "The orbits are:" << endl;
	//Sch_on_sets->print_and_list_orbits(cout);

	int N_planes;
	int N_lines_in_plane;
	N_planes = Surf_A->Surf->P->nb_rk_k_subspaces_as_lint(3);
	N_lines_in_plane = Surf_A->Surf->P2->nb_rk_k_subspaces_as_lint(2);
	int *plane_rk;
	plane_rk = NEW_int(N_planes);

	{
		ofstream fp("table.tex");
		ofstream fp2("table_planes.tex");
		ofstream fp3("table_planes_detailed.tex");

	for (orbit_idx = 0; orbit_idx < Sch_on_sets->nb_orbits; orbit_idx++) {

		int orbit_length1;

		orbit_length1 = Sch_on_sets->orbit_len[orbit_idx];
		cout << " Orbit " << orbit_idx << " / " << Sch_on_sets->nb_orbits
				<< " : " << Sch_on_sets->orbit_first[orbit_idx]
				<< " : " << Sch_on_sets->orbit_len[orbit_idx];
		cout << " : ";

		fp << orbit_idx;

		first = Sch_on_sets->orbit_first[orbit_idx];
		a = Sch_on_sets->orbit[first + 0];
		cout << a << " : ";
		lint_vec_print(cout, Table + a * set_size, set_size);
		cout << endl;

		OS->coset_rep(a);
		cout << "transporter:" << endl;
		A->element_print_quick(OS->cosetrep, cout);
		//Sch_on_sets->print_and_list_orbit_tex(orbit_idx, ost);

		A->element_invert(OS->cosetrep, transporter_inv, 0);
		cout << "transporter_inv:" << endl;
		A->element_print_quick(transporter_inv, cout);


		sorting Sorting;
		long int *intersection;
		int intersection_size;
		long int *intersection_on_quadric;

		Sorting.vec_intersect(
				Table + a * set_size, set_size,
				SO->Pts, SO->nb_pts,
				intersection, intersection_size);

		cout << "The intersection has size " << intersection_size << " : ";
		lint_vec_print(cout, intersection, intersection_size);
		cout << endl;

		fp << " & " << intersection_size;

		fp << " & " << orbit_length1;


		intersection_on_quadric = NEW_lint(intersection_size);
		A->map_a_set_and_reorder(intersection, intersection_on_quadric,
				intersection_size, transporter_inv, 0 /* verbose_level */);

		cout << "The intersection mapped back to the quadric : ";
		lint_vec_print(cout, intersection_on_quadric, intersection_size);
		cout << endl;

		strong_generators *stab_gens;

		stab_gens = Sch_on_sets->stabilizer_orbit_rep(
				A,
				go,
				orbit_idx, 0 /* verbose_level */);
		cout << "The stabilizer is:" << endl;
		stab_gens->print_generators_tex(cout);

		fp << " & " << stab_gens->group_order_as_lint();


		vector<int> plane_ranks;
		int s = 5;

		cout << "conic_type before PA->P->find_planes_"
				"which_intersect_in_at_least_s_points" << endl;
		Surf_A->Surf->P->find_planes_which_intersect_in_at_least_s_points(
				intersection, intersection_size,
				s,
				plane_ranks,
				verbose_level);
		int len;

		len = plane_ranks.size();
		cout << "we found " << len << " planes which intersect in "
				"at least " << s << " points" << endl;
		cout << "They are: ";
		for (int i = 0; i < len; i++) {
			cout << plane_ranks[i];
			if (i < len - 1) {
				cout << ", ";
			}
		}
		cout << endl;

		fp << " & " << len;

		long int *tri_planes;
		int nb_tri_planes;
		long int *five_planes_tri;
		long int *five_planes_not_tri;
		int nb_five_planes_tri = 0;
		int nb_five_planes_not_tri = 0;
		int idx;

		five_planes_tri = NEW_lint(len);
		five_planes_not_tri = NEW_lint(len);
		tri_planes = SoA->SO->Tritangent_planes;
		nb_tri_planes = SoA->SO->nb_tritangent_planes;
		Sorting.lint_vec_heapsort(tri_planes, nb_tri_planes);


		for (int i = 0; i < len; i++) {
			a = plane_ranks[i];
			if (Sorting.lint_vec_search(tri_planes, nb_tri_planes, a, idx, 0)) {
				five_planes_tri[nb_five_planes_tri++] = a;
			}
			else {
				five_planes_not_tri[nb_five_planes_not_tri++] = a;
			}
		}
		cout << "orbit_idx = " << orbit_idx << " we found "
				<< nb_five_planes_tri << " tritangent planes "
						"which intersect in "
				"at least " << s << " points" << endl;
		cout << "They are: ";
		lint_vec_print(cout, five_planes_tri, nb_five_planes_tri);
		cout << endl;
		cout << "orbit_idx = " << orbit_idx << " we found "
				<< nb_five_planes_not_tri << " non-tritangent planes "
						"which intersect in "
				"at least " << s << " points" << endl;
		cout << "They are: ";
		lint_vec_print(cout, five_planes_not_tri, nb_five_planes_not_tri);
		cout << endl;

		fp << " & " << nb_five_planes_tri;
		fp << " & " << nb_five_planes_not_tri;
		fp << "\\\\" << endl;


		fp2 << orbit_idx;
		fp3 << orbit_idx;

		int *type;
		int *line_type;
		type = NEW_int(N_planes);
		line_type = NEW_int(N_lines_in_plane);


		int e, max_value, plane_rank;

		for (e = 0; e < 2; e++) {
			// e = 0 tritangent planes
			// e = 1: non tritangent planes



			int nb = 0;

			for (int i = 0; i < N_planes; i++) {

				if (e == 0) {
					if (!Sorting.lint_vec_search(tri_planes, nb_tri_planes, i, idx, 0)) {
						continue;
					}
				}
				else {
					if (Sorting.lint_vec_search(tri_planes, nb_tri_planes, i, idx, 0)) {
						continue;
					}
				}
				vector<int> point_indices;
				vector<int> point_local_coordinates;
				Surf_A->Surf->P->plane_intersection(i,
						intersection, intersection_size,
						point_indices,
						point_local_coordinates,
						verbose_level);
				plane_rk[nb] = i;
				type[nb] = point_indices.size();
				nb++;
			}
			classify C;

			fp2 << " & ";
			fp3 << " & ";

			C.init(type, nb, FALSE, 0);
			C.print_file_tex_we_are_in_math_mode(fp2, TRUE);

			max_value = C.get_largest_value();

			int *special_plane_rk;
			int nb_special_planes;

			C.get_class_by_value(special_plane_rk, nb_special_planes, max_value,
					verbose_level);
			for (int i = 0; i < nb_special_planes; i++) {
				special_plane_rk[i] = plane_rk[special_plane_rk[i]];
			}

			fp3 << max_value << " & ";

			fp3 << endl;
			fp3 << "\\begin{array}[t]{c}" << endl;

			for (int i = 0; i < nb_special_planes; i++) {
				plane_rank = special_plane_rk[i];

				vector<int> point_indices;
				vector<int> point_local_coordinates;
				long int *pts_local;
				int nb_pts_local;

				Surf_A->Surf->P->plane_intersection(plane_rank,
						intersection, intersection_size,
							point_indices,
							point_local_coordinates,
							verbose_level);
				nb_pts_local = point_local_coordinates.size();
				pts_local = NEW_lint(nb_pts_local);
				for (int j = 0; j < nb_pts_local; j++) {
					pts_local[j] = point_local_coordinates[j];
				}

				long int **Pts_on_conic;
				int *nb_pts_on_conic;
				int nb_conics;

				Surf_A->Surf->P2->conic_type(
						pts_local, nb_pts_local,
						Pts_on_conic, nb_pts_on_conic, nb_conics,
						verbose_level);
				classify C2;

				C2.init(nb_pts_on_conic, nb_conics, FALSE, 0);

				C2.print_file_tex_we_are_in_math_mode(fp3, TRUE);
				for (int j = 0; j < nb_conics; j++) {
					FREE_lint(Pts_on_conic[j]);
				}
				FREE_plint(Pts_on_conic);
				FREE_int(nb_pts_on_conic);


				fp3 << ":";


				Surf_A->Surf->P2->line_intersection_type_basic(
						pts_local, nb_pts_local, line_type, verbose_level);
				classify C3;

				C3.init(line_type, N_lines_in_plane, FALSE, 0);
				C3.print_file_tex_we_are_in_math_mode(fp3, TRUE);

#if 0
				if (i < nb_special_planes - 1) {
					fp3 << ";";
				}
#endif
				fp3 << "\\\\" << endl;

				// type[N_lines]
			}
			fp3 << "\\end{array}" << endl;

			FREE_int(special_plane_rk);

		} // next e
		fp2 << "\\\\" << endl;
		fp3 << "\\\\" << endl;
		fp3 << "\\hline" << endl;

		FREE_int(type);





		FREE_OBJECT(stab_gens);
		FREE_lint(intersection);
		}
	} // end ofstream

	char fname_orbits[1000];

	sprintf(fname_orbits, "quadric_orbit_reps.txt");

	{
		ofstream ost(fname_orbits);

		for (i = 0; i < Sch_on_sets->nb_orbits; i++) {

			first = Sch_on_sets->orbit_first[i];
			a = Sch_on_sets->orbit[first + 0];
			ost << set_size;
			for (j = 0; j < set_size; j++) {
				ost << " " << Table[a * set_size + j];
			}
			ost << endl;
		}
		ost << -1 << " " << Sch_on_sets->nb_orbits << endl;
	}


	FREE_lint(Table);


	cout << "before FREE_OBJECT(OS)" << endl;
	FREE_OBJECT(OS);
	cout << "after FREE_OBJECT(OS)" << endl;


}

