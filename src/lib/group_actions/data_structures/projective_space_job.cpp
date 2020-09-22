/*
 * projective_space_job.cpp
 *
 *  Created on: May 26, 2020
 *      Author: betten
 */





#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

projective_space_job::projective_space_job()
{
	t0 = 0;
	F = NULL;
	PA = NULL;
	back_end_counter = 0;
	Descr = NULL;

	f_homogeneous_polynomial_domain_has_been_allocated = FALSE;
	HPD = NULL;

	intersect_with_set_from_file_set_has_beed_read = FALSE;
	intersect_with_set_from_file_set = NULL;
	intersect_with_set_from_file_set_size = 0;

	t_lines = NULL;
	nb_t_lines = 0;
	u_lines = NULL;
	nb_u_lines = 0;
}

void projective_space_job::perform_job(projective_space_job_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;
	os_interface Os;

	if (f_v) {
		cout << "projective_space_job::perform_job" << endl;
	}

	projective_space_job::Descr = Descr;

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(Descr->q, Descr->poly, 0);


	int nb_objects_to_test;
	int input_idx;
	int f_semilinear;
	int f_init_incidence_structure = TRUE;

	if (NT.is_prime(Descr->q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	PA = NEW_OBJECT(projective_space_with_action);

	PA->init(
		F, Descr->n, f_semilinear,
		f_init_incidence_structure,
		verbose_level);


	nb_objects_to_test = Descr->Data->count_number_of_objects_to_test(
		verbose_level - 1);

	cout << "nb_objects_to_test = " << nb_objects_to_test << endl;

	t0 = Os.os_ticks();

	file_io Fio;
	string fname_out_txt;
	string fname_out_tex;

	fname_out_txt.assign(Descr->fname_base_out);
	fname_out_txt.append(".txt");
	fname_out_tex.assign(Descr->fname_base_out);
	fname_out_tex.append(".tex");

	{
		ofstream fp(fname_out_txt);
		ofstream fp_tex(fname_out_tex);

		latex_interface L;

		L.head_easy(fp_tex);

		for (input_idx = 0; input_idx < Descr->Data->nb_inputs; input_idx++) {
			cout << "input " << input_idx << " / " << Descr->Data->nb_inputs
				<< " is:" << endl;


			if (Descr->Data->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
				cout << "input set of points "
					<< Descr->Data->input_string[input_idx] << ":" << endl;


				object_in_projective_space *OiP;
				string dummy;

				dummy.assign("command_line");
				OiP = PA->create_object_from_string(t_PTS,
						dummy, Descr->n,
						Descr->Data->input_string[input_idx], verbose_level);
				back_end(input_idx,
						OiP,
						fp,
						fp_tex,
						verbose_level);
				FREE_OBJECT(OiP);

			}
			else if (Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {
				cout << "input set of points from file "
					<< Descr->Data->input_string[input_idx] << ":" << endl;

				long int *the_set;
				int set_size;

				Fio.read_set_from_file(Descr->Data->input_string[input_idx],
					the_set, set_size, verbose_level);

				object_in_projective_space *OiP;
				OiP = PA->create_object_from_int_vec(t_PTS,
						Descr->Data->input_string[input_idx], Descr->n,
						the_set, set_size, verbose_level);

				back_end(input_idx,
						OiP,
						fp,
						fp_tex,
						verbose_level);
				FREE_OBJECT(OiP);

			}
			else if (Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS ||
					Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES ||
					Descr->Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS ||
					Descr->Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				cout << "input from file " << Descr->Data->input_string[input_idx]
					<< ":" << endl;

				set_of_sets *SoS;

				SoS = NEW_OBJECT(set_of_sets);

				cout << "Reading the file " << Descr->Data->input_string[input_idx] << endl;
				SoS->init_from_file(
						PA->P->N_points /* underlying_set_size */,
						Descr->Data->input_string[input_idx], verbose_level);
				cout << "Read the file " << Descr->Data->input_string[input_idx] << endl;

				int h;


				// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
				long int *Spread_table;
				int nb_spreads;
				int spread_size;

				if (Descr->Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
					cout << "Reading spread table from file "
						<< Descr->Data->input_string2[input_idx] << endl;
					Fio.lint_matrix_read_csv(Descr->Data->input_string2[input_idx],
							Spread_table, nb_spreads, spread_size,
							0 /* verbose_level */);
					cout << "Reading spread table from file "
							<< Descr->Data->input_string2[input_idx] << " done" << endl;
					cout << "The spread table contains " << nb_spreads
							<< " spreads" << endl;
					}

				cout << "processing " << SoS->nb_sets << " objects" << endl;

				for (h = 0; h < SoS->nb_sets; h++) {


					long int *the_set_in;
					int set_size_in;
					object_in_projective_space *OiP;

					OiP = NEW_OBJECT(object_in_projective_space);

					set_size_in = SoS->Set_size[h];
					the_set_in = SoS->Sets[h];

					cout << "The input set " << h << " / " << SoS->nb_sets
						<< " has size " << set_size_in << ":" << endl;

#if 0
					if (f_vv || ((h % 1024) == 0)) {
						cout << "The input set " << h << " / " << SoS->nb_sets
							<< " has size " << set_size_in << ":" << endl;
						}

					if (f_vvv) {
						cout << "The input set is:" << endl;
						int_vec_print(cout, the_set_in, set_size_in);
						cout << endl;
						}
#endif

					if (Descr->Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_POINTS) {
						OiP->init_point_set(PA->P, the_set_in, set_size_in,
								0 /* verbose_level*/);
						}
					else if (Descr->Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_LINES) {
						OiP->init_line_set(PA->P, the_set_in, set_size_in,
								0 /* verbose_level*/);
						}
					else if (Descr->Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_PACKINGS) {
						OiP->init_packing_from_set(PA->P,
								the_set_in, set_size_in, verbose_level);
						}
					else if (Descr->Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
						OiP->init_packing_from_spread_table(PA->P, the_set_in,
							Spread_table, nb_spreads, spread_size,
							verbose_level);
						}
					else {
						cout << "unknown type" << endl;
						exit(1);
						}

					back_end(input_idx,
							OiP,
							fp,
							fp_tex,
							verbose_level);
					FREE_OBJECT(OiP);

				}
			}
			else {
				cout << "unknown type of input object" << endl;
				exit(1);
			}

		} // next input_idx

		L.foot(fp_tex);
		fp << -1 << endl;
	}
	cout << "Written file " << fname_out_txt << " of size "
			<< Fio.file_size(fname_out_txt) << endl;

	cout << "Written file " << fname_out_tex << " of size "
			<< Fio.file_size(fname_out_tex) << endl;
}

void projective_space_job::back_end(int input_idx,
		object_in_projective_space *OiP,
		ostream &fp,
		ostream &fp_tex,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int *the_set_out = NULL;
	int set_size_out = 0;

	if (f_v) {
		cout << "projective_space_job::back_end" << endl;
	}
	perform_job_for_one_set(back_end_counter,
			OiP,
			the_set_out, set_size_out,
			fp_tex,
			verbose_level);

	fp << set_size_out;
	for (int i = 0; i < set_size_out; i++) {
		fp << " " << the_set_out[i];
	}
	fp << endl;

	back_end_counter++;

	if (the_set_out) {
		FREE_lint(the_set_out);
	}
	if (f_v) {
		cout << "projective_space_job::back_end done" << endl;
	}

}

void projective_space_job::perform_job_for_one_set(
	int back_end_counter,
	object_in_projective_space *OiP,
	long int *&the_set_out,
	int &set_size_out,
	ostream &fp_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	long int *the_set_in;
	int set_size_in;

	if (f_v) {
		cout << "projective_space_job::perform_job_for_one_set" << endl;
	}
	the_set_in = OiP->set;
	set_size_in = OiP->sz;

	if (Descr->f_embed) {
		if (f_v) {
			cout << "perform_job_for_one_set f_embed" << endl;
		}
		if (Descr->f_orthogonal) {
			F->do_embed_orthogonal(Descr->orthogonal_epsilon, Descr->n,
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
		}
		else {
			F->do_embed_points(Descr->n,
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
		}
	}
	else if (Descr->f_cone_over) {
		if (f_v) {
			cout << "perform_job_for_one_set f_cone_over" << endl;
		}
		F->do_cone_over(Descr->n,
			the_set_in, set_size_in, the_set_out, set_size_out,
			verbose_level - 1);
	}
	else if (Descr->f_andre) {
		if (f_v) {
			cout << "perform_job_for_one_set f_andre" << endl;
		}
		if (!Descr->f_Q) {
			cout << "please use option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Descr->Q, Descr->poly_Q, 0);

		FQ->do_andre(F,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level - 1);

		FREE_OBJECT(FQ);

	}
	else if (Descr->f_print) {
		if (f_v) {
			cout << "perform_job_for_one_set f_print" << endl;
		}
		if (Descr->f_lines_in_PG) {
			F->do_print_lines_in_PG(Descr->n,
				the_set_in, set_size_in);
		}
		else if (Descr->f_points_in_PG) {
			F->do_print_points_in_PG(Descr->n,
				the_set_in, set_size_in);
		}
		else if (Descr->f_points_on_grassmannian) {
			F->do_print_points_on_grassmannian(Descr->n, Descr->points_on_grassmannian_k,
				the_set_in, set_size_in);
		}
		else if (Descr->f_orthogonal) {
			F->do_print_points_in_orthogonal_space(Descr->orthogonal_epsilon, Descr->n,
				the_set_in, set_size_in, verbose_level);
		}
		else if (Descr->f_homogeneous_polynomials_LEX) {
			if (!f_homogeneous_polynomial_domain_has_been_allocated) {
				HPD = NEW_OBJECT(homogeneous_polynomial_domain);

				HPD->init(F, Descr->n + 1, Descr->homogeneous_polynomials_degree,
					FALSE /* f_init_incidence_structure */,
					t_LEX,
					verbose_level);
				f_homogeneous_polynomial_domain_has_been_allocated = TRUE;
			}
			fp_tex << back_end_counter << ": $";
			HPD->print_equation_lint(fp_tex, the_set_in);
			fp_tex << "$\\\\" << endl;

		}
		else if (Descr->f_homogeneous_polynomials_PART) {
			if (!f_homogeneous_polynomial_domain_has_been_allocated) {
				HPD = NEW_OBJECT(homogeneous_polynomial_domain);

				HPD->init(F, Descr->n + 1, Descr->homogeneous_polynomials_degree,
					FALSE /* f_init_incidence_structure */,
					t_PART,
					verbose_level);
				f_homogeneous_polynomial_domain_has_been_allocated = TRUE;
			}
			fp_tex << back_end_counter << ": $";
			HPD->print_equation_lint(fp_tex, the_set_in);
			fp_tex << "$\\\\" << endl;

		}
	}
	else if (Descr->f_line_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_line_type" << endl;
		}
#if 0
		F->do_line_type(n,
			the_set_in, set_size_in,
			f_show, verbose_level);
#endif
		long int N_lines;
		N_lines = PA->P->nb_rk_k_subspaces_as_lint(2);

		int *type;
		type = NEW_int(N_lines);
		for (int i = 0; i < N_lines; i++) {
			vector<int> point_indices;
			PA->P->line_intersection(i,
					the_set_in, set_size_in,
					point_indices,
					0 /*verbose_level*/);
			type[i] = point_indices.size();
		}
		tally C;

		C.init(type, N_lines, FALSE, 0);

		fp_tex << back_end_counter << ": ";
		C.print_file_tex(fp_tex, TRUE);
		fp_tex << "\\\\" << endl;

		cout << "line type:" << endl;
		cout << back_end_counter << ": ";
		C.print_file_tex(cout, TRUE);
		cout << "\\\\" << endl;



		set_of_sets *SoS;
		int *types;
		int nb_types;
		int i;

		SoS = C.get_set_partition_and_types(
				types, nb_types, verbose_level);
		SoS->print_table();
		for (i = 0; i < nb_types; i++) {
			cout << i << " : " << types[i] << endl;
		}
		FREE_int(types);
		FREE_OBJECT(SoS);
		if (f_v) {
			cout << "perform_job_for_one_set f_line_type done" << endl;
		}
	}
	else if (Descr->f_plane_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_plane_type" << endl;
		}

		long int N_planes;
		N_planes = PA->P->nb_rk_k_subspaces_as_lint(3);

		int *type;
		type = NEW_int(N_planes);
		for (int i = 0; i < N_planes; i++) {
			vector<int> point_indices;
			vector<int> point_local_coordinates;
			PA->P->plane_intersection(i,
					the_set_in, set_size_in,
					point_indices,
					point_local_coordinates,
					verbose_level);
			type[i] = point_indices.size();
		}
		tally C;

		C.init(type, N_planes, FALSE, 0);
		fp_tex << back_end_counter << ": ";
		C.print_file_tex(fp_tex, TRUE);
		fp_tex << "\\\\" << endl;


#if 0
		int *intersection_type;
		int highest_intersection_number;

		F->do_plane_type(n,
			the_set_in, set_size_in,
			intersection_type, highest_intersection_number, verbose_level);

		for (int i = 0; i <= highest_intersection_number; i++) {
			if (intersection_type[i]) {
				cout << i << "^" << intersection_type[i] << " ";
				}
			}
		cout << endl;

		FREE_int(intersection_type);
#endif
	}
	else if (Descr->f_plane_type_failsafe) {
		if (f_v) {
			cout << "perform_job_for_one_set f_plane_type_failsafe" << endl;
		}

		F->do_plane_type_failsafe(Descr->n,
			the_set_in, set_size_in,
			verbose_level);


	}
	else if (Descr->f_conic_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_conic_type" << endl;
		}


		if (Descr->n > 2) {

			projective_space *P2;

			P2 = NEW_OBJECT(projective_space);
			P2->init(2, F,
					FALSE /* f_init_incidence_structure */,
					verbose_level);

			cout << "conic_type n > 2:" << endl;
			vector<int> plane_ranks;
			int s = 5;

			cout << "conic_type before PA->P->find_planes_"
					"which_intersect_in_at_least_s_points" << endl;
			PA->P->find_planes_which_intersect_in_at_least_s_points(
					the_set_in, set_size_in,
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
			the_set_out = NEW_lint(plane_ranks.size());
			set_size_out = plane_ranks.size();
			for (int i = 0; i < len; i++) {
				the_set_out[i] = plane_ranks[i];
			}

			cout << "we will compute the non-degenerate conic "
					"type of all these planes" << endl;

			int *type;
			type = NEW_int(len);
			int_vec_zero(type, len);
			for (int i = 0; i < len; i++) {
				vector<int> point_indices;
				vector<int> point_local_coordinates;
				PA->P->plane_intersection(plane_ranks[i],
						the_set_in, set_size_in,
						point_indices,
						point_local_coordinates,
						verbose_level - 2);

				long int *pts;
				int nb_pts;

				nb_pts = point_local_coordinates.size();
				pts = NEW_lint(nb_pts);
				for (int j = 0; j < nb_pts; j++) {
					pts[j] = point_local_coordinates[j];
				}
				cout << "plane " << i << " is " << plane_ranks[i]
					<< " has " << nb_pts << " points, in local "
							"coordinates they are ";
				lint_vec_print(cout, pts, nb_pts);
				cout << endl;

				long int **Pts_on_conic;
				int *nb_pts_on_conic;
				int len1;
				P2->conic_type(
						pts, nb_pts,
						Pts_on_conic, nb_pts_on_conic, len1,
						verbose_level);
				for (int j = 0; j < len1; j++) {
					if (nb_pts_on_conic[j] == Descr->q + 1) {
						type[i]++;
					}
				}
				for (int j = 0; j < len1; j++) {
					FREE_lint(Pts_on_conic[j]);
				}
				FREE_plint(Pts_on_conic);
				FREE_int(nb_pts_on_conic);
				FREE_lint(pts);
			}




			tally C;

			C.init(type, len, FALSE, 0);
			fp_tex << back_end_counter << ": ";
			C.print_file_tex(fp_tex, TRUE);
			fp_tex << "\\\\" << endl;

			FREE_OBJECT(P2);
		}
		else if (Descr->n == 2) {
			cout << "conic_type n == 2:" << endl;
			int *intersection_type;
			int highest_intersection_number;

			F->do_conic_type(Descr->n, Descr->f_randomized, Descr->nb_times,
				the_set_in, set_size_in,
				intersection_type, highest_intersection_number,
				verbose_level);


			for (int i = 0; i <= highest_intersection_number; i++) {
				if (intersection_type[i]) {
					cout << i << "^" << intersection_type[i] << " ";
					}
				}
			cout << endl;

			FREE_int(intersection_type);
		}
		else {
			cout << "conic type needs n >= 2" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "perform_job_for_one_set f_conic_type done" << endl;
		}
	}
	else if (Descr->f_hyperplane_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_hyperplane_type" << endl;
		}
		F->do_m_subspace_type(Descr->n, Descr->n - 1,
			the_set_in, set_size_in,
			Descr->f_show, verbose_level);
	}
	else if (Descr->f_bsf3) {
		if (f_v) {
			cout << "perform_job_for_one_set f_bsf3" << endl;
		}
		F->do_blocking_set_family_3(Descr->n,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
	}
	else if (Descr->f_test_diagonals) {
		if (f_v) {
			cout << "perform_job_for_one_set f_test_diagonals" << endl;
		}
		F->do_test_diagonal_line(Descr->n,
			the_set_in, set_size_in,
			Descr->test_diagonals_fname,
			verbose_level);
	}
	else if (Descr->f_klein) {
		if (f_v) {
			cout << "perform_job_for_one_set f_klein" << endl;
		}
		F->do_Klein_correspondence(Descr->n,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
	}
	else if (Descr->f_draw_points_in_plane) {
		if (f_v) {
			cout << "perform_job_for_one_set f_draw_points_in_plane" << endl;
		}
		F->do_draw_points_in_plane(
			the_set_in, set_size_in,
			Descr->draw_points_in_plane_fname_base, Descr->f_point_labels,
			Descr->f_embedded, Descr->f_sideways,
			verbose_level);
	}
	else if (Descr->f_canonical_form) {
		if (f_v) {
			cout << "perform_job_for_one_set f_canonical_form" << endl;
		}
		int f_semilinear = TRUE;
		number_theory_domain NT;
		if (NT.is_prime(F->q)) {
			f_semilinear = FALSE;
		}
		do_canonical_form(
			the_set_in, set_size_in,
			f_semilinear, Descr->canonical_form_fname_base,
			verbose_level);
	}
	else if (Descr->f_ideal_LEX) {
		if (f_v) {
			cout << "perform_job_for_one_set f_ideal" << endl;
		}
		F->do_ideal(Descr->n,
			the_set_in, set_size_in, Descr->ideal_degree,
			the_set_out, set_size_out,
			t_LEX,
			verbose_level);
	}
	else if (Descr->f_ideal_PART) {
		if (f_v) {
			cout << "perform_job_for_one_set f_ideal" << endl;
		}
		F->do_ideal(Descr->n,
			the_set_in, set_size_in, Descr->ideal_degree,
			the_set_out, set_size_out,
			t_PART,
			verbose_level);
	}
	else if (Descr->f_intersect_with_set_from_file) {
		if (f_v) {
			cout << "perform_job_for_one_set f_intersect_with_set_from_file" << endl;
		}
		if (!intersect_with_set_from_file_set_has_beed_read) {
			file_io Fio;
			sorting Sorting;

			Fio.read_set_from_file(Descr->intersect_with_set_from_file_fname,
					intersect_with_set_from_file_set,
					intersect_with_set_from_file_set_size,
					verbose_level);
			Sorting.lint_vec_heapsort(intersect_with_set_from_file_set,
					intersect_with_set_from_file_set_size);

			intersect_with_set_from_file_set_has_beed_read = TRUE;
		}

		if (intersect_with_set_from_file_set_has_beed_read) {
			sorting Sorting;

			cout << "before intersecting the sets of size " << set_size_in
					<< " and " << intersect_with_set_from_file_set_size
					<< ":" << endl;

			the_set_out = NEW_lint(set_size_in);
			Sorting.lint_vec_intersect_sorted_vectors(the_set_in, set_size_in,
					intersect_with_set_from_file_set,
					intersect_with_set_from_file_set_size,
					the_set_out, set_size_out);

			cout << "the intersection has size " << set_size_out << endl;
		}
	}
	else if (Descr->f_arc_with_given_set_as_s_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_given_set_of_s_lines_diophant(
				the_set_in /*one_lines*/, set_size_in /* nb_one_lines */,
				Descr->arc_size /*target_sz*/, Descr->arc_d /* target_d */, Descr->arc_d_low, Descr->arc_s /* target_s */,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (f_vv) {
			D->print_tight();
		}

		if (f_save_system) {
			char str[1000];
			string fname_system;

			sprintf(str, "system_%d.diophant", back_end_counter);
			fname_system.assign(str);
			//sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
			}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
			}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
			}
		char fname_solutions[1000];

		sprintf(fname_solutions, "system_%d.solutions", back_end_counter);

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
					}
				fp << endl;
				}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);


	}
	else if (Descr->f_arc_with_two_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;


		lint_vec_scan(Descr->t_lines_string, t_lines, nb_t_lines);

		cout << "The t-lines, t=" << Descr->arc_t << " are ";
		lint_vec_print(cout, t_lines, nb_t_lines);
		cout << endl;


		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_two_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, Descr->arc_s,
				t_lines, nb_t_lines, Descr->arc_t,
				Descr->arc_size /*target_sz*/, Descr->arc_d, Descr->arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (f_vv) {
			D->print_tight();
		}

		if (f_save_system) {
			char str[1000];
			string fname_system;

			sprintf(str, "system_%d.diophant", back_end_counter);
			fname_system.assign(str);

			//sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
			}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
			}
		char fname_solutions[1000];

		sprintf(fname_solutions, "system_%d.solutions", back_end_counter);

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
					}
				fp << endl;
				}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);


	}
	else if (Descr->f_arc_with_three_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;


		lint_vec_scan(Descr->t_lines_string, t_lines, nb_t_lines);
		lint_vec_scan(Descr->u_lines_string, u_lines, nb_u_lines);
		//lint_vec_print(cout, t_lines, nb_t_lines);
		//cout << endl;

		cout << "The t-lines, t=" << Descr->arc_t << " are ";
		lint_vec_print(cout, t_lines, nb_t_lines);
		cout << endl;
		cout << "The u-lines, u=" << Descr->arc_u << " are ";
		lint_vec_print(cout, u_lines, nb_u_lines);
		cout << endl;


		PA->P->arc_with_three_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, Descr->arc_s,
				t_lines, nb_t_lines, Descr->arc_t,
				u_lines, nb_u_lines, Descr->arc_u,
				Descr->arc_size /*target_sz*/, Descr->arc_d, Descr->arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (f_vv) {
			D->print_tight();
		}
		if (f_save_system) {
			char str[1000];
			string fname_system;

			sprintf(str, "system_%d.diophant", back_end_counter);
			fname_system.assign(str);

			//sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
			}

		long int nb_backtrack_nodes;
		long int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, INT_MAX, verbose_level);

		if (f_v) {
			cout << "before D->get_solutions" << endl;
			}
		D->get_solutions(Sol, nb_sol, verbose_level);
		if (f_v) {
			cout << "after D->get_solutions, nb_sol=" << nb_sol << endl;
			}
		char fname_solutions[1000];

		sprintf(fname_solutions, "system_%d.solutions", back_end_counter);

		{
			ofstream fp(fname_solutions);
			int i, j, a;

			for (i = 0; i < nb_sol; i++) {
				fp << D->sum;
				for (j = 0; j < D->sum; j++) {
					a = Sol[i * D->sum + j];
					fp << " " << a;
					}
				fp << endl;
				}
			fp << -1 << " " << nb_sol << endl;
		}
		file_io Fio;

		cout << "Written file " << fname_solutions << " of size "
				<< Fio.file_size(fname_solutions) << endl;
		FREE_lint(Sol);


	}
	else if (Descr->f_dualize_hyperplanes_to_points) {
		if (f_v) {
			cout << "projective_space_job_description::perform_job_for_one_set f_dualize_hyperplanes_to_points" << endl;
		}
		if (OiP->type != t_LNS) {
			cout << "projective_space_job_description::perform_job_for_one_set OiP->type != t_LNS" << endl;
			exit(1);
		}
		if (OiP->P->n != 2) {
			cout << "projective_space_job_description::perform_job_for_one_set OiP->P->n != 2" << endl;
			exit(1);
		}

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = OiP->P->Polarity_hyperplane_to_point[a];
		}

		// only if n = 2:
		//int *Polarity_point_to_hyperplane; // [N_points]
		//int *Polarity_hyperplane_to_point; // [N_points]

	}
	else if (Descr->f_dualize_points_to_hyperplanes) {
		if (f_v) {
			cout << "projective_space_job_description::perform_job_for_one_set f_dualize_points_to_hyperplanes" << endl;
		}
		if (OiP->type != t_PTS) {
			cout << "projective_space_job_description::perform_job_for_one_set OiP->type != t_PTS" << endl;
			exit(1);
		}
		if (OiP->P->n != 2) {
			cout << "projective_space_job_description::perform_job_for_one_set OiP->P->n != 2" << endl;
			exit(1);
		}

		int i;
		long int a;

		set_size_out = set_size_in;
		the_set_out = NEW_lint(set_size_in);
		for (i = 0; i < set_size_in; i++) {
			a = the_set_in[i];
			the_set_out[i] = OiP->P->Polarity_point_to_hyperplane[a];
		}

	}
	if (f_v) {
		cout << "projective_space_job::perform_job_for_one_set done" << endl;
	}


}


void projective_space_job::do_canonical_form(
	long int *set, int set_size, int f_semilinear,
	const char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	projective_space *P;
	int canonical_pt;

	if (f_v) {
		cout << "projective_space_job::do_canonical_form" << endl;
	}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "projective_space_job::do_canonical_form before P->init" << endl;
	}

	P->init(Descr->n, F,
		TRUE /* f_init_incidence_structure */,
		verbose_level);

	if (f_v) {
		cout << "projective_space_job::do_canonical_form after P->init" << endl;
	}

	strong_generators *SG;
	action *A_linear;
	vector_ge *nice_gens;

	A_linear = NEW_OBJECT(action);
	A_linear->init_projective_group(Descr->n + 1, F, f_semilinear,
			TRUE /* f_basis */, FALSE /* f_init_sims */,
			nice_gens,
			verbose_level);

	if (f_v) {
		cout << "projective_space_job::do_canonical_form before "
				"set_stabilizer_in_projective_space" << endl;
	}
	SG = A_linear->set_stabilizer_in_projective_space(
		P,
		set, set_size, canonical_pt, NULL /* canonical_set_or_NULL */,
		FALSE, NULL,
		verbose_level);
	//P->draw_point_set_in_plane(fname_base, set, set_size,
	// TRUE /*f_with_points*/, 0 /* verbose_level */);
	FREE_OBJECT(nice_gens);
	FREE_OBJECT(SG);
	FREE_OBJECT(A_linear);
	FREE_OBJECT(P);

	if (f_v) {
		cout << "projective_space_job::do_canonical_form done" << endl;
	}
}



}}

