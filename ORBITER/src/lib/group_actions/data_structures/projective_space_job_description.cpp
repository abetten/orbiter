/*
 * projective_space_job_description.cpp
 *
 *  Created on: Apr 28, 2019
 *      Author: betten
 */




#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {


projective_space_job_description::projective_space_job_description()
{
	t0 = 0;
	F = NULL;
	PA = NULL;
	back_end_counter = 0;

	f_input = FALSE;
	Data = NULL;

	f_fname_base_out = FALSE;
	fname_base_out = NULL;

	f_q = FALSE;
	q = 0;
	f_n = FALSE;
	n = 0;
	f_poly = FALSE;
	poly = NULL;

	f_embed = FALSE;
		// follow up option for f_print:
		//f_orthogonal, orthogonal_epsilon

	f_andre = FALSE;
		// follow up option for f_andre:
		f_Q = FALSE;
		Q = 0;
		f_poly_Q = FALSE;
		poly_Q = NULL;


	f_print = FALSE;
		// follow up option for f_print:
		f_lines_in_PG = FALSE;
		f_points_in_PG = FALSE;
		f_points_on_grassmannian = FALSE;
		points_on_grassmannian_k = 0;
		f_orthogonal = FALSE;
		orthogonal_epsilon = 0;
		f_homogeneous_polynomials = FALSE;
		homogeneous_polynomials_degree = 0;
		f_homogeneous_polynomial_domain_has_been_allocated = FALSE;
		HPD = NULL;


	f_list_group_elements = FALSE;
	f_line_type = FALSE;
	f_plane_type = FALSE;
	f_plane_type_failsafe = FALSE;
	f_conic_type = FALSE;
		// follow up option for f_conic_type:
		f_randomized = FALSE;
		nb_times = 0;

	f_hyperplane_type = FALSE;
	// follow up option for f_hyperplane_type:
		f_show = FALSE;


	f_cone_over = FALSE;

	f_bsf3 = FALSE;
	f_test_diagonals = FALSE;
	test_diagonals_fname = NULL;
	f_klein = FALSE;

	f_draw_points_in_plane = FALSE;
		draw_points_in_plane_fname_base = NULL;
		// follow up option for f_draw_points_in_plane:

		f_point_labels = FALSE;
		f_embedded = FALSE;
		f_sideways = FALSE;

	f_canonical_form = FALSE;
	canonical_form_fname_base = NULL;
	f_ideal = FALSE;
	ideal_degree = 0;

	f_intersect_with_set_from_file = FALSE;
	intersect_with_set_from_file_fname = NULL;
	intersect_with_set_from_file_set_has_beed_read = FALSE;
	intersect_with_set_from_file_set = NULL;
	intersect_with_set_from_file_set_size = 0;

	f_arc_with_given_set_as_s_lines_after_dualizing = FALSE;
	arc_size = 0;
	arc_d = 0;
	arc_d_low = 0;
	arc_s = 0;

	f_arc_with_two_given_sets_of_lines_after_dualizing = FALSE;
	//int arc_size;
	//int arc_d;
	arc_t = 0;
	t_lines_string = NULL;
	t_lines = NULL;
	nb_t_lines = 0;


	f_arc_with_three_given_sets_of_lines_after_dualizing = FALSE;
	arc_u = 0;
	u_lines_string = NULL;
	u_lines = NULL;
	nb_u_lines = 0;


}

projective_space_job_description::~projective_space_job_description()
{

}

void projective_space_job_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "projective_space_job_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "projective_space_job_description::read_arguments_from_string "
				"done" << endl;
	}
}

int projective_space_job_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "projective_space_job_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -q " << q << endl;
			}
		else if (strcmp(argv[i], "-Q") == 0) {
			f_Q = TRUE;
			Q = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -Q " << Q << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -n " << n << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "projective_space_job_description::read_arguments -poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-poly_Q") == 0) {
			f_poly_Q = TRUE;
			poly_Q = argv[++i];
			cout << "projective_space_job_description::read_arguments -poly_Q " << poly_Q << endl;
			}
		else if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			cout << "projective_space_job_description::read_arguments -input" << endl;
			i += Data->read_arguments(argc - i,
				argv + i + 1, verbose_level) + 1;
			cout << "projective_space_job_description::read_arguments finished reading -input" << endl;
			}
		else if (strcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out = argv[++i];
			cout << "projective_space_job_description::read_arguments -fname_base_out " << fname_base_out << endl;
			}
		else if (strcmp(argv[i], "-embed") == 0) {
			f_embed = TRUE;
			cout << "projective_space_job_description::read_arguments -embed" << endl;
			}
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			orthogonal_epsilon = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -orthogonal " << orthogonal_epsilon << endl;
			}
		else if (strcmp(argv[i], "-homogeneous_polynomials") == 0) {
			f_homogeneous_polynomials = TRUE;
			homogeneous_polynomials_degree = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -homogeneous_polynomials " << homogeneous_polynomials_degree << endl;
			}
		else if (strcmp(argv[i], "-andre") == 0) {
			f_andre = TRUE;
			cout << "projective_space_job_description::read_arguments -andre " << endl;
			}
		else if (strcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "projective_space_job_description::read_arguments -print " << endl;
			}
		else if (strcmp(argv[i], "-lines_in_PG") == 0) {
			f_lines_in_PG = TRUE;
			cout << "projective_space_job_description::read_arguments -lines_in_PG " << endl;
			}
		else if (strcmp(argv[i], "-points_in_PG") == 0) {
			f_points_in_PG = TRUE;
			cout << "projective_space_job_description::read_arguments -points_in_PG " << endl;
			}
		else if (strcmp(argv[i], "-points_on_grassmannian") == 0) {
			f_points_on_grassmannian = TRUE;
			points_on_grassmannian_k = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -points_on_grassmannian " << points_on_grassmannian_k << endl;
			}
		else if (strcmp(argv[i], "-list_group_elements") == 0) {
			f_list_group_elements = TRUE;
			cout << "projective_space_job_description::read_arguments -list_group_elements" << endl;
			}
		else if (strcmp(argv[i], "-line_type") == 0) {
			f_line_type = TRUE;
			cout << "projective_space_job_description::read_arguments -line_type" << endl;
			}
		else if (strcmp(argv[i], "-plane_type") == 0) {
			f_plane_type = TRUE;
			cout << "projective_space_job_description::read_arguments -plane_type" << endl;
			}
		else if (strcmp(argv[i], "-plane_type_failsafe") == 0) {
			f_plane_type_failsafe = TRUE;
			cout << "projective_space_job_description::read_arguments -plane_type_failsafe" << endl;
			}
		else if (strcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			cout << "projective_space_job_description::read_arguments -conic_type " << endl;
			}
		else if (strcmp(argv[i], "-randomized") == 0) {
			f_randomized = TRUE;
			nb_times = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -randomized " << nb_times << endl;
			}
		else if (strcmp(argv[i], "-hyperplane_type") == 0) {
			f_hyperplane_type = TRUE;
			cout << "projective_space_job_description::read_arguments -hyperplane_type" << endl;
			}
		else if (strcmp(argv[i], "-show") == 0) {
			f_show = TRUE;
			cout << "projective_space_job_description::read_arguments -show" << endl;
			}
		else if (strcmp(argv[i], "-cone_over") == 0) {
			f_cone_over = TRUE;
			cout << "projective_space_job_description::read_arguments -cone_over" << endl;
			}
		else if (strcmp(argv[i], "-bsf3") == 0) {
			f_bsf3 = TRUE;
			cout << "projective_space_job_description::read_arguments -bsf3" << endl;
			}
		else if (strcmp(argv[i], "-test_diagonals") == 0) {
			f_test_diagonals = TRUE;
			test_diagonals_fname = argv[++i];
			cout << "projective_space_job_description::read_arguments -test_diagonals " << test_diagonals_fname << endl;
			}
		else if (strcmp(argv[i], "-klein") == 0) {
			f_klein = TRUE;
			cout << "projective_space_job_description::read_arguments -klein" << endl;
			}
		else if (strcmp(argv[i], "-draw_points_in_plane") == 0) {
			f_draw_points_in_plane = TRUE;
			draw_points_in_plane_fname_base = argv[++i];
			cout << "projective_space_job_description::read_arguments -draw_points_in_plane" << endl;
			}
		else if (strcmp(argv[i], "-point_labels") == 0) {
			f_point_labels = TRUE;
			cout << "projective_space_job_description::read_arguments -point_labels" << endl;
			}
		else if (strcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			canonical_form_fname_base = argv[++i];
			cout << "projective_space_job_description::read_arguments -canonical_form" << canonical_form_fname_base << endl;
			}
		else if (strcmp(argv[i], "-ideal") == 0) {
			f_ideal = TRUE;
			ideal_degree = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -ideal " << ideal_degree << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "projective_space_job_description::read_arguments -embedded" << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "projective_space_job_description::read_arguments -sideways" << endl;
			}
		else if (strcmp(argv[i], "-intersect_with_set_from_file") == 0) {
			f_intersect_with_set_from_file = TRUE;
			intersect_with_set_from_file_fname = argv[++i];
			cout << "projective_space_job_description::read_arguments -intersect_with_set_from_file " << intersect_with_set_from_file_fname << endl;
		}
		else if (strcmp(argv[i], "-arc_with_given_set_as_s_lines_after_dualizing") == 0) {
			f_arc_with_given_set_as_s_lines_after_dualizing = TRUE;
			arc_size = atoi(argv[++i]);
			arc_d = atoi(argv[++i]);
			arc_d_low = atoi(argv[++i]);
			arc_s = atoi(argv[++i]);
			cout << "projective_space_job_description::read_arguments -arc_with_given_set_as_s_lines_after_dualizing "
					<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << endl;
		}
		else if (strcmp(argv[i], "-arc_with_two_given_sets_of_lines_after_dualizing") == 0) {
			f_arc_with_two_given_sets_of_lines_after_dualizing = TRUE;
			arc_size = atoi(argv[++i]);
			arc_d = atoi(argv[++i]);
			arc_d_low = atoi(argv[++i]);
			arc_s = atoi(argv[++i]);
			arc_t = atoi(argv[++i]);
			t_lines_string = argv[++i];
			int_vec_scan(t_lines_string, t_lines, nb_t_lines);
			cout << "projective_space_job_description::read_arguments -arc_with_two_given_sets_of_lines_after_dualizing src_size="
					<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << " t=" << arc_t << " ";
			int_vec_print(cout, t_lines, nb_t_lines);
			cout << endl;
		}
		else if (strcmp(argv[i], "-arc_with_three_given_sets_of_lines_after_dualizing") == 0) {
			f_arc_with_three_given_sets_of_lines_after_dualizing = TRUE;
			arc_size = atoi(argv[++i]);
			arc_d = atoi(argv[++i]);
			arc_d_low = atoi(argv[++i]);
			arc_s = atoi(argv[++i]);
			arc_t = atoi(argv[++i]);
			t_lines_string = argv[++i];
			arc_u = atoi(argv[++i]);
			u_lines_string = argv[++i];
			int_vec_scan(t_lines_string, t_lines, nb_t_lines);
			int_vec_scan(u_lines_string, u_lines, nb_u_lines);
			cout << "projective_space_job_description::read_arguments -arc_with_three_given_sets_of_lines_after_dualizing "
					<< arc_size << " d=" << arc_d << " d_low=" << arc_d_low << " s=" << arc_s << endl;
			cout << "arc_t = " << arc_t << " t_lines_string = " << t_lines_string << endl;
			cout << "arc_u = " << arc_u << " u_lines_string = " << u_lines_string << endl;
			cout << "The t-lines, t=" << arc_t << " are ";
			int_vec_print(cout, t_lines, nb_t_lines);
			cout << endl;
			cout << "The u-lines, u=" << arc_u << " are ";
			int_vec_print(cout, u_lines, nb_u_lines);
			cout << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "projective_space_job_description::read_arguments -end" << endl;
			break;
		}
		else {
			cout << "projective_space_job_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
			exit(1);
		}
	} // next i
	cout << "projective_space_job_description::read_arguments done" << endl;
	return i;
}

void projective_space_job_description::perform_job(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory_domain NT;

	if (f_v) {
		cout << "projective_space_job_description::perform_job" << endl;
	}

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0);


	int nb_objects_to_test;
	int input_idx;
	int f_semilinear;
	int f_init_incidence_structure = TRUE;

	if (NT.is_prime(q)) {
		f_semilinear = FALSE;
	}
	else {
		f_semilinear = TRUE;
	}

	PA = NEW_OBJECT(projective_space_with_action);

	PA->init(
		F, n, f_semilinear,
		f_init_incidence_structure,
		verbose_level);


	nb_objects_to_test = Data->count_number_of_objects_to_test(
		verbose_level - 1);

	cout << "nb_objects_to_test = " << nb_objects_to_test << endl;

	t0 = os_ticks();

	file_io Fio;
	char fname_out_txt[1000];
	char fname_out_tex[1000];

	sprintf(fname_out_txt, "%s.txt", fname_base_out);
	sprintf(fname_out_tex, "%s.tex", fname_base_out);

	{
		ofstream fp(fname_out_txt);
		ofstream fp_tex(fname_out_tex);

		latex_interface L;

		L.head_easy(fp_tex);

		for (input_idx = 0; input_idx < Data->nb_inputs; input_idx++) {
			cout << "input " << input_idx << " / " << Data->nb_inputs
				<< " is:" << endl;


			if (Data->input_type[input_idx] == INPUT_TYPE_SET_OF_POINTS) {
				cout << "input set of points "
					<< Data->input_string[input_idx] << ":" << endl;

				object_in_projective_space *OiP;
				OiP = PA->create_object_from_string(t_PTS,
						"command_line", n,
						Data->input_string[input_idx], verbose_level);
				back_end(input_idx,
						OiP,
						fp,
						fp_tex,
						verbose_level);
				FREE_OBJECT(OiP);

			}
			else if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINT_SET) {
				cout << "input set of points from file "
					<< Data->input_string[input_idx] << ":" << endl;

				int *the_set;
				int set_size;

				Fio.read_set_from_file(Data->input_string[input_idx],
					the_set, set_size, verbose_level);

				object_in_projective_space *OiP;
				OiP = PA->create_object_from_int_vec(t_PTS,
						Data->input_string[input_idx], n,
						the_set, set_size, verbose_level);

				back_end(input_idx,
						OiP,
						fp,
						fp_tex,
						verbose_level);
				FREE_OBJECT(OiP);

			}
			else if (Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_POINTS ||
					Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_LINES ||
					Data->input_type[input_idx] == INPUT_TYPE_FILE_OF_PACKINGS ||
					Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
				cout << "input from file " << Data->input_string[input_idx]
					<< ":" << endl;

				set_of_sets *SoS;

				SoS = NEW_OBJECT(set_of_sets);

				cout << "Reading the file " << Data->input_string[input_idx] << endl;
				SoS->init_from_file(
						PA->P->N_points /* underlying_set_size */,
						Data->input_string[input_idx], verbose_level);
				cout << "Read the file " << Data->input_string[input_idx] << endl;

				int h;


				// for use if INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE
				int *Spread_table;
				int nb_spreads;
				int spread_size;

				if (Data->input_type[input_idx] ==
						INPUT_TYPE_FILE_OF_PACKINGS_THROUGH_SPREAD_TABLE) {
					cout << "Reading spread table from file "
						<< Data->input_string2[input_idx] << endl;
					Fio.int_matrix_read_csv(Data->input_string2[input_idx],
							Spread_table, nb_spreads, spread_size,
							0 /* verbose_level */);
					cout << "Reading spread table from file "
							<< Data->input_string2[input_idx] << " done" << endl;
					cout << "The spread table contains " << nb_spreads
							<< " spreads" << endl;
					}

				cout << "processing " << SoS->nb_sets << " objects" << endl;

				for (h = 0; h < SoS->nb_sets; h++) {


					int *the_set_in;
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

					if (Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_POINTS) {
						OiP->init_point_set(PA->P, the_set_in, set_size_in,
								0 /* verbose_level*/);
						}
					else if (Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_LINES) {
						OiP->init_line_set(PA->P, the_set_in, set_size_in,
								0 /* verbose_level*/);
						}
					else if (Data->input_type[input_idx] ==
							INPUT_TYPE_FILE_OF_PACKINGS) {
						OiP->init_packing_from_set(PA->P,
								the_set_in, set_size_in, verbose_level);
						}
					else if (Data->input_type[input_idx] ==
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

void projective_space_job_description::back_end(int input_idx,
		object_in_projective_space *OiP,
		ostream &fp,
		ostream &fp_tex,
		int verbose_level)
{
	int *the_set_out = NULL;
	int set_size_out = 0;

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
		FREE_int(the_set_out);
	}

}

void projective_space_job_description::perform_job_for_one_set(
	int back_end_counter,
	object_in_projective_space *OiP,
	int *&the_set_out,
	int &set_size_out,
	ostream &fp_tex,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *the_set_in;
	int set_size_in;

	if (f_v) {
		cout << "perform_job_for_one_set" << endl;
	}
	the_set_in = OiP->set;
	set_size_in = OiP->sz;

	if (f_embed) {
		if (f_v) {
			cout << "perform_job_for_one_set f_embed" << endl;
		}
		if (f_orthogonal) {
			F->do_embed_orthogonal(orthogonal_epsilon, n,
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
		}
		else {
			F->do_embed_points(n,
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
		}
	}
	else if (f_cone_over) {
		if (f_v) {
			cout << "perform_job_for_one_set f_cone_over" << endl;
		}
		F->do_cone_over(n,
			the_set_in, set_size_in, the_set_out, set_size_out,
			verbose_level - 1);
	}
	else if (f_andre) {
		if (f_v) {
			cout << "perform_job_for_one_set f_andre" << endl;
		}
		if (!f_Q) {
			cout << "please use option -Q <Q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);

		FQ->do_andre(F,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level - 1);

		FREE_OBJECT(FQ);

	}
	else if (f_print) {
		if (f_v) {
			cout << "perform_job_for_one_set f_print" << endl;
		}
		if (f_lines_in_PG) {
			F->do_print_lines_in_PG(n,
				the_set_in, set_size_in);
		}
		else if (f_points_in_PG) {
			F->do_print_points_in_PG(n,
				the_set_in, set_size_in);
		}
		else if (f_points_on_grassmannian) {
			F->do_print_points_on_grassmannian(n, points_on_grassmannian_k,
				the_set_in, set_size_in);
		}
		else if (f_orthogonal) {
			F->do_print_points_in_orthogonal_space(orthogonal_epsilon, n,
				the_set_in, set_size_in, verbose_level);
		}
		else if (f_homogeneous_polynomials) {
			if (!f_homogeneous_polynomial_domain_has_been_allocated) {
				HPD = NEW_OBJECT(homogeneous_polynomial_domain);

				HPD->init(F, n + 1, homogeneous_polynomials_degree,
					FALSE /* f_init_incidence_structure */,
					verbose_level);
				f_homogeneous_polynomial_domain_has_been_allocated = TRUE;
			}
			fp_tex << back_end_counter << ": $";
			HPD->print_equation(fp_tex, the_set_in);
			fp_tex << "$\\\\" << endl;

		}
	}
	else if (f_line_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_line_type" << endl;
		}
#if 0
		F->do_line_type(n,
			the_set_in, set_size_in,
			f_show, verbose_level);
#endif
		int N_lines;
		N_lines = PA->P->nb_rk_k_subspaces_as_int(2);

		int *type;
		type = NEW_int(N_lines);
		for (int i = 0; i < N_lines; i++) {
			vector<int> point_indices;
			PA->P->line_intersection(i,
					the_set_in, set_size_in,
					point_indices,
					verbose_level);
			type[i] = point_indices.size();
		}
		classify C;

		C.init(type, N_lines, FALSE, 0);
		fp_tex << back_end_counter << ": ";
		C.print_file_tex(fp_tex, TRUE);
		fp_tex << "\\\\" << endl;
	}
	else if (f_plane_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_plane_type" << endl;
		}

		int N_planes;
		N_planes = PA->P->nb_rk_k_subspaces_as_int(3);

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
		classify C;

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
	else if (f_plane_type_failsafe) {
		if (f_v) {
			cout << "perform_job_for_one_set f_plane_type_failsafe" << endl;
		}

		F->do_plane_type_failsafe(n,
			the_set_in, set_size_in,
			verbose_level);


	}
	else if (f_conic_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_conic_type" << endl;
		}


		if (n > 2) {

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
			the_set_out = NEW_int(plane_ranks.size());
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

				int *pts;
				int nb_pts;

				nb_pts = point_local_coordinates.size();
				pts = NEW_int(nb_pts);
				for (int j = 0; j < nb_pts; j++) {
					pts[j] = point_local_coordinates[j];
				}
				cout << "plane " << i << " is " << plane_ranks[i]
					<< " has " << nb_pts << " points, in local "
							"coordinates they are ";
				int_vec_print(cout, pts, nb_pts);
				cout << endl;

				int **Pts_on_conic;
				int *nb_pts_on_conic;
				int len1;
				P2->conic_type(
						pts, nb_pts,
						Pts_on_conic, nb_pts_on_conic, len1,
						verbose_level);
				for (int j = 0; j < len1; j++) {
					if (nb_pts_on_conic[j] == q + 1) {
						type[i]++;
					}
				}
				for (int j = 0; j < len1; j++) {
					FREE_int(Pts_on_conic[j]);
				}
				FREE_pint(Pts_on_conic);
				FREE_int(nb_pts_on_conic);
				FREE_int(pts);
			}




			classify C;

			C.init(type, len, FALSE, 0);
			fp_tex << back_end_counter << ": ";
			C.print_file_tex(fp_tex, TRUE);
			fp_tex << "\\\\" << endl;

			FREE_OBJECT(P2);
		}
		else if (n == 2) {
			cout << "conic_type n == 2:" << endl;
			int *intersection_type;
			int highest_intersection_number;

			F->do_conic_type(n, f_randomized, nb_times,
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
	}
	else if (f_hyperplane_type) {
		if (f_v) {
			cout << "perform_job_for_one_set f_hyperplane_type" << endl;
		}
		F->do_m_subspace_type(n, n - 1,
			the_set_in, set_size_in,
			f_show, verbose_level);
	}
	else if (f_bsf3) {
		if (f_v) {
			cout << "perform_job_for_one_set f_bsf3" << endl;
		}
		F->do_blocking_set_family_3(n,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
	}
	else if (f_test_diagonals) {
		if (f_v) {
			cout << "perform_job_for_one_set f_test_diagonals" << endl;
		}
		F->do_test_diagonal_line(n,
			the_set_in, set_size_in,
			test_diagonals_fname,
			verbose_level);
	}
	else if (f_klein) {
		if (f_v) {
			cout << "perform_job_for_one_set f_klein" << endl;
		}
		F->do_Klein_correspondence(n,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
	}
	else if (f_draw_points_in_plane) {
		if (f_v) {
			cout << "perform_job_for_one_set f_draw_points_in_plane" << endl;
		}
		F->do_draw_points_in_plane(
			the_set_in, set_size_in,
			draw_points_in_plane_fname_base, f_point_labels,
			f_embedded, f_sideways,
			verbose_level);
	}
	else if (f_canonical_form) {
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
			f_semilinear, canonical_form_fname_base,
			verbose_level);
	}
	else if (f_ideal) {
		if (f_v) {
			cout << "perform_job_for_one_set f_ideal" << endl;
		}
		F->do_ideal(n,
			the_set_in, set_size_in, ideal_degree,
			the_set_out, set_size_out,
			verbose_level);
	}
	else if (f_intersect_with_set_from_file) {
		if (f_v) {
			cout << "perform_job_for_one_set f_intersect_with_set_from_file" << endl;
		}
		if (!intersect_with_set_from_file_set_has_beed_read) {
			file_io Fio;
			sorting Sorting;

			Fio.read_set_from_file(intersect_with_set_from_file_fname,
					intersect_with_set_from_file_set,
					intersect_with_set_from_file_set_size,
					verbose_level);
			Sorting.int_vec_heapsort(intersect_with_set_from_file_set,
					intersect_with_set_from_file_set_size);

			intersect_with_set_from_file_set_has_beed_read = TRUE;
		}

		if (intersect_with_set_from_file_set_has_beed_read) {
			sorting Sorting;

			cout << "before intersecting the sets of size " << set_size_in
					<< " and " << intersect_with_set_from_file_set_size
					<< ":" << endl;

			the_set_out = NEW_int(set_size_in);
			Sorting.int_vec_intersect_sorted_vectors(the_set_in, set_size_in,
					intersect_with_set_from_file_set,
					intersect_with_set_from_file_set_size,
					the_set_out, set_size_out);

			cout << "the intersection has size " << set_size_out << endl;
		}
	}
	else if (f_arc_with_given_set_as_s_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_given_set_as_i_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_given_set_of_s_lines_diophant(
				the_set_in /*one_lines*/, set_size_in /* nb_one_lines */,
				arc_size /*target_sz*/, arc_d /* target_d */, arc_d_low, arc_s /* target_s */,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (f_vv) {
			D->print_tight();
		}

		if (f_save_system) {
			char fname_system[1000];

			sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
			}

		long int nb_backtrack_nodes;
		int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, verbose_level);

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
		FREE_int(Sol);


	}
	else if (f_arc_with_two_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_two_given_sets_of_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_two_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, arc_s,
				t_lines, nb_t_lines, arc_t,
				arc_size /*target_sz*/, arc_d, arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (f_vv) {
			D->print_tight();
		}

		if (f_save_system) {
			char fname_system[1000];

			sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			}

		long int nb_backtrack_nodes;
		int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, verbose_level);

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
		FREE_int(Sol);


	}
	else if (f_arc_with_three_given_sets_of_lines_after_dualizing) {
		if (f_v) {
			cout << "perform_job_for_one_set f_arc_with_three_given_sets_of_lines_after_dualizing" << endl;
		}
		//int arc_size;
		//int arc_d;
		diophant *D = NULL;
		int f_save_system = TRUE;

		PA->P->arc_with_three_given_line_sets_diophant(
				the_set_in /* s_lines */, set_size_in /* nb_s_lines */, arc_s,
				t_lines, nb_t_lines, arc_t,
				u_lines, nb_u_lines, arc_u,
				arc_size /*target_sz*/, arc_d, arc_d_low,
				TRUE /* f_dualize */,
				D,
				verbose_level);

		if (f_vv) {
			D->print_tight();
		}
		if (f_save_system) {
			char fname_system[1000];

			sprintf(fname_system, "system_%d.diophant", back_end_counter);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << endl;
			D->save_in_general_format(fname_system, 0 /* verbose_level */);
			cout << "perform_job_for_one_set saving the system "
					"to file " << fname_system << " done" << endl;
			//D->print();
			//D->print_tight();
			}

		long int nb_backtrack_nodes;
		int *Sol;
		int nb_sol;

		D->solve_all_mckay(nb_backtrack_nodes, verbose_level);

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
		FREE_int(Sol);


	}
	if (f_v) {
		cout << "perform_job_for_one_set done" << endl;
	}


}


void projective_space_job_description::do_canonical_form(
	int *set, int set_size, int f_semilinear,
	const char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	projective_space *P;
	int canonical_pt;

	if (f_v) {
		cout << "projective_space_job_description::do_canonical_form" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "projective_space_job_description::do_canonical_form before P->init" << endl;
		}

	P->init(n, F,
		TRUE /* f_init_incidence_structure */,
		verbose_level);

	if (f_v) {
		cout << "projective_space_job_description::do_canonical_form after P->init" << endl;
		}

	strong_generators *SG;
	action *A_linear;
	vector_ge *nice_gens;

	A_linear = NEW_OBJECT(action);
	A_linear->init_projective_group(n + 1, F, f_semilinear,
			TRUE /* f_basis */,
			nice_gens,
			verbose_level);

	if (f_v) {
		cout << "projective_space_job_description::do_canonical_form before "
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
		cout << "projective_space_job_description::do_canonical_form done" << endl;
		}

}



}}

