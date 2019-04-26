// process.C
// 
// Anton Betten
// July 28, 2011
//
// 
// Performs some task for each input set.
// The input sets are defined using the data_input_stream class.
// An output file is generated.

#include "orbiter.h"

using namespace std;



using namespace orbiter;




// global data:

int t0; // the system time when the program started


void back_end(int input_idx,
		projective_space_with_action *PA,
		object_in_projective_space *OiP,
		ostream &fp,
		int verbose_level);
void perform_job(int input_idx, projective_space_with_action *PA,
	object_in_projective_space *OiP,
	int *&the_set_out,
	int &set_size_out,
	int verbose_level);
void do_canonical_form(int n, finite_field *F,
	int *set, int set_size, int f_semilinear,
	const char *fname_base, int verbose_level);


int f_Q = FALSE;
int Q;
int f_poly_Q = FALSE;
const char *poly_Q = NULL;

int f_embed = FALSE;
int f_orthogonal = FALSE;
int orthogonal_epsilon = 0;
int f_andre = FALSE;
int f_print = FALSE;
int f_lines_in_PG = FALSE;
int f_points_in_PG = FALSE;
int f_points_on_grassmannian = FALSE;
int points_on_grassmannian_k = 0;
int f_group = FALSE;
int f_list_group_elements = FALSE;
int f_line_type = FALSE;
int f_plane_type = FALSE;
int f_plane_type_failsafe = FALSE;
int f_conic_type = FALSE;
int f_randomized = FALSE;
int nb_times = 0;
int f_hyperplane_type = FALSE;
int f_fast = FALSE;
int f_show = FALSE;
int f_cone = FALSE;
int f_move_line = FALSE;
int from_line = 0, to_line = 0;
int f_bsf3 = FALSE;
int f_test_diagonals = FALSE;
const char *test_diagonals_fname = NULL;
int f_klein = FALSE;
int f_draw_points_in_plane = FALSE;
const char *draw_points_in_plane_fname_base = NULL;
int f_point_labels = FALSE;
int f_canonical_form = FALSE;
const char *canonical_form_fname_base = NULL;
int f_ideal = FALSE;
int ideal_degree = 0;
int f_find_Eckardt_points_from_arc = FALSE;
int f_embedded = FALSE;
int f_sideways = FALSE;

int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	int f_n = FALSE;
	int n;
	int f_poly = FALSE;
	const char *poly = NULL;

	int f_input = FALSE;
	data_input_stream *Data = NULL;

	int f_fname_base_out = FALSE;
	const char *fname_base_out = NULL;

	

	t0 = os_ticks();


	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-Q") == 0) {
			f_Q = TRUE;
			Q = atoi(argv[++i]);
			cout << "-Q " << Q << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-poly") == 0) {
			f_poly = TRUE;
			poly = argv[++i];
			cout << "-poly " << poly << endl;
			}
		else if (strcmp(argv[i], "-poly_Q") == 0) {
			f_poly_Q = TRUE;
			poly_Q = argv[++i];
			cout << "-poly_Q " << poly_Q << endl;
			}
		else if (strcmp(argv[i], "-input") == 0) {
			f_input = TRUE;
			Data = NEW_OBJECT(data_input_stream);
			i += Data->read_arguments(argc - (i - 1),
				argv + i, verbose_level);

			cout << "-input" << endl;
			}
		else if (strcmp(argv[i], "-fname_base_out") == 0) {
			f_fname_base_out = TRUE;
			fname_base_out = argv[++i];
			cout << "-fname_base_out " << fname_base_out << endl;
			}
		else if (strcmp(argv[i], "-embed") == 0) {
			f_embed = TRUE;
			cout << "-embed" << endl;
			}
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			orthogonal_epsilon = atoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_epsilon << endl;
			}
		else if (strcmp(argv[i], "-andre") == 0) {
			f_andre = TRUE;
			cout << "-andre " << endl;
			}
		else if (strcmp(argv[i], "-print") == 0) {
			f_print = TRUE;
			cout << "-print " << endl;
			}
		else if (strcmp(argv[i], "-lines_in_PG") == 0) {
			f_lines_in_PG = TRUE;
			cout << "-lines_in_PG " << endl;
			}
		else if (strcmp(argv[i], "-points_in_PG") == 0) {
			f_points_in_PG = TRUE;
			cout << "-points_in_PG " << endl;
			}
		else if (strcmp(argv[i], "-points_on_grassmannian") == 0) {
			f_points_on_grassmannian = TRUE;
			points_on_grassmannian_k = atoi(argv[++i]);
			cout << "-points_on_grassmannian " << points_on_grassmannian_k << endl;
			}
		else if (strcmp(argv[i], "-group") == 0) {
			f_group = TRUE;
			cout << "-group" << endl;
			}
		else if (strcmp(argv[i], "-list_group_elements") == 0) {
			f_list_group_elements = TRUE;
			cout << "-list_group_elements" << endl;
			}
		else if (strcmp(argv[i], "-line_type") == 0) {
			f_line_type = TRUE;
			cout << "-line_type" << endl;
			}
		else if (strcmp(argv[i], "-plane_type") == 0) {
			f_plane_type = TRUE;
			cout << "-plane_type" << endl;
			}
		else if (strcmp(argv[i], "-plane_type_failsafe") == 0) {
			f_plane_type_failsafe = TRUE;
			cout << "-plane_type_failsafe" << endl;
			}
		else if (strcmp(argv[i], "-conic_type") == 0) {
			f_conic_type = TRUE;
			cout << "-conic_type " << endl;
			}
		else if (strcmp(argv[i], "-randomized") == 0) {
			f_randomized = TRUE;
			nb_times = atoi(argv[++i]);
			cout << "-randomized " << nb_times << endl;
			}
		else if (strcmp(argv[i], "-hyperplane_type") == 0) {
			f_hyperplane_type = TRUE;
			cout << "-hyperplane_type" << endl;
			}
		else if (strcmp(argv[i], "-fast") == 0) {
			f_fast = TRUE;
			cout << "-fast" << endl;
			}
		else if (strcmp(argv[i], "-show") == 0) {
			f_show = TRUE;
			cout << "-show" << endl;
			}
		else if (strcmp(argv[i], "-cone") == 0) {
			f_cone = TRUE;
			cout << "-cone" << endl;
			}
		else if (strcmp(argv[i], "-move_line") == 0) {
			f_move_line = TRUE;
			from_line = atoi(argv[++i]);
			to_line = atoi(argv[++i]);
			cout << "-move_line" << from_line << " " << to_line << endl;
			}
		else if (strcmp(argv[i], "-bsf3") == 0) {
			f_bsf3 = TRUE;
			cout << "-bsf3" << endl;
			}
		else if (strcmp(argv[i], "-test_diagonals") == 0) {
			f_test_diagonals = TRUE;
			test_diagonals_fname = argv[++i];
			cout << "-test_diagonals " << test_diagonals_fname << endl;
			}
		else if (strcmp(argv[i], "-klein") == 0) {
			f_klein = TRUE;
			cout << "-klein" << endl;
			}
		else if (strcmp(argv[i], "-draw_points_in_plane") == 0) {
			f_draw_points_in_plane = TRUE;
			draw_points_in_plane_fname_base = argv[++i];
			cout << "-draw_points_in_plane" << endl;
			}
		else if (strcmp(argv[i], "-point_labels") == 0) {
			f_point_labels = TRUE;
			cout << "-point_labels" << endl;
			}
		else if (strcmp(argv[i], "-find_Eckardt_points_from_arc") == 0) {
			f_find_Eckardt_points_from_arc = TRUE;
			cout << "-find_Eckardt_points_from_arc" << endl;
			}
		else if (strcmp(argv[i], "-canonical_form") == 0) {
			f_canonical_form = TRUE;
			canonical_form_fname_base = argv[++i];
			cout << "-canonical_form" << canonical_form_fname_base << endl;
			}
		else if (strcmp(argv[i], "-ideal") == 0) {
			f_ideal = TRUE;
			ideal_degree = atoi(argv[++i]);
			cout << "-ideal " << ideal_degree << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded" << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways" << endl;
			}
		}

	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "please use option -n <n> to specify the projective dimension" << endl;
		exit(1);
		}
	if (!f_fname_base_out) {
		cout << "please use option -fname_base_out <fname_base_out>" << endl;
		exit(1);
		}

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0);
	

	projective_space_with_action *PA;
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

	PA->projective_space_with_action::init(
		F, n, f_semilinear,
		f_init_incidence_structure,
		verbose_level);


	nb_objects_to_test = Data->count_number_of_objects_to_test(
		verbose_level - 1);

	cout << "nb_objects_to_test = " << nb_objects_to_test << endl;

	t0 = os_ticks();

	file_io Fio;
	char fname_out[1000];

	sprintf(fname_out, "%s.txt", fname_base_out);

	{
	ofstream fp(fname_out);
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
					PA,
					OiP,
					fp,
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
					PA,
					OiP,
					fp,
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
						PA,
						OiP,
						fp,
						verbose_level);
				FREE_OBJECT(OiP);

			}
		}
		else {
			cout << "unknown type of input object" << endl;
			exit(1);
		}

	} // next input_idx

		fp << -1 << endl;
	}
	cout << "Written file " << fname_out << " of size "
			<< Fio.file_size(fname_out) << endl;





	the_end(t0);
}

void back_end(int input_idx,
		projective_space_with_action *PA,
		object_in_projective_space *OiP,
		ostream &fp,
		int verbose_level)
{
	int *the_set_out = NULL;
	int set_size_out = 0;

	perform_job(input_idx, PA,
			OiP,
			the_set_out, set_size_out,
			verbose_level);

	fp << set_size_out;
	for (int i = 0; i < set_size_out; i++) {
		fp << " " << the_set_out[i];
	}
	fp << endl;

	if (the_set_out) {
		FREE_int(the_set_out);
	}

}

void perform_job(int input_idx,
	projective_space_with_action *PA,
	object_in_projective_space *OiP,
	int *&the_set_out,
	int &set_size_out,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *the_set_in;
	int set_size_in;
	finite_field *F;
	int n;

	if (f_v) {
		cout << "perform_job" << endl;
	}
	F = PA->F;
	n = PA->n;
	the_set_in = OiP->set;
	set_size_in = OiP->sz;

	if (f_embed) {
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
	else if (f_cone) {
		F->do_cone_over(n,
			the_set_in, set_size_in, the_set_out, set_size_out,
			verbose_level - 1);
		}
	else if (f_andre) {
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
		}
	else if (f_group) {

		cout << "do_group_in_PG is disabled" << endl;

#if 0
		do_group_in_PG(n, F,
			the_set_in, set_size_in, f_list_group_elements, verbose_level);
#endif
		}
	else if (f_line_type) {
		F->do_line_type(n,
			the_set_in, set_size_in,
			f_show, verbose_level);
		}
	else if (f_plane_type) {
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
#if 0
		if (f_fast) {
			do_m_subspace_type_fast(n, q, 2, f_poly, poly,
				the_set_in, set_size_in,
				f_show, verbose_level);
			}
		else {
			do_m_subspace_type(n, q, 2, f_poly, poly,
				the_set_in, set_size_in,
				f_show, verbose_level);
			}
#endif

		}
	else if (f_plane_type_failsafe) {

		F->do_plane_type_failsafe(n,
			the_set_in, set_size_in,
			verbose_level);


		}
	else if (f_conic_type) {
		int *intersection_type;
		int highest_intersection_number;

		F->do_conic_type(n, f_randomized, nb_times,
			the_set_in, set_size_in,
			intersection_type, highest_intersection_number, verbose_level);
			// in GALOIS/geometric_operations.C


		for (int i = 0; i <= highest_intersection_number; i++) {
			if (intersection_type[i]) {
				cout << i << "^" << intersection_type[i] << " ";
				}
			}
		cout << endl;

		FREE_int(intersection_type);
		}
	else if (f_hyperplane_type) {
		F->do_m_subspace_type(n, n - 1,
			the_set_in, set_size_in,
			f_show, verbose_level);
		}
	else if (f_move_line) {

		cout << "move_line is disabled" << endl;
		exit(1);

#if 0
		do_move_line_in_PG(n, F,
			from_line, to_line,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
		sprintf(fname_out, "%s_moved_line.txt", fname_base);
#endif
		}
	else if (f_bsf3) {
		F->do_blocking_set_family_3(n,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
		}
	else if (f_test_diagonals) {
		F->do_test_diagonal_line(n,
			the_set_in, set_size_in,
			test_diagonals_fname,
			verbose_level);
		}
	else if (f_klein) {
		F->do_Klein_correspondence(n,
			the_set_in, set_size_in,
			the_set_out, set_size_out,
			verbose_level);
		}
	else if (f_draw_points_in_plane) {
		F->do_draw_points_in_plane(
			the_set_in, set_size_in,
			draw_points_in_plane_fname_base, f_point_labels,
			f_embedded, f_sideways,
			verbose_level);
		}
	else if (f_canonical_form) {
		int f_semilinear = TRUE;
		number_theory_domain NT;
		if (NT.is_prime(F->q)) {
			f_semilinear = FALSE;
			}
		do_canonical_form(n, F,
			the_set_in, set_size_in,
			f_semilinear, canonical_form_fname_base,
			verbose_level);
		}
	else if (f_ideal) {
		F->do_ideal(n,
			the_set_in, set_size_in, ideal_degree,
			the_set_out, set_size_out,
			verbose_level);
		}
	else if (f_find_Eckardt_points_from_arc) {
#if 0
		do_find_Eckardt_points_from_arc(n, F,
			the_set_in, set_size_in,
			verbose_level);
#endif
		}

}

void do_canonical_form(int n, finite_field *F,
	int *set, int set_size, int f_semilinear,
	const char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	projective_space *P;
	int canonical_pt;

	if (f_v) {
		cout << "do_canonical_form" << endl;
		}

	P = NEW_OBJECT(projective_space);

	if (f_v) {
		cout << "do_canonical_form before P->init" << endl;
		}

	P->init(n, F,
		TRUE /* f_init_incidence_structure */,
		verbose_level);

	if (f_v) {
		cout << "do_canonical_form after P->init" << endl;
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
		cout << "do_canonical_form before "
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
		cout << "do_canonical_form done" << endl;
		}

}


