// process.C
// 
// Anton Betten
// July 28, 2011
//
// 
// -embed -orthogonal <epsilon> turns points from Q^<epsilon>(n,q) into points from PG(n,q)
//
//

#include "orbiter.h"





// global data:

INT t0; // the system time when the program started


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT f_q = FALSE;
	INT q;
	INT f_n = FALSE;
	INT n;
	INT f_k = FALSE;
	INT k;
	INT f_poly = FALSE;
	char *poly = NULL;
	INT f_Q = FALSE;
	INT Q;
	INT f_poly_Q = FALSE;
	char *poly_Q = NULL;
	INT f_embed = FALSE;
	INT f_orthogonal = FALSE;
	INT orthogonal_epsilon = 0;
	INT f_file = FALSE;
	char *fname;
	INT f_andre = FALSE;
	INT f_print = FALSE;
	INT f_lines_in_PG = FALSE;
	INT f_points_in_PG = FALSE;
	INT f_points_on_grassmannian = FALSE;
	INT f_group = FALSE;
	INT f_list_group_elements = FALSE;
	INT f_line_type = FALSE;
	INT f_plane_type = FALSE;
	INT f_plane_type_failsafe = FALSE;
	INT f_conic_type = FALSE;
	INT f_randomized = FALSE;
	INT nb_times = 0;
	INT f_hyperplane_type = FALSE;
	INT f_fast = FALSE;
	INT f_show = FALSE;
	INT f_cone = FALSE;
	INT f_move_line = FALSE;
	INT from_line = 0, to_line = 0;
	INT f_bsf3 = FALSE;
	INT f_test_diagonals = FALSE;
	char *test_diagonals_fname = NULL;
	INT f_klein = FALSE;
	INT f_draw_points_in_plane = FALSE;
	INT f_point_labels = FALSE;
	INT f_set = FALSE;
	const char *set_label = NULL;
	const char *the_set = NULL;
	INT f_canonical_form = FALSE;
	INT f_ideal = FALSE;
	INT ideal_degree = 0;
	INT f_find_Eckardt_points_from_arc = FALSE;
	INT f_embedded = FALSE;
	INT f_sideways = FALSE;
	
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
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
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
		else if (strcmp(argv[i], "-embed") == 0) {
			f_embed = TRUE;
			cout << "-embed" << endl;
			}
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal = TRUE;
			orthogonal_epsilon = atoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_epsilon << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
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
			cout << "-points_on_grassmannian " << endl;
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
			cout << "-canonical_form" << endl;
			}
		else if (strcmp(argv[i], "-ideal") == 0) {
			f_ideal = TRUE;
			ideal_degree = atoi(argv[++i]);
			cout << "-ideal " << ideal_degree << endl;
			}
		else if (strcmp(argv[i], "-set") == 0) {
			f_set = TRUE;
			set_label = argv[++i];
			the_set = argv[++i];
			cout << "-set " << set_label << " " << the_set << endl;
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

	INT *the_set_in = NULL;
	INT *the_set_out = NULL;
	INT set_size_in = 0;
	INT set_size_out = 0;
	char fname_out[1000];
	char fname_base[1000];
	char ext[1000];
	
	if (!f_file && !f_set) {
		cout << "please use option -file <fname> to specify input file or -set <set_label> <list_of_elements>" << endl;
		}
	
	if (f_file) {
		read_set_from_file(fname, the_set_in, set_size_in, verbose_level - 1);
		cout << "read set of size " << set_size_in << " from file " << fname << endl;
		strcpy(fname_base, fname);
		get_extension_if_present(fname_base, ext);
		chop_off_extension_if_present(fname_base, ext);
		}
	else if (f_set) {
		strcpy(fname_base, set_label);
		INT_vec_scan(the_set, the_set_in, set_size_in);
		}
	else {
		cout << "should not be here" << endl;
		exit(1);
		}

	cout << "The input set has size " << set_size_in << ":" << endl;
	cout << "The input set is:" << endl;
	INT_vec_print(cout, the_set_in, set_size_in);
	cout << endl;

	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	finite_field *F;

	F = new finite_field;
	F->init_override_polynomial(q, poly, 0);
	
	if (f_embed) {
		if (f_orthogonal) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			do_embed_orthogonal(orthogonal_epsilon, n, F, 
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
			sprintf(fname_out, "%s_embedded.txt", fname_base);
			}
		else {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			do_embed_points(n, F, 
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
			sprintf(fname_out, "%s_embedded.txt", fname_base);
			}
		}
	else if (f_cone) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_cone_over(n, F, 
			the_set_in, set_size_in, the_set_out, set_size_out, 
			verbose_level - 1);
		sprintf(fname_out, "%s_cone.txt", fname_base);
		}
	else if (f_andre) {
		if (!f_q) {
			cout << "please use option -q <q>" << endl;
			exit(1);
			}

		finite_field *FQ;

		FQ = new finite_field;
		FQ->init_override_polynomial(Q, poly_Q, 0);
		
		do_andre(FQ, F, 
			the_set_in, set_size_in, 
			the_set_out, set_size_out, 
			verbose_level - 1);
		sprintf(fname_out, "%s_andre.txt", fname_base);

		delete FQ;
		
		}
	else if (f_print) {
		if (f_lines_in_PG) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			do_print_lines_in_PG(n, F, 
				the_set_in, set_size_in);
			}
		else if (f_points_in_PG) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			do_print_points_in_PG(n, F, 
				the_set_in, set_size_in);
			}
		else if (f_points_on_grassmannian) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			if (!f_k) {
				cout << " please use option -k <k>" << endl;
				exit(1);
				}
			do_print_points_on_grassmannian(n, k, F, 
				the_set_in, set_size_in);
			}
		else if (f_orthogonal) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			do_print_points_in_orthogonal_space(orthogonal_epsilon, n, F, 
				the_set_in, set_size_in, verbose_level);
			}
		}
	else if (f_group) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}

		cout << "do_group_in_PG is disabled" << endl;

#if 0
		do_group_in_PG(n, F, 
			the_set_in, set_size_in, f_list_group_elements, verbose_level);
#endif
		}
	else if (f_line_type) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_line_type(n, F, 
			the_set_in, set_size_in, 
			f_show, verbose_level);
		}
	else if (f_plane_type) {
		INT *intersection_type;
		INT highest_intersection_number;

		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_plane_type(n, F, 
			the_set_in, set_size_in, 
			intersection_type, highest_intersection_number, verbose_level);

			// GALOIS/geometric_operations.C

		for (i = 0; i <= highest_intersection_number; i++) {
			if (intersection_type[i]) {
				cout << i << "^" << intersection_type[i] << " ";
				}
			}
		cout << endl;

		FREE_INT(intersection_type);
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

		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_plane_type_failsafe(n, F, 
			the_set_in, set_size_in, 
			verbose_level);

			// GALOIS/geometric_operations.C

		}
	else if (f_conic_type) {
		INT *intersection_type;
		INT highest_intersection_number;

		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_conic_type(n, F, f_randomized, nb_times, 
			the_set_in, set_size_in, 
			intersection_type, highest_intersection_number, verbose_level);
			// in GALOIS/geometric_operations.C


		for (i = 0; i <= highest_intersection_number; i++) {
			if (intersection_type[i]) {
				cout << i << "^" << intersection_type[i] << " ";
				}
			}
		cout << endl;

		FREE_INT(intersection_type);
		}
	else if (f_hyperplane_type) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_m_subspace_type(n, F, n - 1, 
			the_set_in, set_size_in, 
			f_show, verbose_level);
		}
	else if (f_move_line) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}

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
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_blocking_set_family_3(n, F, 
			the_set_in, set_size_in, 
			the_set_out, set_size_out, 
			verbose_level);
		sprintf(fname_out, "%s_bsf3.txt", fname_base);
		}
	else if (f_test_diagonals) {
		do_test_diagonal_line(n, F, 
			the_set_in, set_size_in, 
			test_diagonals_fname, 
			verbose_level);
		}
	else if (f_klein) {
		do_Klein_correspondence(n, F, 
			the_set_in, set_size_in, 
			the_set_out, set_size_out, 
			verbose_level);
		sprintf(fname_out, "%s_klein.txt", fname_base);
		}
	else if (f_draw_points_in_plane) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_draw_points_in_plane(F, 
			the_set_in, set_size_in, fname_base, f_point_labels,
			f_embedded, f_sideways, 
			verbose_level);
		}
	else if (f_canonical_form) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		INT f_semilinear = FALSE;
		if (!is_prime(q)) {
			f_semilinear = TRUE;
			}
		do_canonical_form(n, F, 
			the_set_in, set_size_in, f_semilinear, fname_base,
			verbose_level);
		}
	else if (f_ideal) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		do_ideal(n, F, 
			the_set_in, set_size_in, ideal_degree, 
			verbose_level);
		}
	else if (f_find_Eckardt_points_from_arc) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
#if 0
		do_find_Eckardt_points_from_arc(n, F, 
			the_set_in, set_size_in, 
			verbose_level);
#endif
		}

	if (the_set_out) {
		cout << "output: ";
		INT_vec_print(cout, the_set_out, set_size_out);
		cout << endl;
		}

	if (the_set_out && set_size_out) {
		write_set_to_file(fname_out, the_set_out, set_size_out, verbose_level);
		cout << "written file " << fname_out << " of size " << file_size(fname_out) << endl;
		}
	
	if (the_set_in) {
		FREE_INT(the_set_in);
		the_set_in = NULL;
		}
	if (the_set_out) {
		FREE_INT(the_set_out);
		the_set_out = NULL;
		}
	the_end(t0);
}


