// process.C
// 
// Anton Betten
// July 28, 2011
//
// 
// -embed -orthogonal <epsilon> turns points
// from Q^<epsilon>(n,q) into points from PG(n,q)
//
//

#include "orbiter.h"

using namespace std;



using namespace orbiter;




// global data:

int t0; // the system time when the program started


void do_canonical_form(int n, finite_field *F,
	int *set, int set_size, int f_semilinear,
	const char *fname_base, int verbose_level);



int main(int argc, char **argv)
{
	int verbose_level = 0;
	int i;
	int f_q = FALSE;
	int q;
	int f_n = FALSE;
	int n;
	int f_k = FALSE;
	int k;
	int f_poly = FALSE;
	char *poly = NULL;
	int f_Q = FALSE;
	int Q;
	int f_poly_Q = FALSE;
	char *poly_Q = NULL;
	int f_embed = FALSE;
	int f_orthogonal = FALSE;
	int orthogonal_epsilon = 0;
	int f_file = FALSE;
	char *fname;
	int f_andre = FALSE;
	int f_print = FALSE;
	int f_lines_in_PG = FALSE;
	int f_points_in_PG = FALSE;
	int f_points_on_grassmannian = FALSE;
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
	char *test_diagonals_fname = NULL;
	int f_klein = FALSE;
	int f_draw_points_in_plane = FALSE;
	int f_point_labels = FALSE;
	int f_set = FALSE;
	const char *set_label = NULL;
	const char *the_set = NULL;
	int f_canonical_form = FALSE;
	int f_ideal = FALSE;
	int ideal_degree = 0;
	int f_find_Eckardt_points_from_arc = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	
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

	int *the_set_in = NULL;
	int *the_set_out = NULL;
	int set_size_in = 0;
	int set_size_out = 0;
	char fname_out[1000];
	char fname_base[1000];
	char ext[1000];
	file_io Fio;

	if (!f_file && !f_set) {
		cout << "please use option -file <fname> "
				"to specify input file or "
				"-set <set_label> <list_of_elements>" << endl;
		}
	
	if (f_file) {
		Fio.read_set_from_file(fname, the_set_in, set_size_in, verbose_level - 1);
		cout << "read set of size " << set_size_in
				<< " from file " << fname << endl;
		strcpy(fname_base, fname);
		get_extension_if_present(fname_base, ext);
		chop_off_extension_if_present(fname_base, ext);
		}
	else if (f_set) {
		strcpy(fname_base, set_label);
		int_vec_scan(the_set, the_set_in, set_size_in);
		}
	else {
		cout << "should not be here" << endl;
		exit(1);
		}

	cout << "The input set has size " << set_size_in << ":" << endl;
	cout << "The input set is:" << endl;
	int_vec_print(cout, the_set_in, set_size_in);
	cout << endl;

	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	finite_field *F;
	number_theory_domain NT;

	F = NEW_OBJECT(finite_field);
	F->init_override_polynomial(q, poly, 0);
	
	if (f_embed) {
		if (f_orthogonal) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			F->do_embed_orthogonal(orthogonal_epsilon, n,
				the_set_in, the_set_out, set_size_in, verbose_level - 1);
			set_size_out = set_size_in;
			sprintf(fname_out, "%s_embedded.txt", fname_base);
			}
		else {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			F->do_embed_points(n,
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
		F->do_cone_over(n,
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

		FQ = NEW_OBJECT(finite_field);
		FQ->init_override_polynomial(Q, poly_Q, 0);
		
		FQ->do_andre(F,
			the_set_in, set_size_in, 
			the_set_out, set_size_out, 
			verbose_level - 1);
		sprintf(fname_out, "%s_andre.txt", fname_base);

		FREE_OBJECT(FQ);
		
		}
	else if (f_print) {
		if (f_lines_in_PG) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			F->do_print_lines_in_PG(n,
				the_set_in, set_size_in);
			}
		else if (f_points_in_PG) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			F->do_print_points_in_PG(n,
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
			F->do_print_points_on_grassmannian(n, k,
				the_set_in, set_size_in);
			}
		else if (f_orthogonal) {
			if (!f_n) {
				cout << " please use option -n <n>" << endl;
				exit(1);
				}
			F->do_print_points_in_orthogonal_space(orthogonal_epsilon, n,
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
		F->do_line_type(n,
			the_set_in, set_size_in, 
			f_show, verbose_level);
		}
	else if (f_plane_type) {
		int *intersection_type;
		int highest_intersection_number;

		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		F->do_plane_type(n,
			the_set_in, set_size_in, 
			intersection_type, highest_intersection_number, verbose_level);

			// GALOIS/geometric_operations.C

		for (i = 0; i <= highest_intersection_number; i++) {
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

		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		F->do_plane_type_failsafe(n,
			the_set_in, set_size_in, 
			verbose_level);

			// GALOIS/geometric_operations.C

		}
	else if (f_conic_type) {
		int *intersection_type;
		int highest_intersection_number;

		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		F->do_conic_type(n, f_randomized, nb_times,
			the_set_in, set_size_in, 
			intersection_type, highest_intersection_number, verbose_level);
			// in GALOIS/geometric_operations.C


		for (i = 0; i <= highest_intersection_number; i++) {
			if (intersection_type[i]) {
				cout << i << "^" << intersection_type[i] << " ";
				}
			}
		cout << endl;

		FREE_int(intersection_type);
		}
	else if (f_hyperplane_type) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		F->do_m_subspace_type(n, n - 1,
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
		F->do_blocking_set_family_3(n,
			the_set_in, set_size_in, 
			the_set_out, set_size_out, 
			verbose_level);
		sprintf(fname_out, "%s_bsf3.txt", fname_base);
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
		sprintf(fname_out, "%s_klein.txt", fname_base);
		}
	else if (f_draw_points_in_plane) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		F->do_draw_points_in_plane(
			the_set_in, set_size_in, fname_base, f_point_labels,
			f_embedded, f_sideways, 
			verbose_level);
		}
	else if (f_canonical_form) {
		if (!f_n) {
			cout << " please use option -n <n>" << endl;
			exit(1);
			}
		int f_semilinear = FALSE;
		if (!NT.is_prime(q)) {
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
		F->do_ideal(n,
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
		int_vec_print(cout, the_set_out, set_size_out);
		cout << endl;
		}

	if (the_set_out && set_size_out) {
		Fio.write_set_to_file(fname_out,
				the_set_out, set_size_out, verbose_level);
		cout << "written file " << fname_out
				<< " of size " << Fio.file_size(fname_out) << endl;
		}
	
	if (the_set_in) {
		FREE_int(the_set_in);
		the_set_in = NULL;
		}
	if (the_set_out) {
		FREE_int(the_set_out);
		the_set_out = NULL;
		}
	the_end(t0);
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

