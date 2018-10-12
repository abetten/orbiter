// draw_colored_graph.C
// 
// Anton Betten
// November 25, 2014
//
// 
//
//

#include "orbiter.h"


// global data:

int t0; // the system time when the program started

void early_test_function_cliques(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
void early_test_function_cocliques(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);
void characteristic_polynomial(colored_graph *CG, int verbose_level);

int main(int argc, char **argv)
{
	int i, j;
	t0 = os_ticks();
	int verbose_level = 0;
	int f_file = FALSE;	
	const char *fname = NULL;
	int f_coordinates = FALSE;
	int xmax_in = ONE_MILLION;
	int ymax_in = ONE_MILLION;
	int xmax_out = ONE_MILLION;
	int ymax_out = ONE_MILLION;
	int f_export_magma = FALSE;
	const char *magma_fname = NULL;
	int f_export_matlab = FALSE;
	const char *matlab_fname = NULL;
	int f_on_circle = FALSE;
	int f_bitmatrix = FALSE;
	int f_labels = FALSE;
	int f_embedded = FALSE;
	int f_sideways = FALSE;
	int f_scale = FALSE;
	double scale = .45;
	int f_line_width = FALSE;
	double line_width = 1.5;
	int f_aut = FALSE;
	int f_is_association_scheme = FALSE;
	int f_all_cliques = FALSE;
	int f_all_cocliques = FALSE;
	int f_characteristic_polynomial = FALSE;
	int f_export = FALSE;
	const char *export_fname = NULL;
	int f_export_laplacian = FALSE;
	const char *export_laplacian_fname = NULL;
	int f_expand_power = FALSE;
	int expand_power = 0;
	int expand_power_nb_graphs;
	const char *expand_power_graph_fname[1000];
	int f_partitioned = FALSE;
	int f_partition = FALSE;
	int part[1000];
	int nb_parts = 0;
	int f_Levi = FALSE;
	int Levi_m = 0;
	int Levi_n = 0;
	int f_Levi_discrete = FALSE;
	int f_radius = FALSE;
	double radius = 1000;
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-file") == 0) {
			f_file = TRUE;
			fname = argv[++i];
			cout << "-file " << fname << endl;
			}
		else if (strcmp(argv[i], "-coordinates") == 0) {
			f_coordinates = TRUE;
			xmax_in = atoi(argv[++i]);
			ymax_in = atoi(argv[++i]);
			xmax_out = atoi(argv[++i]);
			ymax_out = atoi(argv[++i]);
			cout << "-coordinates " << xmax_in << " " << ymax_in
				<< " " << xmax_out << " " << ymax_out << endl;
			}
		else if (strcmp(argv[i], "-export_magma") == 0) {
			f_export_magma = TRUE;
			magma_fname = argv[++i];
			cout << "-export_magma " << magma_fname << endl;
			}
		else if (strcmp(argv[i], "-export_matlab") == 0) {
			f_export_matlab = TRUE;
			matlab_fname = argv[++i];
			cout << "-export_matlab " << matlab_fname << endl;
			}
		else if (strcmp(argv[i], "-on_circle") == 0) {
			f_on_circle = TRUE;
			cout << "-on_circle " << endl;
			}
		else if (strcmp(argv[i], "-bitmatrix") == 0) {
			f_bitmatrix = TRUE;
			cout << "-bitmatrix " << endl;
			}
		else if (strcmp(argv[i], "-labels") == 0) {
			f_labels = TRUE;
			cout << "-labels " << endl;
			}
		else if (strcmp(argv[i], "-embedded") == 0) {
			f_embedded = TRUE;
			cout << "-embedded " << endl;
			}
		else if (strcmp(argv[i], "-sideways") == 0) {
			f_sideways = TRUE;
			cout << "-sideways " << endl;
			}
		else if (strcmp(argv[i], "-radius") == 0) {
			f_radius = TRUE;
			sscanf(argv[++i], "%lf", &radius);
			cout << "-radius " << radius << endl;
			}
		else if (strcmp(argv[i], "-scale") == 0) {
			f_scale = TRUE;
			sscanf(argv[++i], "%lf", &scale);
			cout << "-scale " << scale << endl;
			}
		else if (strcmp(argv[i], "-line_width") == 0) {
			f_line_width = TRUE;
			sscanf(argv[++i], "%lf", &line_width);
			cout << "-line_width " << line_width << endl;
			}
		else if (strcmp(argv[i], "-aut") == 0) {
			f_aut = TRUE;
			cout << "-aut " << endl;
			}
		else if (strcmp(argv[i], "-is_association_scheme") == 0) {
			f_is_association_scheme = TRUE;
			cout << "-is_association_scheme " << endl;
			}
		else if (strcmp(argv[i], "-all_cliques") == 0) {
			f_all_cliques = TRUE;
			cout << "-all_cliques " << endl;
			}
		else if (strcmp(argv[i], "-all_cocliques") == 0) {
			f_all_cocliques = TRUE;
			cout << "-all_cocliques " << endl;
			}
		else if (strcmp(argv[i], "-characteristic_polynomial") == 0) {
			f_characteristic_polynomial = TRUE;
			cout << "-characteristic_polynomial " << endl;
			}
		else if (strcmp(argv[i], "-export") == 0) {
			f_export = TRUE;
			export_fname = argv[++i];
			cout << "-export " << export_fname << endl;
			}
		else if (strcmp(argv[i], "-export_laplacian") == 0) {
			f_export_laplacian = TRUE;
			export_laplacian_fname = argv[++i];
			cout << "-export_laplacian " << export_laplacian_fname << endl;
			}
		else if (strcmp(argv[i], "-expand_power") == 0) {
			f_expand_power = TRUE;
			sscanf(argv[++i], "%d", &expand_power);
			for (j = 0; ; j++) {
				expand_power_graph_fname[j] = argv[++i];
				cout << "j=" << j << " : " << expand_power_graph_fname[j] << endl;
				if (strcmp(expand_power_graph_fname[j], "-1") == 0) {
					cout << "break j=" << j << endl;
					break;
					}
				}
			expand_power_nb_graphs = j;
			cout << "-expand_power " << expand_power << " " << endl;
			for (j = 0; j < expand_power_nb_graphs; j++) {
				cout << expand_power_graph_fname[j] << " " << endl;
				}
			cout << endl;
			}
		else if (strcmp(argv[i], "-partitioned") == 0) {
			f_partitioned = TRUE;
			cout << "-partitioned" << endl;
			}
		else if (strcmp(argv[i], "-partition") == 0) {
			f_partition = TRUE;
			for (j = 0; ; j++) {
				part[j] = atoi(argv[++i]);
				if (part[j] == -1) {
					break;
					}
				}
			nb_parts = j;
			cout << "-partition ";
			for (j = 0; j < nb_parts; j++) {
				cout << " " << part[j] << endl;
				}
			cout << endl;
			}
		else if (strcmp(argv[i], "-Levi") == 0) {
			f_Levi = TRUE;
			Levi_m = atoi(argv[++i]);
			Levi_n = atoi(argv[++i]);
			cout << "-Levi " << Levi_m << " " << Levi_n << endl;
			}
		else if (strcmp(argv[i], "-Levi_discrete") == 0) {
			f_Levi_discrete = TRUE;
			cout << "-Levi_discrete " << endl;
			}

		}

	if (!f_file) {
		cout << "Please specify the file name using -file <fname>" << endl;
		exit(1);
		}
	colored_graph *CG;

	CG = NEW_OBJECT(colored_graph);

	CG->load(fname, verbose_level);

	if (f_export_magma) {
		CG->export_to_magma(magma_fname, 0 /* verbose_level */);
		}

	if (f_export_matlab) {
		CG->export_to_file_matlab(matlab_fname, 0 /* verbose_level */);
		}

	if (f_export) {
		CG->export_to_file(export_fname, 0 /* verbose_level */);
		}
	if (f_export_laplacian) {
		CG->export_laplacian_to_file(export_laplacian_fname, 0 /* verbose_level */);
		}






	if (f_on_circle) {
		char fname2[1000];

		strcpy(fname2, fname);
		replace_extension_with(fname2, "_on_circle");
		CG->draw_on_circle(fname2, 
			xmax_in, ymax_in, xmax_out, ymax_out,
			f_radius, radius, 
			f_labels, f_embedded, f_sideways, 
			scale, line_width);
		}
	else if (f_bitmatrix) {

		char fname2[1000];

		strcpy(fname2, fname);
		replace_extension_with(fname2, "_bitmatrix");

		if (f_partitioned) {
			CG->draw_partitioned(fname2, xmax_in, ymax_in, xmax_out, ymax_out, 
				f_labels, scale, line_width, verbose_level);
			}
		else if (f_partition) {
			CG->draw_with_a_given_partition(fname2, xmax_in, ymax_in, xmax_out, ymax_out, part, nb_parts, 
				scale, line_width, verbose_level);
			}
		else if (f_Levi) {

			int f_partition_Levi = FALSE;
			int nb_row_parts = 0;
			int *row_part_first = NULL;
			int nb_col_parts = 0;
			int *col_part_first = NULL;

			if (f_Levi_discrete) {
				f_partition_Levi = TRUE;
				nb_row_parts = Levi_m;
				row_part_first = NEW_int(Levi_m + 1);
				nb_col_parts = Levi_n;
				col_part_first = NEW_int(Levi_n + 1);
				for (i = 0; i < Levi_m + 1; i++) {
					row_part_first[i] = i;
					}
				for (i = 0; i < Levi_n + 1; i++) {
					col_part_first[i] = i;
					}
				}
			cout << "before CG->draw_Levi" << endl;
			CG->draw_Levi(fname2, xmax_in, ymax_in, xmax_out, ymax_out, 
				f_partition_Levi, nb_row_parts, row_part_first, 
				nb_col_parts, col_part_first, 
				Levi_m, Levi_n, f_labels, 
				scale, line_width, verbose_level);
			}
		else {
			CG->draw(fname2, xmax_in, ymax_in, xmax_out, ymax_out, 
				scale, line_width, verbose_level);
			}
		}

	if (f_aut) {
		
		int *Adj;
		action *Aut;
		longinteger_object ago;

		cout << "computing automorphism group of the graph:" << endl;
		//Aut = create_automorphism_group_of_colored_graph_object(CG, verbose_level);


		Adj = NEW_int(CG->nb_points * CG->nb_points);
		int_vec_zero(Adj, CG->nb_points * CG->nb_points);
		for (i = 0; i < CG->nb_points; i++) {
			for (j = i + 1; j < CG->nb_points; j++) {
				if (CG->is_adjacent(i, j)) {
					Adj[i * CG->nb_points + j] = 1;
					}
				}
			}
		Aut = create_automorphism_group_of_graph(Adj, CG->nb_points, verbose_level);

		Aut->group_order(ago);	
		cout << "ago=" << ago << endl;

		int pt;
		schreier *Sch;
		sims *Stab;
		strong_generators *stab_gens;
		longinteger_object go;
		
		Aut->point_stabilizer_any_point(pt, 
			Sch, Stab, stab_gens, 
			verbose_level);
		Stab->group_order(go);

		cout << "The special point is " << pt << endl;
		cout << "The order of the point stabilizer is " << go << endl;
		cout << "Generators for the point stabilizer are:" << endl;
		stab_gens->print_generators();
		cout << endl;

		//cout << "All the elements in the stabilizer are:" << endl;
		//Stab->print_all_group_elements_as_permutations();
		
		FREE_OBJECT(stab_gens);
		FREE_OBJECT(Stab);
		FREE_OBJECT(Sch);
		FREE_int(Adj);
		}

	if (f_is_association_scheme) {

		int n = CG->nb_points;
		int *Adj;
	
		Adj = NEW_int(n * n);
		int_vec_zero(Adj, n * n);
		for (i = 0; i < n; i++) {
			for (j = i + 1; j < n; j++) {
				if (CG->is_adjacent(i, j)) {
					Adj[i * n + j] = 1;
					}
				}
			}
		for (i = 0; i < n * n; i++) {
			Adj[i] += 1;
			}
		for (i = 0; i < n; i++) {
			Adj[i * n + i] = 0;
			}
	
		int *Pijk;
		//int *colors;
		//int nb_colors;
		
		if (is_association_scheme(Adj, n, Pijk, 
			CG->point_color, CG->nb_colors, verbose_level)) {
			cout << "Is an association scheme" << endl;
			}
		else {
			cout << "Is NOT an association scheme" << endl;
			}

		FREE_int(Adj);
		
		}

	if (f_expand_power) {


		int n = CG->nb_points;
		int *Adj;
		int *A, *B;
		int e, c, k, diag, p;

		if (expand_power <= 1) {
			cout << "expand_power <= 1" << endl;
			exit(1);
			}

		Adj = NEW_int(n * n);
		A = NEW_int(n * n);
		B = NEW_int(n * n);
		int_vec_zero(Adj, n * n);
		for (i = 0; i < n; i++) {
			for (j = i + 1; j < n; j++) {
				if (CG->is_adjacent(i, j)) {
					Adj[i * n + j] = 1;
					Adj[j * n + i] = 1;
					}
				}
			}
		int_vec_copy(Adj, A, n * n);
		e = 1;

		while (e < expand_power) {

			for (i = 0; i < n; i++) {
				for (j = 0; j < n; j++) {
					c = 0;
					for (k = 0; k < n; k++) {
						c += Adj[i * n + k] * A[k * n + j];
						}
					B[i * n + j] = c;
					}
				}
			int_vec_copy(B, A, n * n);
			e++;


			}

		cout << "the " << expand_power << " power of the adjacency matrix is:" << endl;
		int_matrix_print(B, n, n);
		
		diag = B[0 * n + 0];
		for (i = 0; i < n; i++) {
			if (B[i * n + i] != diag) {
				cout << "diagonal is not constant" << endl;
				exit(1);
				}
			}

		for (i = 0; i < n; i++) {
			B[i * n + i] = 0;
			}

		cout << "after subtracting " << diag << " times the identity, the matrix is:" << endl;
		int_matrix_print(B, n, n);

		for (p = 0; p < n * n; p++) {
			if (Adj[p]) {
				break;
				}
			}

		c = B[p];
		if (c) {
			for (i = 0; i < n * n; i++) {
				if (Adj[i]) {
					if (B[i] != c) {
						cout << "B is not constant on the original graph" << endl;
						exit(1);
						}
					}
				}
			for (i = 0; i < n * n; i++) {
				if (Adj[i]) {
					B[i] = 0;
					}
				}
			}
		

		cout << "after subtracting " << c << " times the original graph, the matrix is:" << endl;
		int_matrix_print(B, n, n);


		int h;
		int *coeffs;
		colored_graph *CG_basis;

		coeffs = NEW_int(expand_power_nb_graphs + 2);
		CG_basis = NEW_OBJECTS(colored_graph, expand_power_nb_graphs);
		int_vec_zero(coeffs, expand_power_nb_graphs);
		coeffs[expand_power_nb_graphs] = c;
		coeffs[expand_power_nb_graphs + 1] = diag;

		for (h = 0; h < expand_power_nb_graphs; h++) {
			CG_basis[h].load(expand_power_graph_fname[h], verbose_level);
			
			if (CG_basis[h].nb_points != n) {
				cout << "the graph " << expand_power_graph_fname[h] << " has the wrong number of vertices" << endl;
				exit(1);
				}
			int *H;

			H = NEW_int(n * n);
			int_vec_zero(H, n * n);
			for (i = 0; i < n; i++) {
				for (j = i + 1; j < n; j++) {
					if (CG_basis[h].is_adjacent(i, j)) {
						H[i * n + j] = 1;
						H[j * n + i] = 1;
						}
					}
				}
			
			for (p = 0; p < n * n; p++) {
				if (H[p]) {
					break;
					}
				}

			coeffs[h] = B[p];
			if (coeffs[h]) {
				for (i = 0; i < n * n; i++) {
					if (H[i]) {
						if (B[i] != coeffs[h]) {
							cout << "B is not constant on the graph " << expand_power_graph_fname[h] << endl;
							exit(1);
							}
						}
					}
				for (i = 0; i < n * n; i++) {
					if (H[i]) {
						B[i] = 0;
						}
					}
				}
			cout << "after subtracting " << coeffs[h] << " times the graph " << expand_power_graph_fname[h] << ", the matrix is:" << endl;
			int_matrix_print(B, n, n);
			
			FREE_int(H);
			}

		cout << "coeffs=";
		int_vec_print(cout, coeffs, expand_power_nb_graphs + 2);
		cout << endl;
		
		FREE_int(Adj);
		FREE_int(A);
		FREE_int(B);
		}

	if (f_all_cliques || f_all_cocliques) {


		int *Adj;
		action *Aut;
		longinteger_object ago;

		cout << "computing automorphism group of the graph:" << endl;
		//Aut = create_automorphism_group_of_colored_graph_object(CG, verbose_level);


		Adj = NEW_int(CG->nb_points * CG->nb_points);
		int_vec_zero(Adj, CG->nb_points * CG->nb_points);
		for (i = 0; i < CG->nb_points; i++) {
			for (j = i + 1; j < CG->nb_points; j++) {
				if (CG->is_adjacent(i, j)) {
					Adj[i * CG->nb_points + j] = 1;
					}
				}
			}
		Aut = create_automorphism_group_of_graph(Adj, CG->nb_points, verbose_level);

		Aut->group_order(ago);	
		cout << "ago=" << ago << endl;

		action *Aut_on_points;
		int *points;
		
		Aut_on_points = NEW_OBJECT(action);
		points = NEW_int(CG->nb_points);
		for (i = 0; i < CG->nb_points; i++) {
			points[i] = i;
			}

		Aut_on_points->induced_action_by_restriction(*Aut, 
			TRUE /* f_induce_action */, Aut->Sims, 
			CG->nb_points /* nb_points */, points, verbose_level);
		
		Aut_on_points->group_order(ago);	
		cout << "ago on points = " << ago << endl;

		
		char prefix[1000];
		poset_classification *gen;
		int nb_orbits, depth;
		

		strcpy(prefix, fname);
		replace_extension_with(prefix, "_cliques");

		if (f_all_cliques) {
			compute_orbits_on_subsets(gen, 
				CG->nb_points /* target_depth */,
				prefix, 
				TRUE /* f_W */, FALSE /* f_w */,
				Aut_on_points, Aut_on_points, 
				Aut_on_points->Strong_gens, 
				early_test_function_cliques,
				CG, 
				NULL, 
				NULL, 
				verbose_level);
			}
		else {
			compute_orbits_on_subsets(gen, 
				CG->nb_points /* target_depth */,
				prefix, 
				TRUE /* f_W */, FALSE /* f_w */,
				Aut_on_points, Aut_on_points, 
				Aut_on_points->Strong_gens, 
				early_test_function_cocliques,
				CG, 
				NULL, 
				NULL, 
				verbose_level);
			}

		for (depth = 0; depth < CG->nb_points; depth++) {
			nb_orbits = gen->nb_orbits_at_level(depth);
			if (nb_orbits == 0) {
				depth--;
				break;
				}
			}

		if (f_all_cliques) {
			cout << "the largest cliques have size " << depth << endl;
			for (i = 0; i <= depth; i++) {
				nb_orbits = gen->nb_orbits_at_level(i);
				cout << setw(3) << i << " : " << setw(3) << nb_orbits << endl;
				}
			}
		else if (f_all_cocliques) {
			cout << "the largest cocliques have size " << depth << endl;
			for (i = 0; i <= depth; i++) {
				nb_orbits = gen->nb_orbits_at_level(i);
				cout << setw(3) << i << " : " << setw(3) << nb_orbits << endl;
				}
			}

		int *set;
		longinteger_object go, ol;
		longinteger_domain D;

		set = NEW_int(depth);
		nb_orbits = gen->nb_orbits_at_level(depth);

		cout << "orbit : representative : stabilizer order : orbit length" << endl;
		for (i = 0; i < nb_orbits; i++) {
			gen->get_set_by_level(depth, i, set);

			strong_generators *gens;
			gen->get_stabilizer_generators(gens,  
				depth, i, verbose_level);
			gens->group_order(go);
			D.integral_division_exact(ago, go, ol);


			cout << "Orbit " << i << " is the set ";
			int_vec_print(cout, set, depth);
			cout << " : " << go << " : " << ol << endl;
			cout << endl;

			
			}

		FREE_int(set);
		FREE_int(Adj);
		FREE_int(points);
		FREE_OBJECT(Aut);
		FREE_OBJECT(Aut_on_points);


		
		}


	else if (f_characteristic_polynomial) {
		
		characteristic_polynomial(CG, verbose_level);
		
		}
	
	FREE_OBJECT(CG);

	cout << "draw_colored_graph.out is done" << endl;
	the_end(t0);
	//the_end_quietly(t0);

}

void early_test_function_cliques(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	colored_graph *CG = (colored_graph *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}

	CG->early_test_func_for_clique_search(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);


	if (f_v) {
		cout << "early_test_function done" << endl;
		}
}

void early_test_function_cocliques(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	colored_graph *CG = (colored_graph *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}

	CG->early_test_func_for_coclique_search(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);


	if (f_v) {
		cout << "early_test_function done" << endl;
		}
}

void characteristic_polynomial(colored_graph *CG, int verbose_level)
{
	int q;
	int size;
	matrix M;
	int i, j, sq;
	finite_field Fq;


	//q = (1L << 59) - 55; // is prime according to https://primes.utm.edu/lists/2small/0bit.html

	q = (1L << 13) - 1;

	sq = sqrt(q);

	size = CG->nb_points;
	M.m_mn_n(size, size);
	for (i = 0; i < size; i++) {
		for (j = i + 1; j < size; j++) {
			if (CG->is_adjacent(i, j)) {
				M.m_iji(i, j, 1);
				M.m_iji(j, i, 1);
				}
			}
		}
	cout << "M=" << endl;
	cout << M << endl;

	Fq.init(q, verbose_level);

	domain d(q);
	with w(&d);
	
	// This part uses DISCRETA data structures:

	matrix M1, P, Pv, Q, Qv, S, T;
	
	M.elements_to_unipoly();
	M.X_times_id_minus_self();
	//M.minus_X_times_id();
	M1 = M;

	cout << "x * Id - M = " << endl << M << endl;
	M.smith_normal_form(P, Pv, Q, Qv, verbose_level);

	cout << "the Smith normal form is:" << endl;
	cout << M << endl;

	S.mult(P, Pv);
	cout << "P * Pv=" << endl << S << endl;

	S.mult(Q, Qv);
	cout << "Q * Qv=" << endl << S << endl;

	S.mult(P, M1);
	cout << "T.mult(S, Q):" << endl;
	T.mult(S, Q);
	cout << "T=" << endl << T << endl;


	unipoly charpoly;
	int deg;
	int l, lv, b, c;

	charpoly = M.s_ij(size - 1, size - 1);
		
	cout << "characteristic polynomial:" << charpoly << endl;
	deg = charpoly.degree();
	cout << "has degree " << deg << endl;


	l = charpoly.s_ii(deg);
	cout << "leading coefficient " << l << endl;
	lv = Fq.inverse(l);
	cout << "leading coefficient inverse " << lv << endl;
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		c = Fq.mult(b, lv);
		charpoly.m_ii(i, c);
		}
	for (i = 0; i <= deg; i++) {
		b = charpoly.s_ii(i);
		if (b > sq) {
			b -= q;
			}
		charpoly.m_ii(i, b);
		}
	cout << "monic minimum polynomial:" << charpoly << endl;

}



