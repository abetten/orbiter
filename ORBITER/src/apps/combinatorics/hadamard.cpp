// hadamard.C
// 
// Anton Betten
// December 9, 2014
//
//

#include "orbiter.h"

using namespace orbiter;

typedef class hadamard hadamard;


//! Classification of Hadamard matrices




class hadamard {

public:
	int n;
	int N, N2;
	int bitvector_length;
	uchar *bitvector_adjacency;
	colored_graph *CG;

	action *A;

	int *v;
	
	poset_classification *gen;
	int nb_orbits;

	void init(int n, int f_draw, int verbose_level, int verbose_level_clique);
	int clique_test(int *set, int sz);
	void early_test_func(int *S, int len, 
		int *candidates, int nb_candidates, 
		int *good_candidates, int &nb_good_candidates, 
		int verbose_level);
};



// global data:

int t0; // the system time when the program started

int dot_product(int a, int b, int n);
void early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level);

int main(int argc, char **argv)
{
	int i;
	int verbose_level = 0;
	int verbose_level_clique = 0;
	int f_n = FALSE;
	int n = 0;
	int f_draw = FALSE;
	
 	t0 = os_ticks();
	
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-v_clique") == 0) {
			verbose_level_clique = atoi(argv[++i]);
			cout << "-v_clique " << verbose_level_clique << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-draw") == 0) {
			f_draw = TRUE;
			cout << "-draw " << endl;
			}
		}
	if (!f_n) {
		cout << "please use option -n <n> to specify n" << endl;
		exit(1);
		}

	hadamard H;

	H.init(n, f_draw, verbose_level, verbose_level_clique);

	the_end(t0);

}

void hadamard::init(int n, int f_draw,
		int verbose_level, int verbose_level_clique)
{
	int f_v = (verbose_level = 1);
	int i, j, k, d, cnt, cnt1;

	if (n > (int)sizeof(int) * 8) {
		cout << "n > sizeof(uint) * 8" << endl;
		exit(1);
		}

	hadamard::n = n;

	v = NEW_int(n);
	
	N = (1 << n);

	if (f_v) {
		cout << "n =" << n << endl;
		cout << "N =" << N << endl;
		}

	N2 = (N * (N - 1)) >> 1;

	if (f_v) {
		cout << "N2 = (N * (N - 1)) >> 1 =" << N2 << endl;
		cout << "list of points:" << endl;
		for (i = 0; i < N; i++) {
			AG_element_unrank(2, v, 1, n, i);
			cout << i << " : ";
			for (j = 0; j < n; j++) {
				if (v[j]) {
					cout << "+";
					}
				else {
					cout << "-";
					}
				}
			//int_vec_print(cout, v, n);
			cout << endl;
			}
		}


	bitvector_length = (N2 + 7) >> 3;

	bitvector_adjacency = NEW_uchar(bitvector_length);

	if (f_v) {
		cout << "after allocating adjacency bitvector" << endl;
		cout << "computing adjacency matrix:" << endl;
		}
	k = 0;
	cnt = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {

			d = dot_product(i, j, n);

			if (FALSE) {
				cout << "dotproduct i=" << i << " j=" << j << " is " << d << endl;
				}

			if (d == 0) {
				bitvector_m_ii(bitvector_adjacency, k, 1);
				cnt++;
				}
			else {
				bitvector_m_ii(bitvector_adjacency, k, 0);
				}
			k++;
			if ((k & ((1 << 13) - 1)) == 0) {
				cout << "i=" << i << " j=" << j << " k=" << k << " / " << N2 << endl;
				}
			}
		}
	cout << "We have " << cnt << " edges in the graph" << endl;


#if 0
	// test the graph:

	k = 0;
	cnt1 = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {

			d = dot_product(i, j, n);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				cnt1++;
				}
			if (bitvector_s_i(bitvector_adjacency, k) && d) {
				cout << "something is wrong in entry i=" << i << " j=" << j << endl;
				cout << "dotproduct i=" << i << " j=" << j << " is " << d << endl;
				cout << "bitvector_s_i(bitvector_adjacency, k)="
						<< bitvector_s_i(bitvector_adjacency, k) << endl;
				exit(1);
				}
			k++;
			}
		}
	cout << "We found " << cnt1 << " edges in the graph" << endl;

	if (cnt1 != cnt) {
		cout << "cnt1 != cnt, something is wrong" << endl;
		cout << "cnt=" << cnt << endl;
		cout << "cnt1=" << cnt1 << endl;
		exit(1);
		}
#endif

	{
	colored_graph *CG;
	char fname[1000];

	CG = NEW_OBJECT(colored_graph);
	int *color;

	color = NEW_int(N);
	int_vec_zero(color, N);

	CG->init(N, 1, color, bitvector_adjacency, FALSE, verbose_level);

	sprintf(fname, "Hadamard_graph_%d.colored_graph", n);

	CG->save(fname, verbose_level);


	FREE_int(color);
	FREE_OBJECT(CG);
	}




	CG = NEW_OBJECT(colored_graph);

	if (f_v) {
		cout << "initializing colored graph" << endl;
		}

	int *color;

	color = NEW_int(N);
	int_vec_zero(color, N);

	CG->init(N, 1, color, bitvector_adjacency, FALSE, verbose_level);

	if (f_v) {
		cout << "initializing colored graph done" << endl;
		}

	char fname_graph[1000];
	
	sprintf(fname_graph, "Hadamard_graph_%d.magma", n);
	CG->export_to_magma(fname_graph, 1);

	{
	int *color_graph;

	color_graph = NEW_int(N * N);
	int_vec_zero(color_graph, N * N);
	k = 0;
	cnt1 = 0;
	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {
			if (bitvector_s_i(bitvector_adjacency, k)) {
				cnt1++;
				color_graph[i * N + j] = 2;
				color_graph[j * N + i] = 2;
				}
			else {
				color_graph[i * N + j] = 1;
				color_graph[j * N + i] = 1;
				}
			k++;
			}
		}
	cout << "We found " << cnt1 << " edges in the graph" << endl;

	if (cnt1 != cnt) {
		cout << "cnt1 != cnt, something is wrong" << endl;
		cout << "cnt=" << cnt << endl;
		cout << "cnt1=" << cnt1 << endl;
		exit(1);
		}
	
	cout << "color graph:" << endl;
	if (N < 30) {
		int_matrix_print(color_graph, N, N);
		}
	else {
		cout << "Too big to print" << endl;
		}
	
#if 0
	int *Pijk;
	int *colors;
	int nb_colors;

	is_association_scheme(color_graph, N, Pijk, 
		colors, nb_colors, verbose_level);

	cout << "number of colors = " << nb_colors << endl;
	cout << "colors: ";
	int_vec_print(cout, colors, nb_colors);
	cout << endl;
	cout << "Pijk:" << endl;
	for (i = 0; i < nb_colors; i++) {
		cout << "i=" << i << ":" << endl;
		int_matrix_print(Pijk + i * nb_colors * nb_colors, nb_colors, nb_colors);
		}
	FREE_int(Pijk);
	FREE_int(colors);
#endif

	FREE_int(color_graph);
	}

	if (f_draw) {
		if (f_v) {
			cout << "drawing adjacency matrix" << endl;
			}

		char fname_base[1000];
		int xmax_in = ONE_MILLION;
		int ymax_in = ONE_MILLION;
		int xmax_out = 500000;
		int ymax_out = 500000;
		double scale = 0.4;
		double line_width = 0.5;
	
		sprintf(fname_base, "Hadamard_graph_%d", n);


		//CG->draw_partitioned(fname_base,
		//xmax_in, ymax_in, xmax_out, ymax_out, verbose_level);
		CG->draw(fname_base, xmax_in, ymax_in, xmax_out, ymax_out, 
			scale, line_width, verbose_level);

		if (f_v) {
			cout << "drawing adjacency matrix done" << endl;
			}
		}


	if (f_v) {
		cout << "computing automorphism group of "
				"uncolored graph:" << endl;
		}

	nauty_interface Nauty;


	A = Nauty.create_automorphism_group_of_graph_bitvec(
		CG->nb_points, bitvector_adjacency, 
		verbose_level);
	
	longinteger_object go;
	A->group_order(go);
	if (f_v) {
		cout << "computing automorphism group of "
				"uncolored graph done, group order = " << go << endl;
		}

	char fname_group[1000];
	
	
	sprintf(fname_group, "Hadamard_group_%d.magma", n);
	A->Strong_gens->export_permutation_group_to_magma(
			fname_group, 1 /* verbose_level */);

	char prefix[1000];
	sprintf(prefix, "./had_%d", n);

	if (f_v) {
		cout << "Starting the clique finder, "
				"target_depth = " << n << " prefix=" << prefix << endl;
		}

	poset *Poset;
	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);
	Poset->add_testing_without_group(
			early_test_function,
			this /* void *data */,
			verbose_level);


	compute_orbits_on_subsets(gen, 
		n /* target_depth */,
		prefix, 
		TRUE /* f_W */, FALSE /* f_w */,
		Poset,
		verbose_level_clique);

	nb_orbits = gen->nb_orbits_at_level(n);

	int h, a, c;
	int *set;
	int *H;
	int *Ht;
	int *M;
	
	set = NEW_int(n);
	H = NEW_int(n * n);
	Ht = NEW_int(n * n);
	M = NEW_int(n * n);
	for (h = 0; h < nb_orbits; h++) {
		gen->get_set_by_level(n, h, set);
		cout << "Orbit " << h << " is the set ";
		int_vec_print(cout, set, n);
		cout << endl;

		
		if (clique_test(set, n)) {
			cout << "is a clique" << endl;
			}
		else {
			cout << "is not a cliqe, this should not happen" << endl;
			exit(1);
			}

		for (j = 0; j < n; j++) {
			a = set[j];
			for (i = 0; i < n; i++) {
				if (a % 2) {
					H[i * n + j] = 1;
					}
				else {
					H[i * n + j] = -1;
					}
				a >>= 1;
				}
			}
		cout << "The hadamard matrix " << h << " is:" << endl;
		int_matrix_print(H, n, n);
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				a = H[i * n + j];
				Ht[j * n + i] = a;
				}
			}

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				c = 0;
				for (k = 0; k < n; k++) {
					c += H[i * n + k] * Ht[k * n + j];
					}
				M[i * n + j] = c;
				}
			}
		cout << "The matrix H * H^t is:" << endl;
		int_matrix_print(M, n, n);
		}
}

int hadamard::clique_test(int *set, int sz)
{
	int i, j, a, b, idx;

	for (i = 0; i < n; i++) {
		a = set[i];
		for (j = i + 1; j < n; j++) {
			b = set[j];
			idx = ij2k(a, b, N);
			if (bitvector_s_i(bitvector_adjacency, idx)) {
				//cout << "pair (" << i << "," << j << ") vertices " << a << " and " << b << " are adjacent" << endl;
				}
			else {
				//cout << "pair (" << i << "," << j << ") vertices " << a << " and " << b << " are NOT adjacent" << endl;
				return FALSE;
				}
			}
		}
	return TRUE;
}

void hadamard::early_test_func(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int j, a, pt;

	if (f_v) {
		cout << "hadamard::early_test_func checking set ";
		print_set(cout, len, S);
		cout << endl;
		cout << "candidate set of size " << nb_candidates << ":" << endl;
		int_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}
	if (len == 0) {
		nb_good_candidates = nb_candidates;
		int_vec_copy(candidates, good_candidates, nb_candidates);
		return;
		}

	pt = S[len - 1];

	nb_good_candidates = 0;
	for (j = 0; j < nb_candidates; j++) {
		a = candidates[j];
		
		if (CG->is_adjacent(pt, a)) {
			good_candidates[nb_good_candidates++] = a;
			}
		} // next j
	
}


int dot_product(int a, int b, int n)
{
	int i, c, aa, bb;

	c = 0;
	for (i = 0; i < n; i++) {
		aa = a % 2;
		bb = b % 2;
		if (aa == bb) {
			c++;
			}
		else {
			c--;
			}
		a >>= 1;
		b >>= 1;
		}
	return c;
}

void early_test_function(int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	void *data, int verbose_level)
{
	hadamard *H = (hadamard *) data;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "early_test_function for set ";
		print_set(cout, len, S);
		cout << endl;
		}
	H->early_test_func(S, len, 
		candidates, nb_candidates, 
		good_candidates, nb_good_candidates, 
		verbose_level - 2);
	if (f_v) {
		cout << "early_test_function done" << endl;
		}
}


