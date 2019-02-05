// code_generator.C
//
// Anton Betten
//
// moved here from codes.C: May 18, 2009
//
// December 30, 2003

#include "codes.h"


using namespace orbiter;

// #############################################################################
// start of class code_generator
// #############################################################################


void code_generator::read_arguments(int argc, const char **argv)
{
	int i;
	int f_n = FALSE;
	int f_k = FALSE;
	int f_q = FALSE;
	int f_d = FALSE;
	int f_N = FALSE;
	
	
	gen->read_arguments(argc, argv, 0);

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-debug") == 0) {
			f_debug = TRUE;
			cout << "-debug " << endl;
			}
		else if (strcmp(argv[i], "-report") == 0) {
			f_report = TRUE;
			cout << "-report " << endl;
			}
		else if (strcmp(argv[i], "-read_data_file") == 0) {
			f_read_data_file = TRUE;
			fname_data_file = argv[++i];
			cout << "-read_data_file " << fname_data_file << endl;
			}
		else if (strcmp(argv[i], "-report_schreier_trees") == 0) {
			f_report_schreier_trees = TRUE;
			cout << "-report_schreier_trees " << endl;
			}
		else if (strcmp(argv[i], "-schreier_depth") == 0) {
			schreier_depth = atoi(argv[++i]);
			cout << "-schreier_depth " << schreier_depth << endl;
			}
		else if (strcmp(argv[i], "-n") == 0) {
			f_n = TRUE;
			n = atoi(argv[++i]);
			cout << "-n " << n << endl;
			}
		else if (strcmp(argv[i], "-k") == 0) {
			f_k = TRUE;
			f_linear = TRUE;
			k = atoi(argv[++i]);
			cout << "-k " << k << endl;
			}
		else if (strcmp(argv[i], "-nmk") == 0) {
			f_nmk = TRUE;
			f_linear = TRUE;
			nmk = atoi(argv[++i]);
			cout << "-nmk " << nmk << endl;
			}
		else if (strcmp(argv[i], "-N") == 0) {
			f_N = TRUE;
			f_nonlinear = TRUE;
			N = atoi(argv[++i]);
			cout << "-N " << N << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		else if (strcmp(argv[i], "-d") == 0) {
			f_d = TRUE;
			d = atoi(argv[++i]);
			cout << "-d " << d << endl;
			}
		else if (strcmp(argv[i], "-draw_poset") == 0) {
			f_draw_poset = TRUE;
			cout << "-draw_poset " << endl;
			}
		else if (strcmp(argv[i], "-print_data_structure") == 0) {
			f_print_data_structure = TRUE;
			cout << "-print_data_structure " << endl;
			}
		else if (strcmp(argv[i], "-list") == 0) {
			f_list = TRUE;
			cout << "-list" << endl;
			}
		else if (strcmp(argv[i], "-draw_schreier_trees") == 0) {
			gen->f_draw_schreier_trees = TRUE;
			strcpy(gen->schreier_tree_prefix, argv[++i]);
			gen->schreier_tree_xmax = atoi(argv[++i]);
			gen->schreier_tree_ymax = atoi(argv[++i]);
			gen->schreier_tree_f_circletext = atoi(argv[++i]);
			gen->schreier_tree_rad = atoi(argv[++i]);
			gen->schreier_tree_f_embedded = atoi(argv[++i]);
			gen->schreier_tree_f_sideways = atoi(argv[++i]);
			gen->schreier_tree_scale = atoi(argv[++i]) * 0.01;
			gen->schreier_tree_line_width = atoi(argv[++i]) * 0.01;
			cout << "-draw_schreier_trees " << gen->schreier_tree_prefix 
				<< " " << gen->schreier_tree_xmax 
				<< " " << gen->schreier_tree_ymax 
				<< " " << gen->schreier_tree_f_circletext 
				<< " " << gen->schreier_tree_f_embedded 
				<< " " << gen->schreier_tree_f_sideways 
				<< " " << gen->schreier_tree_scale 
				<< " " << gen->schreier_tree_line_width 
				<< endl;
			}
		else if (strcmp(argv[i], "-table_of_nodes") == 0) {
			f_table_of_nodes = TRUE;
			cout << "-table_of_nodes" << endl;
			}
		}
	
	if (f_linear && f_nonlinear) {
		cout << "We cannot be both linear and nonlinear" << endl;
		exit(1);
		}

	if (!f_n && !f_nmk) {
		cout << "Please use option -n <n> or -nmk <nmk> " << endl;
		exit(1);
		}
	if (!f_q) {
		cout << "Please use option -q <q> to specify q" << endl;
		exit(1);
		}
	if (!f_d) {
		cout << "Please use option -d <d> to specify d" << endl;
		exit(1);
		}
	if (!f_k && !f_nmk && !f_N) {
		cout << "Please use option -k <k> or -nmk <nmk> "
				"for linear codes or -N <N> for nonlinear codes" << endl;
		exit(1);
		}
	if (f_nmk) {
		if (f_n) {
			k = n - nmk;
			}
		else if (f_k) {
			n = k + nmk;
			}
		else {
			n = 200;
			k = n - nmk;
			}
		}
	cout << "n=" << n << endl;
	if (f_linear) {
		cout << "k=" << k << endl;
		}
	if (f_nonlinear) {
		cout << "N=" << N << endl;
		}
	cout << "q=" << q << endl;
	cout << "d=" << d << endl;
		
	f_irreducibility_test = TRUE;
	
	int p, h;
	
	is_prime_power(q, p, h);
	if (h > 1) {
		f_semilinear = TRUE;
		}
	else {
		f_semilinear = FALSE;
		}

}

code_generator::code_generator()
{
	null();
}

code_generator::~code_generator()
{
	freeself();
}

void code_generator::null()
{
	verbose_level = 0;
	f_report = FALSE;

	f_read_data_file = FALSE;
	fname_data_file = NULL;
	depth_completed = 0;

	f_report_schreier_trees = FALSE;
	f_nmk = FALSE;
	f_linear = FALSE;
	f_nonlinear = FALSE;
	Poset = NULL;
	gen = NULL;
	F = NULL;
	v1 = NULL;
	v2 = NULL;
	A = NULL;
	description = NULL;
	L = NULL;
	schreier_depth = 1000;
	f_list = FALSE;
	f_table_of_nodes = FALSE;
	f_use_invariant_subset_if_available = TRUE;
	f_debug = FALSE;
	f_draw_poset = FALSE;
	f_print_data_structure = FALSE;
	f_draw_schreier_trees = FALSE;
}

void code_generator::freeself()
{
	if (A) {
		FREE_OBJECT(A);
		}
	if (Poset) {
		FREE_OBJECT(Poset);
		}
	if (F) {
		FREE_OBJECT(F);
		}
	if (v1) {
		FREE_int(v1);
		}
	if (v2) {
		FREE_int(v2);
		}
	if (gen) {
		FREE_OBJECT(gen);
		}
	if (description) {
		FREE_OBJECT(description);
		}
	if (L) {
		FREE_OBJECT(L);
		}
	null();
}

void code_generator::init(int argc, const char **argv)
{
	F = NEW_OBJECT(finite_field);
	A = NEW_OBJECT(action);
	Poset = NEW_OBJECT(poset);
	gen = NEW_OBJECT(poset_classification);
	int f_basis = TRUE;
	

	read_arguments(argc, argv);
	
	//int verbose_level = gen->verbose_level;
	
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "code_generator::init" << endl;
		}


	if (f_v) {
		cout << "code_generator::init initializing "
				"finite field of order " << q << endl;
		}
	F->init(q, 0);
	if (f_v) {
		cout << "code_generator::init initializing "
				"finite field of order " << q << " done" << endl;
		}



	if (f_linear) {
		if (f_nmk) {
			sprintf(directory_path, "CODES_NMK_%d_%d_%d/", nmk, q, d);
			sprintf(prefix, "codes_nmk_%d_%d_%d", nmk, q, d);
			}
		else {
			sprintf(directory_path, "CODES_%d_%d_%d_%d/", n, k, q, d);
			sprintf(prefix, "codes_%d_%d_%d_%d", n, k, q, d);
			}
		}
	else if (f_nonlinear) {
		sprintf(directory_path, "NONLINEAR_CODES_%d_%d_%d/", n, q, d);
		sprintf(prefix, "nonlinear_codes_%d_%d_%d", n, q, d);
		}
	if (strlen(directory_path)) {
		char cmd[1000];

		sprintf(cmd, "mkdir %s", directory_path);
		system(cmd);
		}
	sprintf(gen->fname_base, "%s%s", directory_path, prefix);
	
	
	if (f_linear) {
		nmk = n - k;

		if (f_v) {
			cout << "code_generator::init calling "
					"init_projective_group, dimension = " << nmk << endl;
			}

		v1 = NEW_int(nmk);
		v2 = NEW_int(nmk);
		vector_ge *nice_gens;

		A->init_projective_group(nmk, F, 
			f_semilinear, 
			f_basis, 
			nice_gens,
			verbose_level - 2);
		FREE_OBJECT(nice_gens);
	
		if (f_v) {
			cout << "code_generator::init finished with "
					"init_projective_group" << endl;
			}

		gen->depth = n;
		rc.init(F, nmk, n, d);
		Strong_gens = A->Strong_gens;
		}
	else if (f_nonlinear) {


		description = NEW_OBJECT(linear_group_description);
		L = NEW_OBJECT(linear_group);
		description->null();
		description->f_affine = TRUE;
		description->n = n;
		description->input_q = q;
		description->F = F;
		description->f_monomial_group = TRUE;
		description->f_semilinear = FALSE;
		description->f_special = FALSE;
		
		L->init(description, verbose_level);
		
		longinteger_object go;

		L->Strong_gens->group_order(go);
		cout << "created strong generators for a group of "
				"order " << go << endl;
		L->Strong_gens->print_generators();

		A = L->A2;
		Strong_gens = L->Strong_gens;
		gen->depth = N;
		}

	if (f_v) {
		cout << "code_generator::init degree = " << A->degree << endl;
		cout << "code_generator::init depth = " << gen->depth << endl;
		}
	

	
	
	if (f_v) {
		cout << "code_generator::init group set up, "
				"calling gen->init" << endl;
		cout << "A->f_has_strong_generators="
				<< A->f_has_strong_generators << endl;
		}
	
	Poset->init_subset_lattice(A, A,
			Strong_gens,
			verbose_level);


	int independence_value = d - 1;

	Poset->add_independence_condition(
			independence_value,
			verbose_level);

	gen->init(Poset, gen->depth /* sz */, verbose_level);



#if 0
	if (FALSE && gen->A->degree < 1000) {
		cout << "the elements of PG(" << n - k - 1 << ","
				<< F->q << ") are:" << endl;
		display_all_PG_elements(n - k - 1, *F);
		}
#endif

	gen->f_print_function = FALSE;
	gen->print_function = print_code;
	gen->print_function_data = this;
	
	int nb_nodes = ONE_MILLION;

	if (f_v) {
		cout << "code_generator::init group set up, "
				"calling gen->init_poset_orbit_node" << endl;
		}

	gen->init_poset_orbit_node(nb_nodes, verbose_level - 1);

	if (f_v) {
		cout << "code_generator::init group set up, "
				"calling gen->root[0].init_root_node" << endl;
		}

	gen->root[0].init_root_node(gen, gen->verbose_level - 2);
	if (f_read_data_file) {
		if (f_v) {
			cout << "code_generator::init reading data file "
					<< fname_data_file << endl;
			}

		gen->read_data_file(depth_completed,
				fname_data_file, verbose_level - 1);
		if (f_v) {
			cout << "code_generator::init after reading data file "
					<< fname_data_file << " depth_completed = "
					<< depth_completed << endl;
			}
		if (f_v) {
			cout << "code_generator::init before "
					"gen->recreate_schreier_vectors_up_to_level" << endl;
			}
		gen->recreate_schreier_vectors_up_to_level(depth_completed - 1,
			verbose_level - 1);
		if (f_v) {
			cout << "code_generator::init after "
					"gen->recreate_schreier_vectors_up_to_level" << endl;
			}

	}
	
}


void code_generator::main(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int depth;
	int f_embedded = TRUE;
	int f_sideways = FALSE;

	if (f_v) {
		cout << "code_generator::main" << endl;
		}
	if (f_read_data_file) {
		int target_depth;
		if (gen->f_max_depth) {
			target_depth = gen->max_depth;
			}
		else {
			target_depth = gen->depth;
			}
		depth = gen->compute_orbits(depth_completed, target_depth,
				verbose_level);
	} else {
		depth = gen->main(t0,
			schreier_depth,
			f_use_invariant_subset_if_available,
			f_debug,
			verbose_level);
	}

	if (f_table_of_nodes) {
		int *Table;
		int nb_rows, nb_cols;

		gen->get_table_of_nodes(Table, 
			nb_rows, nb_cols, 0 /*verbose_level*/);
	
		int_matrix_write_csv("data.csv", Table, nb_rows, nb_cols);


		FREE_int(Table);
		}

	if (f_list) {

		{
		spreadsheet *Sp;
		gen->make_spreadsheet_of_orbit_reps(Sp, depth);
		char fname_csv[1000];
		sprintf(fname_csv, "orbits_%d.csv", depth);
		Sp->save(fname_csv, verbose_level);
		delete Sp;
		}

#if 1
		int f_show_orbit_decomposition = TRUE;
		int f_show_stab = TRUE;
		int f_save_stab = TRUE;
		int f_show_whole_orbit = FALSE;
		
		gen->list_all_orbits_at_level(depth, 
			TRUE, 
			print_code, 
			this, 
			f_show_orbit_decomposition, 
			f_show_stab, 
			f_save_stab, 
			f_show_whole_orbit);

#if 0
		int d;
		for (d = 0; d < 3; d++) {
			gen->print_schreier_vectors_at_depth(d, verbose_level);
			}
#endif
#endif
		}


	cout << "preparing level spreadsheet" << endl;
	{
	spreadsheet *Sp;
	gen->make_spreadsheet_of_level_info(
			Sp, depth, verbose_level);
	char fname_csv[1000];
	sprintf(fname_csv, "levels_%d.csv", depth);
	Sp->save(fname_csv, verbose_level);
	delete Sp;
	}
	cout << "preparing orbit spreadsheet" << endl;
	{
	spreadsheet *Sp;
	gen->make_spreadsheet_of_orbit_reps(
			Sp, depth);
	char fname_csv[1000];
	sprintf(fname_csv, "orbits_%d.csv",
			depth);
	Sp->save(fname_csv, verbose_level);
	delete Sp;
	}
	cout << "preparing orbit spreadsheet done" << endl;



	if (f_draw_poset) {
		gen->draw_poset(gen->fname_base, depth, 
			0 /* data1 */, f_embedded, f_sideways, 
			verbose_level);
		}
	if (f_print_data_structure) {
		gen->print_data_structure_tex(depth, verbose_level);
		}
	if (f_report_schreier_trees) {
		char fname_base[1000];
		char fname_report[1000];
		if (f_linear) {
			if (f_nmk) {
				sprintf(fname_base, "codes_linear_nmk%d_q%d_d%d", nmk, q, d);
				}
			else {
				sprintf(fname_base, "codes_linear_n%d_k%d_q%d_d%d", n, k, q, d);
				}
			}
		else if (f_nonlinear) {
			sprintf(fname_base, "codes_nonlinear_n%d_k%d_d%d", n, k, d);
			}
		sprintf(fname_report, "%s.txt", fname_base);
		{
		ofstream fp(fname_report);

		gen->report_schreier_trees(fp, verbose_level);
		}
	}
	if (f_report) {
		char fname_base[1000];
		char fname_report[1000];
		char title[10000];
		char author[10000];

		author[0] = 0;
		//sprintf(author, "");


		if (f_linear) {
			if (f_nmk) {
				sprintf(title, "Classification of Optimal Linear Codes by Redundancy");
				sprintf(fname_base, "codes_linear_nmk%d_q%d_d%d", nmk, q, d);
				}
			else {
				sprintf(title, "Classification of Optimal Linear Codes");
				sprintf(fname_base, "codes_linear_n%d_k%d_q%d_d%d", n, k, q, d);
				}
			}
		else if (f_nonlinear) {
			sprintf(title, "Classification of Optimal Nonlinear Codes");
			sprintf(fname_base, "codes_nonlinear_n%d_k%d_d%d", n, k, d);
			}

		sprintf(fname_report, "%s.tex", fname_base);
		{
		ofstream fp(fname_report);


		latex_head(fp,
			FALSE /* f_book */,
			TRUE /* f_title */,
			title, author,
			FALSE /*f_toc */,
			FALSE /* f_landscape */,
			FALSE /* f_12pt */,
			TRUE /*f_enlarged_page */,
			TRUE /* f_pagenumbers*/,
			NULL /* extra_praeamble */);

		A->report(fp);

		gen->report(fp);

		latex_foot(fp);

		} // close fp

	}
	if (f_v) {
		cout << "code_generator::main done" << endl;
		}
}

void code_generator::print(ostream &ost, int len, int *S)
{
	int N, j;
	int *codewords;

	if (len == 0) {
		return;
		}

	if (f_linear) {
		ost << "generator matrix:" << endl;
		for (j = 0; j < len; j++) {
			F->PG_element_unrank_modified(
				rc.M1 + j, len /* stride */, nmk /* len */,
				S[j]);
			}
		print_integer_matrix(ost, rc.M1, nmk, len);

		int_matrix_print_tex(ost, rc.M1, nmk, len);

		N = i_power_j(F->q, nmk);
		codewords = NEW_int(N);
		F->codewords_affine(len, nmk,
			rc.M1,
			codewords,
			verbose_level);

		ost << "The " << N << " codewords are: ";
		int_vec_print(ost, codewords, N);
		ost << endl;

		FREE_int(codewords);


		F->compute_and_print_projective_weights(ost, rc.M1, len, nmk);

		if (len > k) {
			int *A, *B;
			int k1;

			ost << "computing the dual code:" << endl;

			A = NEW_int(n * n);
			int_vec_copy(rc.M1, A, nmk * len);
			F->perp_standard(len, nmk, A, 0 /* verbose_level*/);
			B = A + nmk * len;
			print_integer_matrix(ost, B, len - nmk, len);

			int_matrix_print_tex(ost, B, len - nmk, len);


			k1 = len - nmk;

			N = i_power_j(F->q, k1);
			codewords = NEW_int(N);
			F->codewords_affine(len, k1,
				B,
				codewords,
				verbose_level);

			ost << "The " << N << " codewords are: ";
			int_vec_print(ost, codewords, N);
			ost << endl;

			FREE_int(codewords);

			ost << "before F->compute_and_print_"
					"projective_weights" << endl;
			F->compute_and_print_projective_weights(
					ost, B, len, len - nmk);
			}
		}
	else if (f_nonlinear) {
		ost << "print nonlinear not yet implemented" << endl;
		}
}

#if 0
void code_generator::early_test_func_by_using_group(
	int *S, int len, 
	int *candidates, int nb_candidates, 
	int *good_candidates, int &nb_good_candidates, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "code_generator::early_test_func_by_using_group" << endl;
		}

	if (f_vv) {
		cout << "S=";
		int_vec_print(cout, S, len);
		cout << " testing " << nb_candidates << " candidates" << endl;
		//int_vec_print(cout, candidates, nb_candidates);
		//cout << endl;
		}

	int i, j, node, f, l, pt, nb_good_orbits;
	poset_orbit_node *O;
	int f_orbit_is_good;
	int s, a;

	node = gen->find_poset_orbit_node_for_set(len, S,
		FALSE /* f_tolerant */, 0);
	O = gen->root + node;

	if (f_v) {
		cout << "code_generator::early_test_func_by_using_group for ";
		O->print_set(gen);
		cout << endl;
		}

	schreier Schreier;

	Schreier.init(gen->Poset->A2);

	Schreier.init_generators_by_hdl(
		O->nb_strong_generators, O->hdl_strong_generators, 0);

	Schreier.orbits_on_invariant_subset_fast(
		nb_candidates, candidates, 
		0/*verbose_level*/);

	if (f_v) {
		cout << "code_generator::early_test_func_by_"
				"using_group after Schreier.compute_all_orbits_"
				"on_invariant_subset, we found "
		<< Schreier.nb_orbits << " orbits" << endl;
		}
	nb_good_candidates = 0;
	nb_good_orbits = 0;
	for (i = 0; i < Schreier.nb_orbits; i++) {
		f = Schreier.orbit_first[i];
		l = Schreier.orbit_len[i];
		pt = Schreier.orbit[f];
		S[len] = pt;
		if (f_linear) {
			if (rc.check_rank_last_two_are_fixed(len + 1, 
				S, verbose_level - 1)) {
				f_orbit_is_good = TRUE;
				}
			else {
				f_orbit_is_good = FALSE;
				}
			}
		else {
			f_orbit_is_good = TRUE;
			for (s = 0; s < len; s++) {
				a = S[s];
				if (Hamming_distance(a, pt) < d) {
					f_orbit_is_good = FALSE;
					break;
					}
				}
			}
		if (f_orbit_is_good) {
			for (j = 0; j < l; j++) {
				pt = Schreier.orbit[f + j];
				good_candidates[nb_good_candidates++] = pt;
				}	
			nb_good_orbits++;
			}
		}

	int_vec_heapsort(good_candidates, nb_good_candidates);
	if (f_v) {
		cout << "code_generator::early_test_func_by_using_group "
			"after Schreier.compute_all_orbits_on_invariant_subset, "
			"we found "
			<< nb_good_candidates << " good candidates in " 
			<< nb_good_orbits << " good orbits" << endl;
		}
}
#endif

int code_generator::Hamming_distance(int a, int b)
{
	int f_v = TRUE;
	int d = 0;
	int i;

	if (f_v) {
		cout << "code_generator::Hamming_distance "
				"a=" << a << " b=" << b << endl;
		}
	if (f_nonlinear) {
		if (q == 2) {
			d = Hamming_distance_binary(a, b, n);
			}
		else {
			cout << "code_generator::Hamming_distance "
					"f_nonlinear and q != 2" << endl;
			exit(1);
			}
		}
	if (f_linear) {
		F->PG_element_unrank_modified(
			v1, 1 /* stride */, nmk /* len */, 
			a);
		F->PG_element_unrank_modified(
			v2, 1 /* stride */, nmk /* len */, 
			b);
		for (i = 0; i < nmk; i++) {
			if (v1[i] != v2[i]) {
				d++;
				}
			}
		}
	if (f_v) {
		cout << "code_generator::Hamming_distance "
				"a=" << a << " b=" << b << " d=" << d << endl;
		}
	return d;
}


// #############################################################################
// callback functions
// #############################################################################

void print_code(ostream &ost, int len, int *S, void *data)
{
	code_generator *cg = (code_generator *) data;
	
	cg->print(ost, len, S);
}



