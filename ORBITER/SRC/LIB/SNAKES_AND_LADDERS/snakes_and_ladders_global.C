// snakes_and_ladders_global.C
//
// Anton Betten
//
// October 12, 2013

#include "orbiter.h"

#define MY_OWN_BUFSIZE 1000000

void read_orbit_rep_and_candidates_from_files_and_process(action *A, BYTE *prefix, 
	INT level, INT orbit_at_level, INT level_of_candidates_file, 
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level), 
	void *early_test_func_callback_data, 
	INT *&starter,
	INT &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens, 
	INT *&candidates,
	INT &nb_candidates,
	INT &nb_cases, 
	INT verbose_level)
// A needs to be the base action
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *candidates1;
	INT nb_candidates1;
	INT h; //, i;

	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files_and_process" << endl;
		}

	read_orbit_rep_and_candidates_from_files(A, prefix, 
		level, orbit_at_level, level_of_candidates_file, 
		starter,
		starter_sz,
		Stab,
		Strong_gens, 
		candidates1,
		nb_candidates1,
		nb_cases, 
		verbose_level - 1);

	for (h = level_of_candidates_file; h < level; h++) {

		INT *candidates2;
		INT nb_candidates2;

		if (f_vv) {
			cout << "read_orbit_rep_and_candidates_from_files_and_process testing candidates at level " << h << " number of candidates = " << nb_candidates1 << endl;
			}
		candidates2 = NEW_INT(nb_candidates1);
		
		(*early_test_func_callback)(starter, h + 1, 
			candidates1, nb_candidates1, 
			candidates2, nb_candidates2, 
			early_test_func_callback_data, 0 /*verbose_level - 1*/);
		
		if (f_vv) {
			cout << "read_orbit_rep_and_candidates_from_files_and_process number of candidates at level " << h + 1 << " reduced from " << nb_candidates1 << " to " << nb_candidates2 << " by " << nb_candidates1 - nb_candidates2 << endl;
			}
	
		INT_vec_copy(candidates2, candidates1, nb_candidates2);
		nb_candidates1 = nb_candidates2;

		FREE_INT(candidates2);
		}

	candidates = candidates1;
	nb_candidates = nb_candidates1;
	
	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files_and_process done" << endl;
		}
}

void read_orbit_rep_and_candidates_from_files(action *A, BYTE *prefix, 
	INT level, INT orbit_at_level, INT level_of_candidates_file, 
	INT *&starter,
	INT &starter_sz,
	sims *&Stab,
	strong_generators *&Strong_gens, 
	INT *&candidates,
	INT &nb_candidates,
	INT &nb_cases, 
	INT verbose_level)
// A needs to be the base action
{
	INT f_v = (verbose_level >= 1);
	INT orbit_at_candidate_level = -1;


	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files" << endl;
		}

	{
	candidates = NULL;
	//longinteger_object stab_go;

	BYTE fname1[1000];
	BYTE fname2[1000];
	BYTE fname3[1000];

	sprintf(fname1, "%s_lvl_%ld", prefix, level);
	sprintf(fname3, "%s_lvl_%ld", prefix, level_of_candidates_file);

	sprintf(fname2, "%s_lvl_%ld_candidates.bin", prefix, level_of_candidates_file);
	
	A->read_set_and_stabilizer(fname1, 
		orbit_at_level, starter, starter_sz, Stab, 
		Strong_gens, 
		nb_cases, 
		verbose_level);



	//Stab->group_order(stab_go);

	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files Read starter " << orbit_at_level << " / " << nb_cases << " : ";
		INT_vec_print(cout, starter, starter_sz);
		cout << endl;
		//cout << "read_orbit_rep_and_candidates_from_files Group order=" << stab_go << endl;
		}

	if (level == level_of_candidates_file) {
		orbit_at_candidate_level = orbit_at_level;
		}
	else {
		// level_of_candidates_file < level
		// Now, we need to find out the orbit representative at level_of_candidates_file
		// that matches with the prefix of starter
		// so that we can retrieve it's set of candidates.
		// Once we have the candidates for the prefix, we run it through the 
		// test function to find the candidate set of starter as a subset 
		// of this set. 
		if (file_size(fname3) <= 0) {
			cout << "read_orbit_rep_and_candidates_from_files file " << fname3 << " does not exist" << endl;
			exit(1);
			}
		ifstream f(fname3);
		INT a, i, cnt;
		INT *S;
		BYTE buf[MY_OWN_BUFSIZE];
		INT len, str_len;
		BYTE *p_buf;

		S = NEW_INT(level_of_candidates_file);
	
		cnt = 0;
		f.getline(buf, MY_OWN_BUFSIZE, '\n'); // skip the first line
		while (TRUE) {
			if (f.eof()) {
				break;
				}
			f.getline(buf, MY_OWN_BUFSIZE, '\n');
			//cout << "Read line " << cnt << "='" << buf << "'" << endl;
			str_len = strlen(buf);
			if (str_len == 0) {
				cout << "read_orbit_rep_and_candidates_from_files str_len == 0" << endl;
				exit(1);
				}
		
			// check for comment line:
			if (buf[0] == '#')
				continue;
			
			p_buf = buf;
			s_scan_int(&p_buf, &a);
			if (a == -1) {
				break;
				}
			len = a;
			if (a != level_of_candidates_file) {
				cout << "a != level_of_candidates_file" << endl;
				cout << "a=" << a << endl;
				cout << "level_of_candidates_file=" << level_of_candidates_file << endl;
				exit(1);
				}
			for (i = 0; i < len; i++) {
				s_scan_int(&p_buf, &S[i]);
				}
			for (i = 0; i < level_of_candidates_file; i++) {
				if (S[i] != starter[i]) {
					break;
					}
				}
			if (i == level_of_candidates_file) {
				// We found the representative that matches the prefix:
				orbit_at_candidate_level = cnt;
				break;
				}
			else {
				cnt++;
				}
			}
		FREE_INT(S);
		}
	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files Found starter, orbit_at_candidate_level=" << orbit_at_candidate_level << endl;
		}
	

	// read the set of candidates from the binary file:

	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files before generator_read_candidates_of_orbit" << endl;
		}
 	generator_read_candidates_of_orbit(fname2, orbit_at_candidate_level, 
		candidates, nb_candidates, verbose_level - 1);

	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files  generator_read_candidates_of_orbit done" << endl;
		}


	if (candidates == NULL) {
		cout << "read_orbit_rep_and_candidates_from_files cound not read the candidates" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files Found " << nb_candidates << " candidates at level " << level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "read_orbit_rep_and_candidates_from_files done" << endl;
		}
}


void compute_orbits_on_subsets(generator *&gen, 
	INT target_depth,
	const BYTE *prefix, 
	INT f_W, INT f_w,
	action *A, action *A2, 
	strong_generators *Strong_gens, 
	void (*early_test_func_callback)(INT *S, INT len, 
		INT *candidates, INT nb_candidates, 
		INT *good_candidates, INT &nb_good_candidates, 
		void *data, INT verbose_level),
	void *early_test_func_data, 
	INT (*candidate_incremental_check_func)(INT len, INT *S, void *data, INT verbose_level), 
	void *candidate_incremental_check_data, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT nb_oracle_nodes = 1000;
	INT schreier_depth = target_depth;
	INT f_use_invariant_subset_if_available = TRUE;
	//INT f_implicit_fusion = FALSE;
	INT f_debug = FALSE;
	INT t0 = os_ticks();
	

	if (f_v) {
		cout << "compute_orbits_on_subsets verbose_level=" << verbose_level << endl;
		}
	gen = new generator;
	

	gen->f_W = f_W;
	gen->depth = target_depth;
	gen->downstep_orbits_print_max_orbits = 50;
	gen->downstep_orbits_print_max_points_per_orbit = INT_MAX;
	

	// !!!
	gen->f_allowed_to_show_group_elements = FALSE;

	if (f_v) {
		cout << "compute_orbits_on_subsets calling gen->init" << endl;
		}
	gen->init(A, A2, 
		Strong_gens, 
		target_depth, verbose_level - 1);

	strcpy(gen->fname_base, prefix);


	if (early_test_func_callback) {
		gen->init_early_test_func(
			early_test_func_callback, 
			early_test_func_data,  
			verbose_level);
		}


	if (candidate_incremental_check_func) {
		gen->init_incremental_check_func(
			candidate_incremental_check_func, 
			candidate_incremental_check_data);
		}

	gen->init_oracle(nb_oracle_nodes, verbose_level - 1);
	gen->init_root_node(verbose_level - 1);

	gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		f_debug, 
		verbose_level - 1);

	INT i, fst, len;

	if (f_v) {
		cout << "compute_orbits_on_subsets done" << endl;
		cout << "depth : number of orbits" << endl;
		}
	for (i = 0; i < target_depth + 1; i++) {
		fst = gen->first_oracle_node_at_level[i];
		len = gen->first_oracle_node_at_level[i + 1] - fst;
		if (f_v) {
			cout << i << " : " << len << endl;
			}
		}
}

void orbits_on_k_sets(action *A1, action *A2, 
	strong_generators *Strong_gens, 
	INT k, INT *&orbit_reps, INT &nb_orbits, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	generator *Gen;

	if (f_v) {
		cout << "orbits_on_k_sets" << endl;
		}
	
	Gen = orbits_on_k_sets_compute(A1, A2, 
		Strong_gens, 
		k, verbose_level);
	if (f_v) {
		cout << "orbits_on_k_sets: done with orbits_on_k_sets_compute" << endl;
		}

	Gen->get_orbit_representatives(k, nb_orbits, orbit_reps, verbose_level);
	
#if 0
	nb_orbits = Gen->nb_orbits_at_level(k);
	orbit_reps = NEW_INT(k * nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		Gen->get_set_by_level(k, i, orbit_reps + i * k);
		}
#endif

	if (f_v) {
		cout << "orbits_on_k_sets: we found " << nb_orbits << " orbits on " << k << "-sets" << endl;
		}

	delete Gen;
	if (f_v) {
		cout << "orbits_on_k_sets done" << endl;
		}
}

generator *orbits_on_k_sets_compute(action *A1, action *A2, 
	strong_generators *Strong_gens, 
	INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	generator *Gen;
	

	if (f_v) {
		cout << "orbits_on_k_sets_compute" << endl;
		}
	Gen = new generator;

	sprintf(Gen->fname_base, "orbits_on_k_sets");
	
	
	Gen->depth = k;
	
	if (f_v) {
		cout << "orbits_on_k_sets_compute calling Gen->init" << endl;
		}
	Gen->init(A1, A2, 
		Strong_gens, 
		Gen->depth /* sz */, verbose_level - 1);
	//Gen->init_check_func(
	//	check_zero_lines, 
	//	this /* candidate_check_data */);
	//Gen->init_incremental_check_func(
		//check_mindist_incremental, 
		//this /* candidate_check_data */);


#if 0
	Gen->f_print_function = TRUE;
	Gen->print_function = print_set;
	Gen->print_function_data = this;
#endif	

	INT nb_oracle_nodes = 1000;
	
	if (f_v) {
		cout << "orbits_on_k_sets calling Gen->init_oracle" << endl;
		}
	Gen->init_oracle(nb_oracle_nodes, verbose_level - 1);
	if (f_v) {
		cout << "orbits_on_k_sets calling Gen->init_root_node" << endl;
		}
	Gen->root[0].init_root_node(Gen, verbose_level - 1);
	
	INT schreier_depth = Gen->depth;
	INT f_use_invariant_subset_if_available = TRUE;
	//INT f_implicit_fusion = FALSE;
	INT f_debug = FALSE;
	INT t0 = os_ticks();
	
	if (f_v) {
		cout << "orbits_on_k_sets_compute: calling generator_main" << endl;
		}
	Gen->main(t0, 
		schreier_depth, 
		f_use_invariant_subset_if_available, 
		//f_implicit_fusion, 
		f_debug, 
		verbose_level - 1);
	

	return Gen;
}





void print_extension_type(ostream &ost, INT t)
{
	if (t == EXTENSION_TYPE_UNPROCESSED) {
		ost << "   unprocessed";
		}
	else if (t == EXTENSION_TYPE_EXTENSION) {
		ost << "     extension";
		}
	else if (t == EXTENSION_TYPE_FUSION) {
		ost << "        fusion";
		}
	else if (t == EXTENSION_TYPE_PROCESSING) {
		ost << "    processing";
		}
	else if (t == EXTENSION_TYPE_NOT_CANONICAL) {
		ost << " not canonical";
		}
	else {
		ost << "type=" << t;
		}
}

const BYTE *trace_result_as_text(trace_result r)
{
	if (r == found_automorphism) {
		return "found_automorphism";
		}
	else if (r == not_canonical) {
		return "not_canonical";
		}
	else if (r == no_result_extension_not_found) {
		return "no_result_extension_not_found";
		}
	else if (r == no_result_fusion_node_installed) {
		return "no_result_fusion_node_installed";
		}
	else if (r == no_result_fusion_node_already_installed) {
		return "no_result_fusion_node_already_installed";
		}
	else {
		return "unkown trace result";
		}
}

INT trace_result_is_no_result(trace_result r)
{
	if (r == no_result_extension_not_found || 
		r == no_result_fusion_node_installed || 
		r == no_result_fusion_node_already_installed) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


#if 0
void oracle_downstep_call_back_clique_found(clique_finder *CF, INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	//cout << "extending.C: call_back_clique_found" << endl;
	//cout << "CF->call_back_clique_found_data=" << CF->call_back_clique_found_data << endl;

	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;

#if 0
	if (CFI->node == 50) {
		cout << "oracle_downstep_call_back_clique_found before (*CFI->call_back_clique_found) " << endl;
		CFI->gen->root[49].print_extensions(cout);
		}
#endif

	return (*CFI->call_back_clique_found)(CF, verbose_level);

	//extending *E = (extending *) CF->call_back_clique_found_data;
	//E->CF = CF;
	//return E->clique_found(CF->current_clique, verbose_level);
}

void oracle_downstep_call_back_add_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level)
{
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;

	//INT f_v = (verbose_level >= 1);
	//extending *E = (extending *) CF->call_back_clique_found_data;
	//E->add_point(pt, current_clique_size, current_clique, verbose_level);
	(*CFI->call_back_add_point)(CF, current_clique_size, current_clique, 
		pt, verbose_level);
}

void oracle_downstep_call_back_delete_point(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT pt, INT verbose_level)
{
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;

	//INT f_v = (verbose_level >= 1);
	//extending *E = (extending *) CF->call_back_clique_found_data;
	//E->delete_point(pt, current_clique_size, current_clique, verbose_level);
	(*CFI->call_back_delete_point)(CF, current_clique_size, current_clique, 
		pt, verbose_level);
}

INT oracle_downstep_call_back_find_candidates(clique_finder *CF, 
	INT current_clique_size, INT *current_clique, 
	INT nb_pts, INT &reduced_nb_pts, 
	INT *pt_list, INT *pt_list_inv, 
	INT *candidates, INT verbose_level)
{
	clique_finder_interface *CFI = (clique_finder_interface *) CF->call_back_clique_found_data;
	INT ret;
	
#if 0
	if (CFI->node == 50) {
		cout << "oracle_downstep_call_back_clique_found before (*CFI->call_back_find_candidates) " << endl;
		CFI->gen->root[49].print_extensions(cout);
		}
#endif

	ret = (*CFI->call_back_find_candidates)(CF, 
		current_clique_size, current_clique, 
		nb_pts, reduced_nb_pts, 
		pt_list, pt_list_inv,
		candidates, verbose_level);

	
#if 0
	if (CFI->node == 50) {
		cout << "ret=" << ret;
		cout << "candidates:" << endl;
		INT_vec_print(cout, candidates, ret);
		cout << endl;
		cout << "oracle_downstep_call_back_clique_found after (*CFI->call_back_find_candidates) " << endl;
		CFI->gen->root[49].print_extensions(cout);
		}
#endif

	return ret;
	//INT f_v = (verbose_level >= 1);
	//extending *E = (extending *) CF->call_back_clique_found_data;
	//INT ret;
	
	//ret = E->find_candidates(current_clique_size, current_clique, nb_pts, pts, candidates, verbose_level);
	
	//return ret;
}
#endif

void wedge_product_export_magma(generator *Gen, INT n, INT q, INT vector_space_dimension, INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "wedge_product_export_magma" << endl;
		}
	
	//INT level;
	INT *the_set;
	INT *v;
	INT a, i, j, h, fst, len, ii, jj;
	longinteger_object go;
	INT *Elt;
	
	//level = depth_completed + 1;


	the_set = NEW_INT(level);
	v = NEW_INT(vector_space_dimension);
	Elt = NEW_INT(Gen->A->elt_size_in_INT);
	
	fst = Gen->first_oracle_node_at_level[level];
	len = Gen->first_oracle_node_at_level[level + 1] - fst;
	if (f_v) {
		cout << "exporting to magma" << endl;
		cout << "fst=" << fst << " len=" << len << endl;
		}
	oracle *O;
	BYTE fname[1000];

	sprintf(fname, "Wedge_n%ld_q%ld_d%ld.magma", n, q, level);

	{
	ofstream f(fname);

	f << "// file " << fname << endl;
	f << "n := " << n << ";" << endl;
	f << "q := " << q << ";" << endl;
	f << "d := " << level << ";" << endl;
	f << "n2 := " << vector_space_dimension << ";" << endl;
	f << "V := VectorSpace (GF (q), n2);" << endl;
	f << endl;
	f << "/* list of orbit reps */" << endl;
	f << "L := [" << endl;
	f << endl;

	for (i = 0; i < len; i++) {
		O = Gen->root + fst + i;
	
		f << "// orbit rep " << i << endl;
		f << "[" << endl;
		O->store_set_to(Gen, level - 1, the_set);
	 	for (j = 0; j < level; j++) {
			a = the_set[j];
			(*Gen->unrank_point_func)(v, a, Gen->rank_point_data);
			f << "[ ";
			for (h = 0; h < vector_space_dimension; h++) {
				f << v[h];
				if (h < vector_space_dimension - 1)
					f << ", ";
				}
			f << " ]";
			if (j < level - 1) {
				f << "," << endl;
				}
			else {
				f << "]" << endl;
				}
			}
		if (i < len - 1) {
			f << "," << endl << endl;
			}
		else {
			f << endl << "];" << endl << endl;
			}
		} // next i

	f << "// list of orbit lengths " << endl;
	f << "len := \[";
	
	for (i = 0; i < len; i++) {

		if ((i % 20) == 0) {
			f << endl;
			f << "// orbits " << i << " and following:" << endl;
			}

		Gen->orbit_length(i, level, go);
		f << go;
		if (i < len - 1) {
			f << ", ";
			}
		}
	f << "];" << endl << endl;


	f << "// subspaces of vector space " << endl;
	f << "L := [sub< V | L[i]>: i in [1..#L]];" << endl;

	f << "// stabilisers " << endl;
	f << "P := GL(n, q);" << endl;
	f << "E := ExteriorSquare (P);" << endl;


	f << "// base:" << endl;
	f << "BV := VectorSpace (GF (q), n);" << endl;
	f << "B := [ BV | " << endl;
	for (i = 0; i < Gen->A->base_len; i++) {
		a = Gen->A->base[i];
		PG_element_unrank_modified(*Gen->F, v, 1, n, a);
		//(*Gen->unrank_point_func)(v, a, Gen->rank_point_data);
		f << "[ ";
		for (h = 0; h < n; h++) {
			f << v[h];
			if (h < n - 1)
				f << ", ";
			}
        	if (i < Gen->A->base_len - 1)
				f << "], " << endl;
		else f << " ]" << endl;
		}
	f << "];" << endl;
	f << endl;
	f << "P`Base := B;" << endl;

	f << "// list of stabilizer generators" << endl;
	f << "S := [" << endl;
	f << endl;

	for (i = 0; i < len; i++) {
		O = Gen->root + fst + i;
	
		f << "// orbit rep " << i << " has " << O->nb_strong_generators << " strong generators";
		if (O->nb_strong_generators) {
			f << ", transversal lengths: ";
			INT_vec_print(f, O->tl, Gen->A->base_len);
			}
		f << endl;
		f << "[" << endl;

	 	for (j = 0; j < O->nb_strong_generators; j++) {

			Gen->A->element_retrieve(O->hdl_strong_generators[j], Elt, 0);
			
				f << "[";
			//Gen->A->element_print_quick(Elt, f);
			for (ii = 0; ii < n; ii++) {
				f << "[";
				for (jj = 0; jj < n; jj++) {
					a = Elt[ii * n + jj];
					f << a;
					if (jj < n - 1) {
						f << ", ";
						}
					else {
						f << "]";
						}
					}
				if (ii < n - 1) {
					f << "," << endl;
					}
				else {
					f << "]";
					}
				}
			
			if (j < O->nb_strong_generators - 1) {
				f << "," << endl;
				}
			}
			f << "]" << endl;
		if (i < len - 1) {
			f << "," << endl << endl;
			}
		else {
			f << endl << "];" << endl << endl;
			}
		} // next i

         f << endl << 
"T := [sub<GL(n, q) | [&cat (s): s in S[i]]> : i in [1..#S]];" << endl << endl;
	} // file f

	FREE_INT(the_set);
	FREE_INT(v);
	FREE_INT(Elt);

	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
	if (f_v) {
		cout << "wedge_product_export_magma done" << endl;
		}
}






