// poset_classification_io.C
//
// Anton Betten
// moved here from DISCRETA/snakesandladders.C
// December 27, 2008
// renamed from io.C to poset_classification_io.C Aug 24, 2011


#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"

void poset_classification::housekeeping(int i,
		int f_write_files, int t0, int verbose_level)
{
	int j, nb_nodes;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_embedded = TRUE;
	
	if (f_v) {
		cout << "poset_classification::housekeeping "
				"level=" << i << endl;
		cout << "poset_classification::housekeeping "
				"verbose_level=" << verbose_level << endl;
		cout << "poset_classification::housekeeping "
				"fname_base=" << fname_base << endl;
		}
	nb_nodes = nb_orbits_at_level(i);
	if (f_v) {
		cout << "###################################################"
				"###############################################" << endl;
		print_problem_label();
		cout << "Found " << nb_nodes << " orbits at depth " << i << endl;
		for (j = 0; j <= i; j++) {
			cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
			}
		cout << "total: " << first_poset_orbit_node_at_level[i + 1] << endl;
		//print_statistic_on_callbacks();
		compute_and_print_automorphism_group_orders(i, cout);
		//registry_dump_sorted();
		//registry_dump_sorted_by_size();
		//cout << "nb_times_trace=" << nb_times_trace << endl;
		//cout << "nb_times_trace_was_saved="
		// << nb_times_trace_was_saved << endl;
		//cout << "f_write_files=" << f_write_files << endl;
		}
	if (f_find_group_order) {
		find_automorphism_group_of_order(i, find_group_order);
		}
	if (f_vv) {
		if (nb_nodes < 1000) {
			int f_with_strong_generators = FALSE;
			int f_long_version = FALSE;
			write_lvl(cout, i, t0, f_with_strong_generators,
					f_long_version, verbose_level - 2);
			}
		}
	
	if (f_write_files) {
		char my_fname_base[1000];
		
		if (f_v) {
			cout << "poset_classification_housekeeping writing files" << endl;
			}
#if 1
		sprintf(my_fname_base, "%sa", fname_base);
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"my_fname_base=" << my_fname_base << endl;
			cout << "poset_classification_housekeeping "
					"before write_level_file_binary" << endl;
			}
		write_level_file_binary(i, my_fname_base,
				0/*verbose_level*/);
		if (i) {		
			sprintf(my_fname_base, "%sb", fname_base);
			if (f_v) {
				cout << "poset_classification_housekeeping "
						"my_fname_base=" << my_fname_base << endl;
				cout << "poset_classification_housekeeping "
						"before write_level_file_binary" << endl;
				}
			write_level_file_binary(i - 1,
					my_fname_base, 0/*verbose_level*/);
			if (f_v) {
				cout << "poset_classification_housekeeping "
						"my_fname_base=" << my_fname_base << endl;
				cout << "poset_classification_housekeeping "
						"before write_sv_level_file_binary" << endl;
				}
			write_sv_level_file_binary(i - 1, my_fname_base, 
				FALSE, 0, 0, 0 /*verbose_level*/);
			}
#endif
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"before write_lvl_file" << endl;
			}
		write_lvl_file(fname_base, i, t0,
				FALSE /* f_with_strong_generators */,
				FALSE /* f_long_version */, 0);
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"before poset_classification::write_data_file" << endl;
			}
		poset_classification::write_data_file(i /* depth_completed */,
				fname_base, 0);

		if (f_v) {
			cout << "poset_classification::housekeeping "
					"writing files done" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"not writing files" << endl;
			}
		}

	if (f_Log) {
		int verbose_level = 1;
		int f = first_poset_orbit_node_at_level[i];
		int len = nb_orbits_at_level(i);
		print_problem_label();
		cout << "There are " << len
				<< " nodes at level " << i << ":" << endl;
		for (j = 0; j < len; j++) {
			Log_nodes(f + j, i, cout, FALSE, verbose_level);
			}
		}

	if (f_log && i == sz) {
		int verbose_level = 1;
		int ii;

		for (ii = 0; ii <= sz; ii++) {
			int f = first_poset_orbit_node_at_level[ii];
			int len = nb_orbits_at_level(ii);
			print_problem_label();
			cout << "There are " << len
					<< " nodes at level " << ii << ":" << endl;
			for (j = 0; j < len; j++) {
				Log_nodes(f + j, ii, cout, FALSE, verbose_level);
				}
			}
		}

	if (f_T || (f_t && i == sz)) {
		if (f_v) {
			cout << "poset_classification::housekeeping "
					"before write_treefile_and_draw_tree" << endl;
			}

		write_treefile_and_draw_tree(fname_base, i, 
			xmax, ymax, radius, f_embedded, 0 /*verbose_level - 1*/);
			// in poset_classification_draw.C

		if (f_v) {
			cout << "poset_classification::housekeeping "
					"after write_treefile_and_draw_tree" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "poset_classification_housekeeping "
					"not writing tree" << endl;
			}
		}

	if (f_v) {
		cout << "poset_classification::housekeeping done" << endl;
		}
}


void poset_classification::housekeeping_no_data_file(int i,
		int t0, int verbose_level)
{
	int j;
	int f_v = (verbose_level >= 1);
	int f_embedded = TRUE;
	
	if (f_v) {
		cout << "poset_classification::housekeeping_no_data_file "
				"verbose_level=" << verbose_level << endl;
		}
	if (f_v) {
		cout << "######################################################"
				"############################################" << endl;
		cout << "depth " << i << " completed, found " 
			<< nb_orbits_at_level(i) << " orbits" << endl;
		for (j = 0; j <= i; j++) {
			cout << j << " : " << nb_orbits_at_level(j)
					<< " orbits" << endl;
			}
		cout << "total: " << first_poset_orbit_node_at_level[i + 1] << endl;
		compute_and_print_automorphism_group_orders(i, cout);
		}

	if (f_W || (f_w && i == sz)) {
		char fname_base2[1000];
		
		sprintf(fname_base2, "%sa", fname_base);
		write_level_file_binary(i, fname_base2, 1/*verbose_level*/);
		if (i) {		
			sprintf(fname_base2, "%sb", fname_base);
			write_level_file_binary(i - 1, fname_base2, 1/*verbose_level*/);
			write_sv_level_file_binary(i - 1, 
				fname_base, FALSE, 0, 0, 1/*verbose_level*/);
			}
		write_lvl_file(fname_base, i, t0,
				FALSE /* f_with_strong_generators */,
				FALSE /* f_long_version */, 0);
		
		//poset_classification_write_data_file(gen,
		// i /* depth_completed */, gen->fname_base, 0);

		}

	if (f_T || (f_t && i == sz)) {
		write_treefile_and_draw_tree(fname_base, i, 
			xmax, ymax, radius, f_embedded, verbose_level - 1);
		}
	if (f_v) {
		cout << "poset_classification::housekeeping_no_data_file done" << endl;
		}
}

int poset_classification::test_sv_level_file_binary(int level, char *fname_base)
{
	char fname[1000];
	
	sprintf(fname, "%s_lvl_%d_sv.data", fname_base, level);
	if (file_size(fname) >= 1)
		return TRUE;
	else
		return FALSE;
}

void poset_classification::read_sv_level_file_binary(
	int level, char *fname_base,
	int f_split, int split_mod, int split_case, 
	int f_recreate_extensions, int f_dont_keep_sv, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	
	sprintf(fname, "%s_lvl_%d_sv.data", fname_base, level);
	
	if (f_v) {
		cout << "poset_classification::read_sv_level_file_binary "
				"reading file " << fname << " of size "
				<< file_size(fname) << endl;
		}


	FILE *fp;

	fp = fopen(fname, "rb");

	read_sv_level_file_binary2(level, fp, 
		f_split, split_mod, split_case, 
		f_recreate_extensions, f_dont_keep_sv, 
		verbose_level - 1);
	
	fclose(fp);

}

void poset_classification::write_sv_level_file_binary(
	int level, char *fname_base,
	int f_split, int split_mod, int split_case, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	
	sprintf(fname, "%s_lvl_%d_sv.data", fname_base, level);
	
	if (f_v) {
		cout << "poset_classification::write_sv_level_file_binary "
				"fname = " << fname << endl;
		}


	FILE *fp;

	fp = fopen(fname, "wb");

	write_sv_level_file_binary2(level, fp, 
		f_split, split_mod, split_case, 
		verbose_level);
	
	fclose(fp);

	if (f_v) {
		cout << "poset_classification::write_sv_level_file_binary "
			"finished written file "
			<< fname << " of size " << file_size(fname) << endl;
		}
}

void poset_classification::read_sv_level_file_binary2(int level, FILE *fp, 
	int f_split, int split_mod, int split_case, 
	int f_recreate_extensions, int f_dont_keep_sv, 
	int verbose_level)
{
	int f, i, nb_nodes;
	int f_v = (verbose_level >= 1);
	int4 I;
	
	f = first_poset_orbit_node_at_level[level];
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_classification::read_sv_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		cout << "f_recreate_extensions="
				<< f_recreate_extensions << endl;
		cout << "f_dont_keep_sv=" << f_dont_keep_sv << endl;
		if (f_split) {
			cout << "f_split is TRUE, split_mod=" << split_mod
					<< " split_case=" << split_case << endl;
			}
		}
	// version number of this file format
	I = fread_int4(fp);
	if (I != 1) { 
		cout << "poset_classification::read_sv_level_file_binary2: "
				"unknown file version" << endl;
		exit(1);
		}
	I = fread_int4(fp);
	if (I != level) {
		cout << "poset_classification::read_sv_level_file_binary2: "
				"level does not match" << endl;
		exit(1);
		}
	I = fread_int4(fp);
	if (I != nb_nodes) {
		cout << "poset_classification::read_sv_level_file_binary2: "
				"nb_nodes does not match" << endl;
		exit(1);
		}
	for (i = 0; i < nb_nodes; i++) {
		if (f_split) {
			if ((i % split_mod) != split_case)
				continue;
			}
		root[f + i].sv_read_file(fp, verbose_level - 2);
		if (f_recreate_extensions) {
			root[f + i].reconstruct_extensions_from_sv(
					this, verbose_level - 1);
			}
		if (f_dont_keep_sv) {
			FREE_int(root[f + i].sv);
			root[f + i].sv = NULL;
			}
		}
	I = fread_int4(fp);
	if (I != MAGIC_SYNC) {
		cout << "poset_classification::read_sv_level_file_binary2: "
				"MAGIC_SYNC does not match" << endl;
		exit(1);
		}
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_classification::read_sv_level_file_binary2 "
				"finished" << endl;
		}
}

void poset_classification::write_sv_level_file_binary2(int level, FILE *fp, 
	int f_split, int split_mod, int split_case, 
	int verbose_level)
{
	int f, i, nb_nodes;
	int f_v = (verbose_level >= 1);
	
	f = first_poset_orbit_node_at_level[level];
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_classification::write_sv_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		}
	// version number of this file format
	fwrite_int4(fp, 1);
	fwrite_int4(fp, level);
	fwrite_int4(fp, nb_nodes);
	for (i = 0; i < nb_nodes; i++) {
		if (f_split) {
			if ((i % split_mod) != split_case)
				continue;
			}
		root[f + i].sv_write_file(fp, verbose_level - 2);
		}
	fwrite_int4(fp, MAGIC_SYNC);
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_classification::write_sv_level_file_binary2 finished" << endl;
		}
}

void poset_classification::read_level_file_binary(int level,
		char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	int nb_group_elements;
	
	sprintf(fname, "%s_lvl_%d.data", fname_base, level);
	
	if (f_v) {
		cout << "poset_classification::read_level_file_binary "
				"reading file " << fname << " of size "
				<< file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "poset_classification::read_level_file_binary "
				"probems while reading file " << fname << endl;
		exit(1);
		}


	FILE *fp;

	fp = fopen(fname, "rb");

	read_level_file_binary2(level, fp, nb_group_elements,
			verbose_level - 1);
	
	fclose(fp);

}

void poset_classification::write_level_file_binary(int level,
		char *fname_base, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname[1000];
	int nb_group_elements;
	
	sprintf(fname, "%s_lvl_%d.data", fname_base, level);
	
	if (f_v) {
		cout << "poset_classification::write_level_file_binary "
				"fname = " << fname << endl;
		}


	FILE *fp;

	fp = fopen(fname, "wb");

	write_level_file_binary2(level, fp,
			nb_group_elements, verbose_level);
	
	fclose(fp);

	if (f_v) {
		cout << "poset_classification::write_level_file_binary "
				"finished written file "
			<< fname << " of size " << file_size(fname) 
			<< " nb_group_elements=" << nb_group_elements << endl;
		}
}

void poset_classification::read_level_file_binary2(
	int level, FILE *fp,
	int &nb_group_elements, int verbose_level)
{
	int f, i, nb_nodes, magic_sync;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int4 I;
	
	if (f_v) {
		cout << "poset_classification::read_level_file_binary2" << endl;
		}
	f = first_poset_orbit_node_at_level[level];
	nb_group_elements = 0;
	I = fread_int4(fp);
	if (I != 1) {
		cout << "poset_classification::read_level_file_binary2 "
				"version = " << I << " unknown" << endl;
		exit(1);
		}

	I = fread_int4(fp);
	if (I != level) {
		cout << "poset_classification::read_level_file_binary2 "
				"level = " << I << " should be " << level << endl;
		exit(1);
		}

	nb_nodes = fread_int4(fp);
	if (f_v) {
		cout << "poset_classification::read_level_file_binary, "
				"nb_nodes = " << nb_nodes << endl;
		}
	first_poset_orbit_node_at_level[level + 1] = f + nb_nodes;
	
	if (f_v) {
		cout << "poset_classification::read_level_file_binary2 "
				"f + nb_nodes = " << f + nb_nodes << endl;
		cout << "poset_classification::read_level_file_binary2 "
				"nb_poset_orbit_nodes_allocated = "
			<< nb_poset_orbit_nodes_allocated << endl;
		}
	if (f + nb_nodes > nb_poset_orbit_nodes_allocated) {
		reallocate_to(f + nb_nodes, verbose_level - 2);
		}
	for (i = 0; i < nb_nodes; i++) {
		if (f_vv && nb_nodes > 1000 && ((i % 1000) == 0)) {
			cout << "reading node " << i << endl;
			}
		root[f + i].read_file(A, fp, nb_group_elements,
				verbose_level - 2);
		}
	if (f_v) {
		cout << "reading nodes completed" << endl;
		}
	magic_sync = fread_int4(fp);
	if (magic_sync != MAGIC_SYNC) {
		cout << "poset_classification::read_level_file_binary2 "
				"could not read MAGIC_SYNC, file is corrupt" << endl;
		cout << "MAGIC_SYNC=" << MAGIC_SYNC << endl;
		cout << "we read   =" << magic_sync << endl;		
		exit(1);
		}
	if (f_v) {
		cout << "poset_classification::read_level_file_binary2 "
				"finished ";
		cout << "level=" << level 
			<< ", with " << nb_nodes << " nodes" 
			<< " and " << nb_group_elements << " group elements"
			<< endl;
		}
}

void poset_classification::write_level_file_binary2(
	int level, FILE *fp,
	int &nb_group_elements, int verbose_level)
{
	int f, i, nb_nodes;
	int f_v = FALSE;//(verbose_level >= 1);
	
	f = first_poset_orbit_node_at_level[level];
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "poset_classification::write_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		}
	nb_group_elements = 0;
	// version number of this file format
	fwrite_int4(fp, 1);
	fwrite_int4(fp, level);
	fwrite_int4(fp, nb_nodes);
	for (i = 0; i < nb_nodes; i++) {
		root[f + i].write_file(A, fp,
				nb_group_elements, verbose_level - 2);
		}
	fwrite_int4(fp, MAGIC_SYNC);
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_classification::write_level_file_binary2 "
				"finished" << endl;
		}
}

int poset_classification::calc_size_on_file(int depth_completed,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, s = 0;
	int nb_nodes;
	
	if (f_v) {
		cout << "poset_classification::calc_size_on_file "
				"depth_completed=" << depth_completed << endl;
		}
	nb_nodes = first_poset_orbit_node_at_level[depth_completed + 1];
	s += 3 * 4;
	for (i = 0; i <= depth_completed + 1; i++) {
		s += 4;
		}
	for (i = 0; i < nb_nodes; i++) {
		s += root[i].calc_size_on_file(A, verbose_level);
		}
	s += 4; // MAGIC_SYNC
	if (f_v) {
		cout << "poset_classification::calc_size_on_file "
				"depth_completed=" << depth_completed
				<< " s=" << s << endl;
		}
	return s;
}

void poset_classification::make_fname_candidates_file_default(
		char *fname, int level)
{
	sprintf(fname, "%s_lvl_%d_candidates.bin", fname_base, level);
}

void poset_classification::write_candidates_binary_using_sv(char *fname_base,
		int lvl, int t0, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	char fname1[1000];
	
	if (f_v) {
		cout << "poset_classification::write_candidates_binary_using_sv "
				"lvl=" << lvl << " fname_base=" << fname_base << endl;
		}
	sprintf(fname1, "%s_lvl_%d_candidates.bin", fname_base, lvl);
	{
	int fst, len;
	int *nb_cand;
	int *cand_first;
	int total_nb_cand = 0;
	int *subset;
	int *Cand;
	int i, j, node, nb, pos;

	fst = first_poset_orbit_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "poset_classification::write_candidates_binary_using_sv "
				"first node at level " << lvl << " is " << fst << endl;
		cout << "poset_classification::write_candidates_binary_using_sv "
				"number of nodes at level " << lvl << " is " << len << endl;
		}
	nb_cand = NEW_int(len);
	cand_first = NEW_int(len);
	for (i = 0; i < len; i++) {
		node = fst + i;
		int *osv = root[node].sv;

		if (f_vv) {
			cout << "poset_classification::write_candidates_binary_using_sv "
					"node " << i << " / " << len << endl;
			}
		if (osv == NULL) {
			cout << "poset_classification::write_candidates_binary_using_sv "
					"osv == NULL, we don't have a Schreier vector; "
					"maybe set -schreier_depth <depth>" << endl;
			exit(1);
			}
		nb = osv[0];
		if (FALSE) {
			cout << "poset_classification::write_candidates_binary_using_sv "
					"i=" << i << endl;
			cout << "poset_classification::write_candidates_binary_using_sv "
					"nb=" << nb << endl;
			}
		subset = osv + 1;
		nb_cand[i] = nb;
		total_nb_cand += nb;
		}
	if (f_vv) {
		cout << "poset_classification::write_candidates_binary_using_sv "
				"total_nb_cand=" << total_nb_cand << endl;
		}
	Cand = NEW_int(total_nb_cand);
	pos = 0;
	for (i = 0; i < len; i++) {
		node = fst + i;
		int *osv = root[node].sv;
		nb = osv[0];
		subset = osv + 1;
		cand_first[i] = pos;
		for (j = 0; j < nb; j++) {
			Cand[pos + j] = subset[j];
			}
		pos += nb;
		}
	FILE *fp;

	fp = fopen(fname1, "wb");

	fwrite_int4(fp, len);
	for (i = 0; i < len; i++) {
		fwrite_int4(fp, nb_cand[i]);
		fwrite_int4(fp, cand_first[i]);
		}
	for (i = 0; i < total_nb_cand; i++) {
		fwrite_int4(fp, Cand[i]);
		}


	fclose(fp);


	FREE_int(nb_cand);
	FREE_int(cand_first);
	FREE_int(Cand);
	}
	if (f_v) {
		cout << "written file " << fname1 << " of size "
				<< file_size(fname1) << endl;
		}
}

void poset_classification_read_candidates_of_orbit(
	const char *fname, int orbit_at_level,
	int *&candidates, int &nb_candidates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	FILE *fp;
	int nb, cand_first, i;


	if (f_v) {
		cout << "poset_classification_read_candidates_of_orbit" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
		}
	
	if (file_size(fname) <= 0) {
		cout << "poset_classification_read_candidates_of_orbit file "
				<< fname << " does not exist" << endl;
		exit(1);
		}

	fp = fopen(fname, "rb");

	nb = fread_int4(fp);
	if (orbit_at_level >= nb) {
		cout << "poset_classification_read_candidates_of_orbit "
				"orbit_at_level >= nb" << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
		cout << "nb=" << nb << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "seeking position "
				<< (1 + orbit_at_level * 2) * sizeof(int4) << endl;
		}
	fseek(fp, (1 + orbit_at_level * 2) * sizeof(int4), SEEK_SET);
	nb_candidates = fread_int4(fp);
	if (f_vv) {
		cout << "nb_candidates=" << nb_candidates << endl;
		}
	cand_first = fread_int4(fp);
	if (f_v) {
		cout << "cand_first=" << cand_first << endl;
		}
	candidates = NEW_int(nb_candidates);
	fseek(fp, (1 + nb * 2 + cand_first) * sizeof(int4), SEEK_SET);
	for (i = 0; i < nb_candidates; i++) {
		candidates[i] = fread_int4(fp);
		}
	fclose(fp);
	if (f_v) {
		cout << "poset_classification_read_candidates_of_orbit done" << endl;
		}
}

void poset_classification::read_level_file(int level,
		char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *set_sizes;
	int **sets;
	char **data;
	int nb_cases;
	int nb_nodes, first_at_level;
	int i, I, J;
	poset_orbit_node *O;
	
	if (f_v) {
		cout << "poset_classification::read_level_file "
				"fname=" << fname << endl;
		}
	
#if 1
	read_and_parse_data_file(fname, nb_cases,
			data, sets, set_sizes, verbose_level - 1);
	
#else

	if (try_to_read_file(fname, nb_cases, data)) {
		if (f_v) {
			cout << "poset_classification::read_level_file read "
					"file " << fname << " nb_cases = " << nb_cases << endl;
			}
		}
	else {
		cout << "poset_classification::read_level_file "
				"couldn't read file " << fname << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "poset_classification::read_level_file parsing" << endl;
		}
	parse_sets(nb_cases, data, set_sizes, sets);
	if (f_vv) {
		cout << "poset_classification::read_level_file "
				"parsing finished" << endl;
		}
		// in GALOIS/util.C
#endif

	first_at_level = first_poset_orbit_node_at_level[level];
	nb_nodes = first_at_level + nb_cases;
	
	if (nb_nodes > nb_poset_orbit_nodes_allocated) {
		if (f_vv) {
			cout << "poset_classification::read_level_file "
					"reallocating to " << nb_nodes << " nodes" << endl;
			}
		reallocate_to(nb_nodes, verbose_level - 1);
		}
	first_poset_orbit_node_at_level[level + 1] = nb_nodes;
	for (i = 0; i < nb_cases; i++) {
		I = first_at_level + i;
		O = &root[I];
		
		cout << setw(10) << i << " : ";
		int_vec_print(cout, sets[i], level);
		cout << endl;
		
		J = find_poset_orbit_node_for_set(level - 1,
				sets[i], FALSE /* f_tolerant */,
				0/*verbose_level*/);
		cout << "J=" << J << endl;
		
		O->node = I;
		O->prev = J;
		O->pt = sets[i][level - 1];
		O->nb_strong_generators = 0;
		O->hdl_strong_generators = NULL;
		O->tl = NULL;
		O->nb_extensions = 0;
		O->E = NULL;
		O->sv = NULL;

		{
		group Aut;
		
		Aut.init(A);
		
		if (strlen(data[i])) {
			Aut.init_ascii_coding(data[i]);
		
			Aut.decode_ascii(FALSE);
		
			// now strong poset_classifications are available
		
			Aut.schreier_sims(0);
		
			cout << "the automorphism group has order ";
			Aut.print_group_order(cout);
			cout << endl;
		
			strong_generators *Strong_gens;

			Strong_gens = NEW_OBJECT(strong_generators);
			Strong_gens->init_from_sims(Aut.S, 0);

#if 0
			cout << "and is strongly generated by the "
					"following " << Aut.SG->len << " elements:" << endl;

			Aut.SG->print(cout);
			cout << endl;
#endif
			O->store_strong_generators(this, Strong_gens);
			cout << "strong poset_classifications stored" << endl;

			FREE_OBJECT(Strong_gens);
			}
		else {
			//cout << "trivial group" << endl;
			//Aut.init_strong_generators_empty_set();
			
			}
		}

		}
	delete [] set_sizes;
	if (f_v) {
		cout << "poset_classification::read_level_file "
				"fname=" << fname << " done" << endl;
		}
}

void poset_classification::read_data_file(int &depth_completed,
		const char *fname, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int size;
	int nb_group_elements;
	memory_object *m;
	
	
	if (f_v) {
		cout << "poset_classification::read_data_file "
				"fname = " << fname << endl;
		cout << "A->elt_size_in_int = "
				<< A->elt_size_in_int << endl;
		cout << "A->coded_elt_size_in_char = "
				<< A->coded_elt_size_in_char << endl;
		}
	size = file_size(fname);
	if (f_v) {
		cout << "file size = " << size << endl;
		}
	if (size == -1) {
		cout << "error: the file does not exist" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "poset_classification::read_data_file before m->alloc" << endl;
		}

	m = NEW_OBJECT(memory_object);
	m->alloc(size, 0);

	if (f_v) {
		cout << "poset_classification::read_data_file after m->alloc" << endl;
		}
	
	m->used_length = 0;

	FILE *fp;
	
	fp = fopen(fname, "r");

	fread(m->char_pointer, 1 /* size */, size /* items */, fp);
	
	fclose(fp);

	if (f_v) {
		cout << "poset_classification::read_data_file after fread" << endl;
		}

	m->used_length = size;
	m->cur_pointer = 0;

	if (f_v) {
		cout << "poset_classification::read_data_file "
				"before poset_classification_read_memory" << endl;
		}
	read_memory_object(depth_completed, m,
			nb_group_elements, verbose_level - 2);
	if (f_v) {
		cout << "poset_classification::read_data_file "
				"after poset_classification_read_memory" << endl;
		}

	FREE_OBJECT(m);
	if (f_v) {
		cout << "poset_classification::read_data_file done" <<endl;
		}
	
}

void poset_classification::write_data_file(int depth_completed,
		const char *fname_base, int verbose_level)
{
	memory_object *m;
	int f_v = (verbose_level >= 1);
	char fname[1000];
	int nb_group_elements;
	int size0;
	
	sprintf(fname, "%s_%d.data", fname_base, depth_completed);
	
	if (f_v) {
		cout << "poset_classification::write_data_file fname = " << fname << endl;
		cout << "A->elt_size_in_int = " << A->elt_size_in_int << endl;
		cout << "A->coded_elt_size_in_char = "
				<< A->coded_elt_size_in_char << endl;
		}
	size0 = calc_size_on_file(depth_completed, verbose_level);
	if (f_v) {
		cout << "size on file = " << size0 << endl;
		}
	
	if (size0 > 100 * ONE_MILLION) {
		cout << "poset_classification::write_data_file file=" << fname << endl;
		cout << "size on file = " << size0 << endl;
		cout << "the size is very big" << endl;
		}
	
	m = NEW_OBJECT(memory_object);
	m->alloc(size0, 0);
	m->used_length = 0;

	write_memory_object(depth_completed, m,
			nb_group_elements, verbose_level);

	FILE *fp;
	int size;

	size = m->used_length;
	
	fp = fopen(fname, "wb");

	fwrite(m->char_pointer, 1 /* size */, size /* items */, fp);
	
	fclose(fp);

	FREE_OBJECT(m);
	
	if (f_v) {
		cout << "poset_classification::write_data_file finished written file " 
			<< fname << " of size " << file_size(fname) 
			<< " nb_group_elements=" << nb_group_elements << endl;
		}
}

void poset_classification::read_memory_object(int &depth_completed,
		memory_object *m, int &nb_group_elements,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, nb_nodes, version, magic_sync;
	
	//nb_nodes = first_oracle_node_at_level[depth_completed + 1];
	if (f_v) {
		cout << "poset_classification::read_memory_object, "
				"data size (in chars) = " << m->used_length << endl;
		}
	nb_group_elements = 0;
	m->read_int(&version);
	if (version != 1) {
		cout << "poset_classification::read_memory_object "
				"version = " << version << " unknown" << endl;
		exit(1);
		}
	m->read_int(&depth_completed);
	if (f_v) {
		cout << "poset_classification::read_memory_object "
				"depth_completed = " << depth_completed << endl;
		}

	if (depth_completed > sz) {
		cout << "poset_classification::read_memory_object "
				"depth_completed > sz" << endl;
		exit(1);
		}

	m->read_int(&nb_nodes);
	if (f_v) {
		cout << "poset_classification::read_memory_object "
				"nb_nodes = " << nb_nodes << endl;
		}

	//G->init_oracle(nb_nodes);

#if 1
	if (nb_nodes > nb_poset_orbit_nodes_allocated) {
		reallocate_to(nb_nodes, verbose_level - 1);
		}
#endif
	for (i = 0; i <= depth_completed + 1; i++) {
		m->read_int(&first_poset_orbit_node_at_level[i]);
		}
	for (i = 0; i < nb_nodes; i++) {
		if ((f_v && ((i % 1000) == 0)) || f_vv) {
			cout << "reading node " << i << endl;
			}
		root[i].read_memory_object(A, m,
				nb_group_elements, verbose_level - 1);
		}
	if (f_v) {
		cout << "poset_classification::read_memory_object "
				"reading nodes completed" << endl;
		}
	m->read_int(&magic_sync);
	if (magic_sync != MAGIC_SYNC) {
		cout << "poset_classification::read_memory_object "
				"could not read MAGIC_SYNC, file is corrupt" << endl;
		exit(1);
		}
	nb_poset_orbit_nodes_used = nb_nodes;
	if (f_v) {
		cout << "poset_classification::read_memory_object finished ";
		cout << "depth_completed=" << depth_completed 
			<< ", with " << nb_nodes << " nodes" 
			<< " and " << nb_group_elements << " group elements"
			<< endl;
		}
}

void poset_classification::write_memory_object(int depth_completed,
		memory_object *m, int &nb_group_elements, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, nb_nodes;
	
	nb_nodes = first_poset_orbit_node_at_level[depth_completed + 1];
	if (f_v) {
		cout << "poset_classification::write_memory_object "
				<< nb_nodes << " nodes" << endl;
		}
	nb_group_elements = 0;
	m->write_int(1); // version number of this file format
	m->write_int(depth_completed);
	m->write_int(nb_nodes);
	for (i = 0; i <= depth_completed + 1; i++) {
		m->write_int(first_poset_orbit_node_at_level[i]);
		}
	for (i = 0; i < nb_nodes; i++) {
		root[i].write_memory_object(A, m,
				nb_group_elements, verbose_level - 2);
		}
	m->write_int(MAGIC_SYNC); // a check to see if the file is not corrupt
	if (f_v) {
		cout << "poset_classification::write_memory_object "
				"finished, data size (in chars) = "
				<< m->used_length << endl;
		}
}

void poset_classification::recover(const char *recover_fname,
		int &depth_completed, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::recover "
				"recovering from file " << recover_fname << endl;
		}
	read_data_file(depth_completed, recover_fname, verbose_level);
	if (f_v) {
		cout << "poset_classification::recover "
				"recovering finished, "
				"depth_completed = " << depth_completed << endl;
		}
}

void poset_classification::write_lvl_file_with_candidates(
		char *fname_base, int lvl, int t0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname1[1000];
	
	sprintf(fname1, "%s_lvl_%d_candidates.txt", fname_base, lvl);
	{
	ofstream f(fname1);
	int cur;
	
	//f << "# " << lvl << endl; 
	for (cur = first_poset_orbit_node_at_level[lvl];
		cur < first_poset_orbit_node_at_level[lvl + 1]; cur++) {
		root[cur].log_current_node_with_candidates(
				this, lvl, f, verbose_level - 2);
		}
	f << "-1 " << first_poset_orbit_node_at_level[lvl + 1]
				- first_poset_orbit_node_at_level[lvl]
		<< " " << first_poset_orbit_node_at_level[lvl] << " in ";
	time_check(f, t0);
	f << endl;
	f << "# in action " << A->label << endl;
	}
	if (f_v) {
		cout << "written file " << fname1
				<< " of size " << file_size(fname1) << endl;
		}
}


void poset_classification::write_lvl_file(char *fname_base,
		int lvl, int t0, int f_with_stabilizer_generators,
		int f_long_version,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	char fname1[1000];
	sprintf(fname1, "%s_lvl_%d", fname_base, lvl);
	{
	ofstream f(fname1);
	int i, fst, len;
	

	fst = first_poset_orbit_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);

	f << "# " << lvl << endl; 
	for (i = 0; i < len; i++) {
		root[fst + i].log_current_node(this,
				lvl, f, f_with_stabilizer_generators,
				f_long_version);
		}
	f << "-1 " << len << " "
			<< first_poset_orbit_node_at_level[lvl] << " in ";
	time_check(f, t0);
	compute_and_print_automorphism_group_orders(lvl, f);
	f << endl;
	f << "# in action " << A->label << endl;
	}
	if (f_v) {
		cout << "written file " << fname1
				<< " of size " << file_size(fname1) << endl;
		}
}

void poset_classification::write_lvl(ostream &f, int lvl, int t0,
		int f_with_stabilizer_generators, int f_long_version,
		int verbose_level)
{
	//int f_v = (verbose_level >= 1);
	int i;
	int fst, len;


	fst = first_poset_orbit_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);
	
	f << "# " << lvl << endl; 
	for (i = 0; i < len; i++) {
		root[fst + i].log_current_node(this, lvl, f,
				f_with_stabilizer_generators, f_long_version);
		}
	f << "-1 " << len << " " << first_poset_orbit_node_at_level[lvl]
		<< " in ";
	time_check(f, t0);
	f << endl;
	compute_and_print_automorphism_group_orders(lvl, f);
	f << endl;
	f << "# in action " << A->label << endl;
}

void poset_classification::log_nodes_for_treefile(int cur, int depth,
		ostream &f, int f_recurse, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, next;
	poset_orbit_node *node = &root[cur];
		

	if (f_v) {
		cout << "poset_classification::log_nodes_for_treefile "
				"cur=" << cur << endl;
		}
	if (f_starter && cur < starter_size) {
		return; // !!!
		}
	
	node->log_current_node(this, depth, f,
			FALSE /* f_with_strong_generators */, 0);
	
	if (f_recurse) {
		//cout << "recursing into dependent nodes" << endl;
		for (i = 0; i < node->nb_extensions; i++) {
			if (node->E[i].type == EXTENSION_TYPE_EXTENSION) {
				if (node->E[i].data >= 0) {
					next = node->E[i].data;
					log_nodes_for_treefile(next,
							depth + 1, f, TRUE, verbose_level);
					}
				}
			}
		}
}

void poset_classification::Log_nodes(int cur, int depth,
		ostream &f, int f_recurse,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, next;
	poset_orbit_node *node = &root[cur];
		

	if (f_v) {
		cout << "Log_nodes cur=" << cur << endl;
		}
	if (f_starter && cur < starter_size) {
		return; // !!!
		}
	if (f_v) {
		f << "Node " << cur << endl;
		f << "===============" << endl;
		node->log_current_node(this, depth, f,
				FALSE /* f_with_strong_generators */, verbose_level);
		//f << "the stabilizer has order ";
		//G.print_group_order(f);
		//f << endl;	
		
		f << "with " << node->nb_strong_generators
				<< " strong poset_classifications:" << endl;
		if (f_v) {
			cout << "Log_nodes cur=" << cur
					<< " printing strong poset_classifications" << endl;
			}
		for (i = 0; i < node->nb_strong_generators; i++) {
			A->element_retrieve(node->hdl_strong_generators[i], Elt1, 0);
			A->element_print_quick(Elt1, f);
			f << endl;
			if (A->degree < 100) {
				A->element_print_as_permutation(Elt1, f);
				f << endl;
				}
			}

		if (node->nb_strong_generators) {
			if (f_v) {
				cout << "Log_nodes cur=" << cur
						<< " printing tl" << endl;
				}
			f << "tl: ";
			int_vec_print(f, node->tl, A->base_len);
			f << endl;
			}
		
		if (f_v) {
			cout << "Log_nodes cur=" << cur
					<< " printing extensions" << endl;
			}
		node->print_extensions(f);
		f << endl;

		for (i = 0; i < node->nb_extensions; i++) {
			if (node->E[i].type == EXTENSION_TYPE_FUSION) {
				f << "fusion node " << i << ":" << endl;
				A->element_retrieve(node->E[i].data, Elt1, 0);
				A->element_print_verbose(Elt1, f);
				f << endl;
				}
			}
		}
	else {
		//cout << "log_current_node node=" << node->node
		// << " prev=" << node->prev << endl;
		node->log_current_node(this, depth, f,
				FALSE /* f_with_strong_generators */, 0);
		}
	
	if (f_recurse) {
		//cout << "recursing into dependent nodes" << endl;
		for (i = 0; i < node->nb_extensions; i++) {
			if (node->E[i].type == EXTENSION_TYPE_EXTENSION) {
				if (node->E[i].data >= 0) {
					next = node->E[i].data;
					Log_nodes(next, depth + 1, f, TRUE, verbose_level);
					}
				}
			}
		}
}

void poset_classification::log_current_node(ostream &f, int size)
{
	//longinteger_object go;
	int i;
	

	f << size << " ";
	for (i = 0; i < size; i++) {
		f << S[i] << " ";
		}
	f << endl;

}



void poset_classification::make_spreadsheet_of_orbit_reps(
		spreadsheet *&Sp, int max_depth)
{
	int Nb_orbits, nb_orbits, i, level, first;
	pchar *Text_level;
	pchar *Text_node;
	pchar *Text_orbit_reps;
	pchar *Text_stab_order;
	pchar *Text_orbit_length;
	pchar *Text_schreier_vector_length;
	longinteger_object stab_order, orbit_length;
	int schreier_vector_length;
	int *rep;
	char str[1000];
	poset_orbit_node *O;

	Nb_orbits = 0;
	for (level = 0; level <= max_depth; level++) {
		Nb_orbits += nb_orbits_at_level(level);
		}

	rep = NEW_int(max_depth);
	Text_level = NEW_pchar(Nb_orbits);
	Text_node = NEW_pchar(Nb_orbits);
	Text_orbit_reps = NEW_pchar(Nb_orbits);
	Text_stab_order = NEW_pchar(Nb_orbits);
	Text_orbit_length = NEW_pchar(Nb_orbits);
	Text_schreier_vector_length = NEW_pchar(Nb_orbits);

	first = 0;
	for (level = 0; level <= max_depth; level++) {
		first = first_poset_orbit_node_at_level[level];
		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {
			sprintf(str, "%d", level);
			Text_level[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_level[first + i], str);

			sprintf(str, "%d", i);
			Text_node[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_node[first + i], str);

			get_set_by_level(level, i, rep);
			int_vec_print_to_str(str, rep, level);
			Text_orbit_reps[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_orbit_reps[first + i], str);
			
			get_orbit_length_and_stabilizer_order(i, level, 
				stab_order, orbit_length);
			stab_order.print_to_string(str);
			Text_stab_order[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_stab_order[first + i], str);
			
			orbit_length.print_to_string(str);
			Text_orbit_length[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_orbit_length[first + i], str);
			
			O = get_node_ij(level, i);
			schreier_vector_length = O->get_nb_of_live_points();
			sprintf(str, "%d", schreier_vector_length);
			Text_schreier_vector_length[first + i] = NEW_char(strlen(str) + 1);
			strcpy(Text_schreier_vector_length[first + i], str);
			}
		}
	Sp = NEW_OBJECT(spreadsheet);
	Sp->init_empty_table(Nb_orbits + 1, 7);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const char **)
			Text_level, "Level");
	Sp->fill_column_with_text(2, (const char **)
			Text_node, "Node");
	Sp->fill_column_with_text(3, (const char **)
			Text_orbit_reps, "Orbit rep");
	Sp->fill_column_with_text(4, (const char **)
			Text_stab_order, "Stab order");
	Sp->fill_column_with_text(5, (const char **)
			Text_orbit_length, "Orbit length");
	Sp->fill_column_with_text(6, (const char **)
			Text_schreier_vector_length, "Schreier vector length");

#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_int(rep);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_level[i]);
		}
	FREE_pchar(Text_level);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_node[i]);
		}
	FREE_pchar(Text_node);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_orbit_reps[i]);
		}
	FREE_pchar(Text_orbit_reps);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_stab_order[i]);
		}
	FREE_pchar(Text_stab_order);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_orbit_length[i]);
		}
	FREE_pchar(Text_orbit_length);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_char(Text_schreier_vector_length[i]);
		}
	FREE_pchar(Text_schreier_vector_length);
	
}

void poset_classification::make_spreadsheet_of_level_info(
		spreadsheet *&Sp, int max_depth)
{
	int nb_rows, Nb_orbits, nb_orbits, i, level;
	pchar *Text_label;
	pchar *Text_nb_orbits;
	pchar *Text_orbit_length_sum;
	pchar *Text_schreier_vector_length_sum;
	pchar *Text_binomial;
	longinteger_object stab_order, orbit_length,
		orbit_length_sum, orbit_length_total;
	longinteger_object a, a_total;
	longinteger_domain D;
	int schreier_vector_length_int;
	longinteger_object schreier_vector_length,
		schreier_vector_length_sum, schreier_vector_length_total;
	int *rep;
	char str[1000];
	poset_orbit_node *O;


	nb_rows = max_depth + 2; // one extra row for totals
	rep = NEW_int(max_depth);
	Text_label = NEW_pchar(nb_rows);
	Text_nb_orbits = NEW_pchar(nb_rows);
	Text_orbit_length_sum = NEW_pchar(nb_rows);
	Text_schreier_vector_length_sum = NEW_pchar(nb_rows);
	Text_binomial = NEW_pchar(nb_rows);

	Nb_orbits = 0;
	orbit_length_total.create(0);
	schreier_vector_length_total.create(0);
	a_total.create(0);
	
	for (level = 0; level <= max_depth; level++) {

		//first = first_oracle_node_at_level[level];
		nb_orbits = nb_orbits_at_level(level);


		sprintf(str, "%d", level);
		Text_label[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_label[level], str);

		sprintf(str, "%d", nb_orbits);
		Text_nb_orbits[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_nb_orbits[level], str);

		orbit_length_sum.create(0);
		schreier_vector_length_sum.create(0);

		for (i = 0; i < nb_orbits; i++) {
			
			get_orbit_length_and_stabilizer_order(i, level, 
				stab_order, orbit_length);

			D.add_in_place(orbit_length_sum, orbit_length);
			
			
			O = get_node_ij(level, i);
			schreier_vector_length_int = O->get_nb_of_live_points();


			if (schreier_vector_length_int <= 0) {
				schreier_vector_length_int = 1;
				}
			schreier_vector_length.create(schreier_vector_length_int);

			if (schreier_vector_length_int >= 0) {
				D.add_in_place(schreier_vector_length_sum,
						schreier_vector_length);
				}

			}

		//cout << "poset_classification::make_spreadsheet_of_level_info
		// computing binomial coeffcient" << endl;
		D.binomial(a, A2->degree, level, FALSE);

		Nb_orbits += nb_orbits;
		D.add_in_place(orbit_length_total, orbit_length_sum);
		D.add_in_place(schreier_vector_length_total,
				schreier_vector_length_sum);
		D.add_in_place(a_total, a);

		orbit_length_sum.print_to_string(str);
		Text_orbit_length_sum[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_orbit_length_sum[level], str);

		schreier_vector_length_sum.print_to_string(str);
		Text_schreier_vector_length_sum[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_schreier_vector_length_sum[level], str);

		a.print_to_string(str);
		Text_binomial[level] = NEW_char(strlen(str) + 1);
		strcpy(Text_binomial[level], str);

		}

	level = max_depth + 1;
	sprintf(str, "total");
	Text_label[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_label[level], str);

	sprintf(str, "%d", Nb_orbits);
	Text_nb_orbits[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_nb_orbits[level], str);

	orbit_length_total.print_to_string(str);
	Text_orbit_length_sum[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_orbit_length_sum[level], str);

	schreier_vector_length_total.print_to_string(str);
	Text_schreier_vector_length_sum[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_schreier_vector_length_sum[level], str);

	a_total.print_to_string(str);
	Text_binomial[level] = NEW_char(strlen(str) + 1);
	strcpy(Text_binomial[level], str);


	Sp = NEW_OBJECT(spreadsheet);
	Sp->init_empty_table(nb_rows + 1, 6);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const char **)
			Text_label, "Level");
	Sp->fill_column_with_text(2, (const char **)
			Text_nb_orbits, "Nb_orbits");
	Sp->fill_column_with_text(3, (const char **)
			Text_orbit_length_sum, "Orbit_length_sum");
	Sp->fill_column_with_text(4, (const char **)
			Text_schreier_vector_length_sum, "Schreier_vector_length_sum");
	Sp->fill_column_with_text(5, (const char **)
			Text_binomial, "Binomial");



#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_int(rep);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_label[i]);
		}
	FREE_pchar(Text_label);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_nb_orbits[i]);
		}
	FREE_pchar(Text_nb_orbits);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_orbit_length_sum[i]);
		}
	FREE_pchar(Text_orbit_length_sum);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_schreier_vector_length_sum[i]);
		}
	FREE_pchar(Text_schreier_vector_length_sum);
	for (i = 0; i < nb_rows; i++) {
		FREE_char(Text_binomial[i]);
		}
	FREE_pchar(Text_binomial);
	
}

void poset_classification::write_file(ofstream &fp,
		int depth_completed, int verbose_level)
{
	memory_object *m;
	int f_v = (verbose_level >= 1);
	int size0;
	int nb_group_elements = 0;
	
	
	if (f_v) {
		cout << "poset_classification::write_file "
				"depth_completed=" << depth_completed << endl;
		}
	size0 = calc_size_on_file(depth_completed, 0 /*verbose_level*/);
	if (f_v) {
		cout << "size on file = " << size0 << endl;
		}
	
	if (size0 > 100 * ONE_MILLION) {
		cout << "poset_classification::write_file" << endl;
		cout << "size on file = " << size0 << endl;
		cout << "the size is very big" << endl;
		}
	
	m = NEW_OBJECT(memory_object);
	m->alloc(size0, 0);
	m->used_length = 0;

	write_memory_object(depth_completed, m,
			nb_group_elements, 0 /*verbose_level*/);

	int size;
	size = m->used_length;

	
	fp.write((char *) &depth_completed, sizeof(int));
	fp.write((char *) &size, sizeof(int));
	fp.write(m->char_pointer, size);
	
	FREE_OBJECT(m);
	
	if (f_v) {
		cout << "poset_classification::write_file done" << endl;
		}
}

void poset_classification::read_file(ifstream &fp,
		int &depth_completed, int verbose_level)
{
	memory_object *m;
	int f_v = (verbose_level >= 1);
	int size;
	int nb_group_elements;
	
	if (f_v) {
		cout << "poset_classification::read_file" << endl;
		}


	fp.read((char *) &depth_completed, sizeof(int));
	fp.read((char *) &size, sizeof(int));

	if (f_v) {
		cout << "poset_classification::read_file size = " << size << endl;
		}
	
	m = NEW_OBJECT(memory_object);

	m->alloc(size, 0);
	m->used_length = 0;

	fp.read(m->char_pointer, size);
	

	m->used_length = size;
	m->cur_pointer = 0;

	if (f_v) {
		cout << "poset_classification::read_file "
				"before poset_classification_read_memory" << endl;
		}
	read_memory_object(depth_completed, m,
			nb_group_elements, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "poset_classification::read_file "
				"after poset_classification_read_memory" << endl;
		}

	FREE_OBJECT(m);
	
	if (f_v) {
		cout << "poset_classification::read_file done, "
				"depth_completed=" << depth_completed << endl;
		}
}





