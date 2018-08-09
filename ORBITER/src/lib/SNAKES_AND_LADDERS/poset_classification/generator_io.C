// generator_io.C
//
// Anton Betten
// moved here from DISCRETA/snakesandladders.C
// December 27, 2008
// renamed from io.C to generator_io.C Aug 24, 2011


#include "GALOIS/galois.h"
#include "ACTION/action.h"
#include "SNAKES_AND_LADDERS/snakesandladders.h"

void generator::housekeeping(INT i,
		INT f_write_files, INT t0, INT verbose_level)
{
	INT j, nb_nodes;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_embedded = TRUE;
	
	if (f_v) {
		cout << "generator::housekeeping "
				"level=" << i << endl;
		cout << "generator::housekeeping "
				"verbose_level=" << verbose_level << endl;
		cout << "generator::housekeeping "
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
		cout << "total: " << first_oracle_node_at_level[i + 1] << endl;
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
			INT f_with_strong_generators = FALSE;
			INT f_long_version = FALSE;
			write_lvl(cout, i, t0, f_with_strong_generators,
					f_long_version, verbose_level - 2);
			}
		}
	
	if (f_write_files) {
		BYTE my_fname_base[1000];
		
		if (f_v) {
			cout << "generator_housekeeping writing files" << endl;
			}
#if 1
		sprintf(my_fname_base, "%sa", fname_base);
		if (f_v) {
			cout << "generator_housekeeping "
					"my_fname_base=" << my_fname_base << endl;
			cout << "generator_housekeeping "
					"before write_level_file_binary" << endl;
			}
		write_level_file_binary(i, my_fname_base,
				0/*verbose_level*/);
		if (i) {		
			sprintf(my_fname_base, "%sb", fname_base);
			if (f_v) {
				cout << "generator_housekeeping "
						"my_fname_base=" << my_fname_base << endl;
				cout << "generator_housekeeping "
						"before write_level_file_binary" << endl;
				}
			write_level_file_binary(i - 1,
					my_fname_base, 0/*verbose_level*/);
			if (f_v) {
				cout << "generator_housekeeping "
						"my_fname_base=" << my_fname_base << endl;
				cout << "generator_housekeeping "
						"before write_sv_level_file_binary" << endl;
				}
			write_sv_level_file_binary(i - 1, my_fname_base, 
				FALSE, 0, 0, 0 /*verbose_level*/);
			}
#endif
		if (f_v) {
			cout << "generator_housekeeping "
					"before write_lvl_file" << endl;
			}
		write_lvl_file(fname_base, i, t0,
				FALSE /* f_with_strong_generators */,
				FALSE /* f_long_version */, 0);
		if (f_v) {
			cout << "generator_housekeeping "
					"before generator::write_data_file" << endl;
			}
		generator::write_data_file(i /* depth_completed */,
				fname_base, 0);

		if (f_v) {
			cout << "generator::housekeeping "
					"writing files done" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "generator_housekeeping "
					"not writing files" << endl;
			}
		}

	if (f_Log) {
		INT verbose_level = 1;
		INT f = first_oracle_node_at_level[i];
		INT len = nb_orbits_at_level(i);
		print_problem_label();
		cout << "There are " << len
				<< " nodes at level " << i << ":" << endl;
		for (j = 0; j < len; j++) {
			Log_nodes(f + j, i, cout, FALSE, verbose_level);
			}
		}

	if (f_log && i == sz) {
		INT verbose_level = 1;
		INT ii;

		for (ii = 0; ii <= sz; ii++) {
			INT f = first_oracle_node_at_level[ii];
			INT len = nb_orbits_at_level(ii);
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
			cout << "generator::housekeeping "
					"before write_treefile_and_draw_tree" << endl;
			}

		write_treefile_and_draw_tree(fname_base, i, 
			xmax, ymax, radius, f_embedded, 0 /*verbose_level - 1*/);
			// in generator_draw.C

		if (f_v) {
			cout << "generator::housekeeping "
					"after write_treefile_and_draw_tree" << endl;
			}
		}
	else {
		if (f_v) {
			cout << "generator_housekeeping "
					"not writing tree" << endl;
			}
		}

	if (f_v) {
		cout << "generator::housekeeping done" << endl;
		}
}


void generator::housekeeping_no_data_file(INT i,
		INT t0, INT verbose_level)
{
	INT j;
	INT f_v = (verbose_level >= 1);
	INT f_embedded = TRUE;
	
	if (f_v) {
		cout << "generator::housekeeping_no_data_file "
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
		cout << "total: " << first_oracle_node_at_level[i + 1] << endl;
		compute_and_print_automorphism_group_orders(i, cout);
		}

	if (f_W || (f_w && i == sz)) {
		BYTE fname_base2[1000];
		
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
		
		//generator_write_data_file(gen,
		// i /* depth_completed */, gen->fname_base, 0);

		}

	if (f_T || (f_t && i == sz)) {
		write_treefile_and_draw_tree(fname_base, i, 
			xmax, ymax, radius, f_embedded, verbose_level - 1);
		}
	if (f_v) {
		cout << "generator::housekeeping_no_data_file done" << endl;
		}
}

INT generator::test_sv_level_file_binary(INT level, BYTE *fname_base)
{
	BYTE fname[1000];
	
	sprintf(fname, "%s_lvl_%ld_sv.data", fname_base, level);
	if (file_size(fname) >= 1)
		return TRUE;
	else
		return FALSE;
}

void generator::read_sv_level_file_binary(
	INT level, BYTE *fname_base,
	INT f_split, INT split_mod, INT split_case, 
	INT f_recreate_extensions, INT f_dont_keep_sv, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];
	
	sprintf(fname, "%s_lvl_%ld_sv.data", fname_base, level);
	
	if (f_v) {
		cout << "generator::read_sv_level_file_binary "
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

void generator::write_sv_level_file_binary(
	INT level, BYTE *fname_base,
	INT f_split, INT split_mod, INT split_case, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];
	
	sprintf(fname, "%s_lvl_%ld_sv.data", fname_base, level);
	
	if (f_v) {
		cout << "generator::write_sv_level_file_binary "
				"fname = " << fname << endl;
		}


	FILE *fp;

	fp = fopen(fname, "wb");

	write_sv_level_file_binary2(level, fp, 
		f_split, split_mod, split_case, 
		verbose_level);
	
	fclose(fp);

	if (f_v) {
		cout << "generator::write_sv_level_file_binary "
			"finished written file "
			<< fname << " of size " << file_size(fname) << endl;
		}
}

void generator::read_sv_level_file_binary2(INT level, FILE *fp, 
	INT f_split, INT split_mod, INT split_case, 
	INT f_recreate_extensions, INT f_dont_keep_sv, 
	INT verbose_level)
{
	INT f, i, nb_nodes;
	INT f_v = (verbose_level >= 1);
	INT4 I;
	
	f = first_oracle_node_at_level[level];
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "generator::read_sv_level_file_binary2 "
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
	I = fread_INT4(fp);
	if (I != 1) { 
		cout << "generator::read_sv_level_file_binary2: "
				"unknown file version" << endl;
		exit(1);
		}
	I = fread_INT4(fp);
	if (I != level) {
		cout << "generator::read_sv_level_file_binary2: "
				"level does not match" << endl;
		exit(1);
		}
	I = fread_INT4(fp);
	if (I != nb_nodes) {
		cout << "generator::read_sv_level_file_binary2: "
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
			FREE_INT(root[f + i].sv);
			root[f + i].sv = NULL;
			}
		}
	I = fread_INT4(fp);
	if (I != MAGIC_SYNC) {
		cout << "generator::read_sv_level_file_binary2: "
				"MAGIC_SYNC does not match" << endl;
		exit(1);
		}
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "generator::read_sv_level_file_binary2 "
				"finished" << endl;
		}
}

void generator::write_sv_level_file_binary2(INT level, FILE *fp, 
	INT f_split, INT split_mod, INT split_case, 
	INT verbose_level)
{
	INT f, i, nb_nodes;
	INT f_v = (verbose_level >= 1);
	
	f = first_oracle_node_at_level[level];
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "generator::write_sv_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		}
	// version number of this file format
	fwrite_INT4(fp, 1);
	fwrite_INT4(fp, level);
	fwrite_INT4(fp, nb_nodes);
	for (i = 0; i < nb_nodes; i++) {
		if (f_split) {
			if ((i % split_mod) != split_case)
				continue;
			}
		root[f + i].sv_write_file(fp, verbose_level - 2);
		}
	fwrite_INT4(fp, MAGIC_SYNC);
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "generator::write_sv_level_file_binary2 finished" << endl;
		}
}

void generator::read_level_file_binary(INT level,
		BYTE *fname_base, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];
	INT nb_group_elements;
	
	sprintf(fname, "%s_lvl_%ld.data", fname_base, level);
	
	if (f_v) {
		cout << "generator::read_level_file_binary "
				"reading file " << fname << " of size "
				<< file_size(fname) << endl;
		}

	if (file_size(fname) < 0) {
		cout << "generator::read_level_file_binary "
				"probems while reading file " << fname << endl;
		exit(1);
		}


	FILE *fp;

	fp = fopen(fname, "rb");

	read_level_file_binary2(level, fp, nb_group_elements,
			verbose_level - 1);
	
	fclose(fp);

}

void generator::write_level_file_binary(INT level,
		BYTE *fname_base, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];
	INT nb_group_elements;
	
	sprintf(fname, "%s_lvl_%ld.data", fname_base, level);
	
	if (f_v) {
		cout << "generator::write_level_file_binary "
				"fname = " << fname << endl;
		}


	FILE *fp;

	fp = fopen(fname, "wb");

	write_level_file_binary2(level, fp,
			nb_group_elements, verbose_level);
	
	fclose(fp);

	if (f_v) {
		cout << "generator::write_level_file_binary "
				"finished written file "
			<< fname << " of size " << file_size(fname) 
			<< " nb_group_elements=" << nb_group_elements << endl;
		}
}

void generator::read_level_file_binary2(
	INT level, FILE *fp,
	INT &nb_group_elements, INT verbose_level)
{
	INT f, i, nb_nodes, magic_sync;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT4 I;
	
	if (f_v) {
		cout << "generator::read_level_file_binary2" << endl;
		}
	f = first_oracle_node_at_level[level];
	nb_group_elements = 0;
	I = fread_INT4(fp);
	if (I != 1) {
		cout << "generator::read_level_file_binary2 "
				"version = " << I << " unknown" << endl;
		exit(1);
		}

	I = fread_INT4(fp);
	if (I != level) {
		cout << "generator::read_level_file_binary2 "
				"level = " << I << " should be " << level << endl;
		exit(1);
		}

	nb_nodes = fread_INT4(fp);
	if (f_v) {
		cout << "generator::read_level_file_binary, "
				"nb_nodes = " << nb_nodes << endl;
		}
	first_oracle_node_at_level[level + 1] = f + nb_nodes;
	
	if (f_v) {
		cout << "generator::read_level_file_binary2 "
				"f + nb_nodes = " << f + nb_nodes << endl;
		cout << "generator::read_level_file_binary2 "
				"nb_oracle_nodes_allocated = "
			<< nb_oracle_nodes_allocated << endl;
		}
	if (f + nb_nodes > nb_oracle_nodes_allocated) {
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
	magic_sync = fread_INT4(fp);
	if (magic_sync != MAGIC_SYNC) {
		cout << "generator::read_level_file_binary2 "
				"could not read MAGIC_SYNC, file is corrupt" << endl;
		cout << "MAGIC_SYNC=" << MAGIC_SYNC << endl;
		cout << "we read   =" << magic_sync << endl;		
		exit(1);
		}
	if (f_v) {
		cout << "generator::read_level_file_binary2 "
				"finished ";
		cout << "level=" << level 
			<< ", with " << nb_nodes << " nodes" 
			<< " and " << nb_group_elements << " group elements"
			<< endl;
		}
}

void generator::write_level_file_binary2(
	INT level, FILE *fp,
	INT &nb_group_elements, INT verbose_level)
{
	INT f, i, nb_nodes;
	INT f_v = FALSE;//(verbose_level >= 1);
	
	f = first_oracle_node_at_level[level];
	nb_nodes = nb_orbits_at_level(level);
	if (f_v) {
		cout << "generator::write_level_file_binary2 "
				<< nb_nodes << " nodes" << endl;
		}
	nb_group_elements = 0;
	// version number of this file format
	fwrite_INT4(fp, 1);
	fwrite_INT4(fp, level);
	fwrite_INT4(fp, nb_nodes);
	for (i = 0; i < nb_nodes; i++) {
		root[f + i].write_file(A, fp,
				nb_group_elements, verbose_level - 2);
		}
	fwrite_INT4(fp, MAGIC_SYNC);
	// a check to see if the file is not corrupt
	if (f_v) {
		cout << "generator::write_level_file_binary2 "
				"finished" << endl;
		}
}

INT generator::calc_size_on_file(INT depth_completed,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, s = 0;
	INT nb_nodes;
	
	if (f_v) {
		cout << "generator::calc_size_on_file "
				"depth_completed=" << depth_completed << endl;
		}
	nb_nodes = first_oracle_node_at_level[depth_completed + 1];
	s += 3 * 4;
	for (i = 0; i <= depth_completed + 1; i++) {
		s += 4;
		}
	for (i = 0; i < nb_nodes; i++) {
		s += root[i].calc_size_on_file(A, verbose_level);
		}
	s += 4; // MAGIC_SYNC
	if (f_v) {
		cout << "generator::calc_size_on_file "
				"depth_completed=" << depth_completed
				<< " s=" << s << endl;
		}
	return s;
}

void generator::make_fname_candidates_file_default(
		BYTE *fname, INT level)
{
	sprintf(fname, "%s_lvl_%ld_candidates.bin", fname_base, level);
}

void generator::write_candidates_binary_using_sv(BYTE *fname_base,
		INT lvl, INT t0, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE;
	BYTE fname1[1000];
	
	if (f_v) {
		cout << "generator::write_candidates_binary_using_sv "
				"lvl=" << lvl << " fname_base=" << fname_base << endl;
		}
	sprintf(fname1, "%s_lvl_%ld_candidates.bin", fname_base, lvl);
	{
	INT fst, len;
	INT *nb_cand;
	INT *cand_first;
	INT total_nb_cand = 0;
	INT *subset;
	INT *Cand;
	INT i, j, node, nb, pos;

	fst = first_oracle_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "generator::write_candidates_binary_using_sv "
				"first node at level " << lvl << " is " << fst << endl;
		cout << "generator::write_candidates_binary_using_sv "
				"number of nodes at level " << lvl << " is " << len << endl;
		}
	nb_cand = NEW_INT(len);
	cand_first = NEW_INT(len);
	for (i = 0; i < len; i++) {
		node = fst + i;
		INT *osv = root[node].sv;

		if (f_vv) {
			cout << "generator::write_candidates_binary_using_sv "
					"node " << i << " / " << len << endl;
			}
		if (osv == NULL) {
			cout << "generator::write_candidates_binary_using_sv "
					"osv == NULL, we don't have a Schreier vector; "
					"maybe set -schreier_depth <depth>" << endl;
			exit(1);
			}
		nb = osv[0];
		if (FALSE) {
			cout << "generator::write_candidates_binary_using_sv "
					"i=" << i << endl;
			cout << "generator::write_candidates_binary_using_sv "
					"nb=" << nb << endl;
			}
		subset = osv + 1;
		nb_cand[i] = nb;
		total_nb_cand += nb;
		}
	if (f_vv) {
		cout << "generator::write_candidates_binary_using_sv "
				"total_nb_cand=" << total_nb_cand << endl;
		}
	Cand = NEW_INT(total_nb_cand);
	pos = 0;
	for (i = 0; i < len; i++) {
		node = fst + i;
		INT *osv = root[node].sv;
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

	fwrite_INT4(fp, len);
	for (i = 0; i < len; i++) {
		fwrite_INT4(fp, nb_cand[i]);
		fwrite_INT4(fp, cand_first[i]);
		}
	for (i = 0; i < total_nb_cand; i++) {
		fwrite_INT4(fp, Cand[i]);
		}


	fclose(fp);


	FREE_INT(nb_cand);
	FREE_INT(cand_first);
	FREE_INT(Cand);
	}
	if (f_v) {
		cout << "written file " << fname1 << " of size "
				<< file_size(fname1) << endl;
		}
}

void generator_read_candidates_of_orbit(
	const BYTE *fname, INT orbit_at_level,
	INT *&candidates, INT &nb_candidates, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	FILE *fp;
	INT nb, cand_first, i;


	if (f_v) {
		cout << "generator_read_candidates_of_orbit" << endl;
		cout << "verbose_level=" << verbose_level << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
		}
	
	if (file_size(fname) <= 0) {
		cout << "generator_read_candidates_of_orbit file "
				<< fname << " does not exist" << endl;
		exit(1);
		}

	fp = fopen(fname, "rb");

	nb = fread_INT4(fp);
	if (orbit_at_level >= nb) {
		cout << "generator_read_candidates_of_orbit "
				"orbit_at_level >= nb" << endl;
		cout << "orbit_at_level=" << orbit_at_level << endl;
		cout << "nb=" << nb << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "seeking position "
				<< (1 + orbit_at_level * 2) * sizeof(INT4) << endl;
		}
	fseek(fp, (1 + orbit_at_level * 2) * sizeof(INT4), SEEK_SET);
	nb_candidates = fread_INT4(fp);
	if (f_vv) {
		cout << "nb_candidates=" << nb_candidates << endl;
		}
	cand_first = fread_INT4(fp);
	if (f_v) {
		cout << "cand_first=" << cand_first << endl;
		}
	candidates = NEW_INT(nb_candidates);
	fseek(fp, (1 + nb * 2 + cand_first) * sizeof(INT4), SEEK_SET);
	for (i = 0; i < nb_candidates; i++) {
		candidates[i] = fread_INT4(fp);
		}
	fclose(fp);
	if (f_v) {
		cout << "generator_read_candidates_of_orbit done" << endl;
		}
}

void generator::read_level_file(INT level,
		BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *set_sizes;
	INT **sets;
	BYTE **data;
	INT nb_cases;
	INT nb_nodes, first_at_level;
	INT i, I, J;
	oracle *O;
	
	if (f_v) {
		cout << "generator::read_level_file "
				"fname=" << fname << endl;
		}
	
#if 1
	read_and_parse_data_file(fname, nb_cases,
			data, sets, set_sizes, verbose_level - 1);
	
#else

	if (try_to_read_file(fname, nb_cases, data)) {
		if (f_v) {
			cout << "generator::read_level_file read "
					"file " << fname << " nb_cases = " << nb_cases << endl;
			}
		}
	else {
		cout << "generator::read_level_file "
				"couldn't read file " << fname << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "generator::read_level_file parsing" << endl;
		}
	parse_sets(nb_cases, data, set_sizes, sets);
	if (f_vv) {
		cout << "generator::read_level_file "
				"parsing finished" << endl;
		}
		// in GALOIS/util.C
#endif

	first_at_level = first_oracle_node_at_level[level];
	nb_nodes = first_at_level + nb_cases;
	
	if (nb_nodes > nb_oracle_nodes_allocated) {
		if (f_vv) {
			cout << "generator::read_level_file "
					"reallocating to " << nb_nodes << " nodes" << endl;
			}
		reallocate_to(nb_nodes, verbose_level - 1);
		}
	first_oracle_node_at_level[level + 1] = nb_nodes;
	for (i = 0; i < nb_cases; i++) {
		I = first_at_level + i;
		O = &root[I];
		
		cout << setw(10) << i << " : ";
		INT_vec_print(cout, sets[i], level);
		cout << endl;
		
		J = find_oracle_node_for_set(level - 1,
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
		
			// now strong generators are available
		
			Aut.schreier_sims(0);
		
			cout << "the automorphism group has order ";
			Aut.print_group_order(cout);
			cout << endl;
		
			strong_generators *Strong_gens;

			Strong_gens = new strong_generators;
			Strong_gens->init_from_sims(Aut.S, 0);

#if 0
			cout << "and is strongly generated by the "
					"following " << Aut.SG->len << " elements:" << endl;

			Aut.SG->print(cout);
			cout << endl;
#endif
			O->store_strong_generators(this, Strong_gens);
			cout << "strong generators stored" << endl;

			delete Strong_gens;
			}
		else {
			//cout << "trivial group" << endl;
			//Aut.init_strong_generators_empty_set();
			
			}
		}

		}
	delete [] set_sizes;
	if (f_v) {
		cout << "generator::read_level_file "
				"fname=" << fname << " done" << endl;
		}
}

void generator::read_data_file(INT &depth_completed,
		const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT size;
	INT nb_group_elements;
	memory_object *m;
	
	
	if (f_v) {
		cout << "generator::read_data_file "
				"fname = " << fname << endl;
		cout << "A->elt_size_in_INT = "
				<< A->elt_size_in_INT << endl;
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
		cout << "generator::read_data_file before m->alloc" << endl;
		}

	m = new memory_object;
	m->alloc(size, 0);

	if (f_v) {
		cout << "generator::read_data_file after m->alloc" << endl;
		}
	
	m->used_length = 0;

	FILE *fp;
	
	fp = fopen(fname, "r");

	fread(m->char_pointer, 1 /* size */, size /* items */, fp);
	
	fclose(fp);

	if (f_v) {
		cout << "generator::read_data_file after fread" << endl;
		}

	m->used_length = size;
	m->cur_pointer = 0;

	if (f_v) {
		cout << "generator::read_data_file "
				"before generator_read_memory" << endl;
		}
	read_memory_object(depth_completed, m,
			nb_group_elements, verbose_level - 2);
	if (f_v) {
		cout << "generator::read_data_file "
				"after generator_read_memory" << endl;
		}

	delete m;
	if (f_v) {
		cout << "generator::read_data_file done" <<endl;
		}
	
}

void generator::write_data_file(INT depth_completed,
		const BYTE *fname_base, INT verbose_level)
{
	memory_object *m;
	INT f_v = (verbose_level >= 1);
	BYTE fname[1000];
	INT nb_group_elements;
	INT size0;
	
	sprintf(fname, "%s_%ld.data", fname_base, depth_completed);
	
	if (f_v) {
		cout << "generator::write_data_file fname = " << fname << endl;
		cout << "A->elt_size_in_INT = " << A->elt_size_in_INT << endl;
		cout << "A->coded_elt_size_in_char = "
				<< A->coded_elt_size_in_char << endl;
		}
	size0 = calc_size_on_file(depth_completed, verbose_level);
	if (f_v) {
		cout << "size on file = " << size0 << endl;
		}
	
	if (size0 > 100 * ONE_MILLION) {
		cout << "generator::write_data_file file=" << fname << endl;
		cout << "size on file = " << size0 << endl;
		cout << "the size is very big" << endl;
		}
	
	m = new memory_object;
	m->alloc(size0, 0);
	m->used_length = 0;

	write_memory_object(depth_completed, m,
			nb_group_elements, verbose_level);

	FILE *fp;
	INT size;

	size = m->used_length;
	
	fp = fopen(fname, "wb");

	fwrite(m->char_pointer, 1 /* size */, size /* items */, fp);
	
	fclose(fp);

	delete m;
	
	if (f_v) {
		cout << "generator::write_data_file finished written file " 
			<< fname << " of size " << file_size(fname) 
			<< " nb_group_elements=" << nb_group_elements << endl;
		}
}

void generator::read_memory_object(INT &depth_completed,
		memory_object *m, INT &nb_group_elements,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, nb_nodes, version, magic_sync;
	
	//nb_nodes = first_oracle_node_at_level[depth_completed + 1];
	if (f_v) {
		cout << "generator::read_memory_object, "
				"data size (in bytes) = " << m->used_length << endl;
		}
	nb_group_elements = 0;
	m->read_int(&version);
	if (version != 1) {
		cout << "generator::read_memory_object "
				"version = " << version << " unknown" << endl;
		exit(1);
		}
	m->read_int(&depth_completed);
	if (f_v) {
		cout << "generator::read_memory_object "
				"depth_completed = " << depth_completed << endl;
		}

	if (depth_completed > sz) {
		cout << "generator::read_memory_object "
				"depth_completed > sz" << endl;
		exit(1);
		}

	m->read_int(&nb_nodes);
	if (f_v) {
		cout << "generator::read_memory_object "
				"nb_nodes = " << nb_nodes << endl;
		}

	//G->init_oracle(nb_nodes);

#if 1
	if (nb_nodes > nb_oracle_nodes_allocated) {
		reallocate_to(nb_nodes, verbose_level - 1);
		}
#endif
	for (i = 0; i <= depth_completed + 1; i++) {
		m->read_int(&first_oracle_node_at_level[i]);
		}
	for (i = 0; i < nb_nodes; i++) {
		if ((f_v && ((i % 1000) == 0)) || f_vv) {
			cout << "reading node " << i << endl;
			}
		root[i].read_memory_object(A, m,
				nb_group_elements, verbose_level - 1);
		}
	if (f_v) {
		cout << "generator::read_memory_object "
				"reading nodes completed" << endl;
		}
	m->read_int(&magic_sync);
	if (magic_sync != MAGIC_SYNC) {
		cout << "generator::read_memory_object "
				"could not read MAGIC_SYNC, file is corrupt" << endl;
		exit(1);
		}
	nb_oracle_nodes_used = nb_nodes;
	if (f_v) {
		cout << "generator::read_memory_object finished ";
		cout << "depth_completed=" << depth_completed 
			<< ", with " << nb_nodes << " nodes" 
			<< " and " << nb_group_elements << " group elements"
			<< endl;
		}
}

void generator::write_memory_object(INT depth_completed,
		memory_object *m, INT &nb_group_elements, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, nb_nodes;
	
	nb_nodes = first_oracle_node_at_level[depth_completed + 1];
	if (f_v) {
		cout << "generator::write_memory_object "
				<< nb_nodes << " nodes" << endl;
		}
	nb_group_elements = 0;
	m->write_int(1); // version number of this file format
	m->write_int(depth_completed);
	m->write_int(nb_nodes);
	for (i = 0; i <= depth_completed + 1; i++) {
		m->write_int(first_oracle_node_at_level[i]);
		}
	for (i = 0; i < nb_nodes; i++) {
		root[i].write_memory_object(A, m,
				nb_group_elements, verbose_level - 2);
		}
	m->write_int(MAGIC_SYNC); // a check to see if the file is not corrupt
	if (f_v) {
		cout << "generator::write_memory_object "
				"finished, data size (in bytes) = "
				<< m->used_length << endl;
		}
}

void generator::recover(const BYTE *recover_fname,
		INT &depth_completed, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generator::recover "
				"recovering from file " << recover_fname << endl;
		}
	read_data_file(depth_completed, recover_fname, verbose_level);
	if (f_v) {
		cout << "generator::recover "
				"recovering finished, "
				"depth_completed = " << depth_completed << endl;
		}
}

void generator::write_lvl_file_with_candidates(
		BYTE *fname_base, INT lvl, INT t0,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname1[1000];
	
	sprintf(fname1, "%s_lvl_%ld_candidates.txt", fname_base, lvl);
	{
	ofstream f(fname1);
	INT cur;
	
	//f << "# " << lvl << endl; 
	for (cur = first_oracle_node_at_level[lvl]; 
		cur < first_oracle_node_at_level[lvl + 1]; cur++) {
		root[cur].log_current_node_with_candidates(
				this, lvl, f, verbose_level - 2);
		}
	f << "-1 " << first_oracle_node_at_level[lvl + 1]
				- first_oracle_node_at_level[lvl]
		<< " " << first_oracle_node_at_level[lvl] << " in ";
	time_check(f, t0);
	f << endl;
	f << "# in action " << A->label << endl;
	}
	if (f_v) {
		cout << "written file " << fname1
				<< " of size " << file_size(fname1) << endl;
		}
}


void generator::write_lvl_file(BYTE *fname_base,
		INT lvl, INT t0, INT f_with_stabilizer_generators,
		INT f_long_version,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	BYTE fname1[1000];
	sprintf(fname1, "%s_lvl_%ld", fname_base, lvl);
	{
	ofstream f(fname1);
	INT i, fst, len;
	

	fst = first_oracle_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);

	f << "# " << lvl << endl; 
	for (i = 0; i < len; i++) {
		root[fst + i].log_current_node(this,
				lvl, f, f_with_stabilizer_generators,
				f_long_version);
		}
	f << "-1 " << len << " "
			<< first_oracle_node_at_level[lvl] << " in ";
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

void generator::write_lvl(ostream &f, INT lvl, INT t0,
		INT f_with_stabilizer_generators, INT f_long_version,
		INT verbose_level)
{
	//INT f_v = (verbose_level >= 1);
	INT i;
	INT fst, len;


	fst = first_oracle_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);
	
	f << "# " << lvl << endl; 
	for (i = 0; i < len; i++) {
		root[fst + i].log_current_node(this, lvl, f,
				f_with_stabilizer_generators, f_long_version);
		}
	f << "-1 " << len << " " << first_oracle_node_at_level[lvl]
		<< " in ";
	time_check(f, t0);
	f << endl;
	compute_and_print_automorphism_group_orders(lvl, f);
	f << endl;
	f << "# in action " << A->label << endl;
}

void generator::log_nodes_for_treefile(INT cur, INT depth,
		ostream &f, INT f_recurse, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, next;
	oracle *node = &root[cur];
		

	if (f_v) {
		cout << "generator::log_nodes_for_treefile "
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

void generator::Log_nodes(INT cur, INT depth,
		ostream &f, INT f_recurse,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, next;
	oracle *node = &root[cur];
		

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
				<< " strong generators:" << endl;
		if (f_v) {
			cout << "Log_nodes cur=" << cur
					<< " printing strong generators" << endl;
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
			INT_vec_print(f, node->tl, A->base_len);
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

void generator::log_current_node(ostream &f, INT size)
{
	//longinteger_object go;
	INT i;
	

	f << size << " ";
	for (i = 0; i < size; i++) {
		f << S[i] << " ";
		}
	f << endl;

}



void generator::make_spreadsheet_of_orbit_reps(
		spreadsheet *&Sp, INT max_depth)
{
	INT Nb_orbits, nb_orbits, i, level, first;
	PBYTE *Text_level;
	PBYTE *Text_node;
	PBYTE *Text_orbit_reps;
	PBYTE *Text_stab_order;
	PBYTE *Text_orbit_length;
	PBYTE *Text_schreier_vector_length;
	longinteger_object stab_order, orbit_length;
	INT schreier_vector_length;
	INT *rep;
	BYTE str[1000];
	oracle *O;

	Nb_orbits = 0;
	for (level = 0; level <= max_depth; level++) {
		Nb_orbits += nb_orbits_at_level(level);
		}

	rep = NEW_INT(max_depth);
	Text_level = NEW_PBYTE(Nb_orbits);
	Text_node = NEW_PBYTE(Nb_orbits);
	Text_orbit_reps = NEW_PBYTE(Nb_orbits);
	Text_stab_order = NEW_PBYTE(Nb_orbits);
	Text_orbit_length = NEW_PBYTE(Nb_orbits);
	Text_schreier_vector_length = NEW_PBYTE(Nb_orbits);

	first = 0;
	for (level = 0; level <= max_depth; level++) {
		first = first_oracle_node_at_level[level];
		nb_orbits = nb_orbits_at_level(level);
		for (i = 0; i < nb_orbits; i++) {
			sprintf(str, "%ld", level);
			Text_level[first + i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Text_level[first + i], str);

			sprintf(str, "%ld", i);
			Text_node[first + i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Text_node[first + i], str);

			get_set_by_level(level, i, rep);
			INT_vec_print_to_str(str, rep, level);
			Text_orbit_reps[first + i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Text_orbit_reps[first + i], str);
			
			get_orbit_length_and_stabilizer_order(i, level, 
				stab_order, orbit_length);
			stab_order.print_to_string(str);
			Text_stab_order[first + i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Text_stab_order[first + i], str);
			
			orbit_length.print_to_string(str);
			Text_orbit_length[first + i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Text_orbit_length[first + i], str);
			
			O = get_node_ij(level, i);
			schreier_vector_length = O->get_nb_of_live_points();
			sprintf(str, "%ld", schreier_vector_length);
			Text_schreier_vector_length[first + i] = NEW_BYTE(strlen(str) + 1);
			strcpy(Text_schreier_vector_length[first + i], str);
			}
		}
	Sp = new spreadsheet;
	Sp->init_empty_table(Nb_orbits + 1, 7);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const BYTE **)
			Text_level, "Level");
	Sp->fill_column_with_text(2, (const BYTE **)
			Text_node, "Node");
	Sp->fill_column_with_text(3, (const BYTE **)
			Text_orbit_reps, "Orbit rep");
	Sp->fill_column_with_text(4, (const BYTE **)
			Text_stab_order, "Stab order");
	Sp->fill_column_with_text(5, (const BYTE **)
			Text_orbit_length, "Orbit length");
	Sp->fill_column_with_text(6, (const BYTE **)
			Text_schreier_vector_length, "Schreier vector length");

#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_INT(rep);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_BYTE(Text_level[i]);
		}
	FREE_PBYTE(Text_level);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_BYTE(Text_node[i]);
		}
	FREE_PBYTE(Text_node);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_BYTE(Text_orbit_reps[i]);
		}
	FREE_PBYTE(Text_orbit_reps);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_BYTE(Text_stab_order[i]);
		}
	FREE_PBYTE(Text_stab_order);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_BYTE(Text_orbit_length[i]);
		}
	FREE_PBYTE(Text_orbit_length);
	for (i = 0; i < Nb_orbits; i++) {
		FREE_BYTE(Text_schreier_vector_length[i]);
		}
	FREE_PBYTE(Text_schreier_vector_length);
	
}

void generator::make_spreadsheet_of_level_info(
		spreadsheet *&Sp, INT max_depth)
{
	INT nb_rows, Nb_orbits, nb_orbits, i, level;
	PBYTE *Text_label;
	PBYTE *Text_nb_orbits;
	PBYTE *Text_orbit_length_sum;
	PBYTE *Text_schreier_vector_length_sum;
	PBYTE *Text_binomial;
	longinteger_object stab_order, orbit_length,
		orbit_length_sum, orbit_length_total;
	longinteger_object a, a_total;
	longinteger_domain D;
	INT schreier_vector_length_INT;
	longinteger_object schreier_vector_length,
		schreier_vector_length_sum, schreier_vector_length_total;
	INT *rep;
	BYTE str[1000];
	oracle *O;


	nb_rows = max_depth + 2; // one extra row for totals
	rep = NEW_INT(max_depth);
	Text_label = NEW_PBYTE(nb_rows);
	Text_nb_orbits = NEW_PBYTE(nb_rows);
	Text_orbit_length_sum = NEW_PBYTE(nb_rows);
	Text_schreier_vector_length_sum = NEW_PBYTE(nb_rows);
	Text_binomial = NEW_PBYTE(nb_rows);

	Nb_orbits = 0;
	orbit_length_total.create(0);
	schreier_vector_length_total.create(0);
	a_total.create(0);
	
	for (level = 0; level <= max_depth; level++) {

		//first = first_oracle_node_at_level[level];
		nb_orbits = nb_orbits_at_level(level);


		sprintf(str, "%ld", level);
		Text_label[level] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_label[level], str);

		sprintf(str, "%ld", nb_orbits);
		Text_nb_orbits[level] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_nb_orbits[level], str);

		orbit_length_sum.create(0);
		schreier_vector_length_sum.create(0);

		for (i = 0; i < nb_orbits; i++) {
			
			get_orbit_length_and_stabilizer_order(i, level, 
				stab_order, orbit_length);

			D.add_in_place(orbit_length_sum, orbit_length);
			
			
			O = get_node_ij(level, i);
			schreier_vector_length_INT = O->get_nb_of_live_points();


			if (schreier_vector_length_INT <= 0) {
				schreier_vector_length_INT = 1;
				}
			schreier_vector_length.create(schreier_vector_length_INT);

			if (schreier_vector_length_INT >= 0) {
				D.add_in_place(schreier_vector_length_sum,
						schreier_vector_length);
				}

			}

		//cout << "generator::make_spreadsheet_of_level_info
		// computing binomial coeffcient" << endl;
		D.binomial(a, A2->degree, level, FALSE);

		Nb_orbits += nb_orbits;
		D.add_in_place(orbit_length_total, orbit_length_sum);
		D.add_in_place(schreier_vector_length_total,
				schreier_vector_length_sum);
		D.add_in_place(a_total, a);

		orbit_length_sum.print_to_string(str);
		Text_orbit_length_sum[level] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_orbit_length_sum[level], str);

		schreier_vector_length_sum.print_to_string(str);
		Text_schreier_vector_length_sum[level] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_schreier_vector_length_sum[level], str);

		a.print_to_string(str);
		Text_binomial[level] = NEW_BYTE(strlen(str) + 1);
		strcpy(Text_binomial[level], str);

		}

	level = max_depth + 1;
	sprintf(str, "total");
	Text_label[level] = NEW_BYTE(strlen(str) + 1);
	strcpy(Text_label[level], str);

	sprintf(str, "%ld", Nb_orbits);
	Text_nb_orbits[level] = NEW_BYTE(strlen(str) + 1);
	strcpy(Text_nb_orbits[level], str);

	orbit_length_total.print_to_string(str);
	Text_orbit_length_sum[level] = NEW_BYTE(strlen(str) + 1);
	strcpy(Text_orbit_length_sum[level], str);

	schreier_vector_length_total.print_to_string(str);
	Text_schreier_vector_length_sum[level] = NEW_BYTE(strlen(str) + 1);
	strcpy(Text_schreier_vector_length_sum[level], str);

	a_total.print_to_string(str);
	Text_binomial[level] = NEW_BYTE(strlen(str) + 1);
	strcpy(Text_binomial[level], str);


	Sp = new spreadsheet;
	Sp->init_empty_table(nb_rows + 1, 6);
	Sp->fill_column_with_row_index(0, "Line");
	Sp->fill_column_with_text(1, (const BYTE **)
			Text_label, "Level");
	Sp->fill_column_with_text(2, (const BYTE **)
			Text_nb_orbits, "Nb_orbits");
	Sp->fill_column_with_text(3, (const BYTE **)
			Text_orbit_length_sum, "Orbit_length_sum");
	Sp->fill_column_with_text(4, (const BYTE **)
			Text_schreier_vector_length_sum, "Schreier_vector_length_sum");
	Sp->fill_column_with_text(5, (const BYTE **)
			Text_binomial, "Binomial");



#if 0
	cout << "before Sp->save " << fname_csv << endl;
	Sp->save(fname_csv, verbose_level);
	cout << "after Sp->save " << fname_csv << endl;
#endif

	FREE_INT(rep);
	for (i = 0; i < nb_rows; i++) {
		FREE_BYTE(Text_label[i]);
		}
	FREE_PBYTE(Text_label);
	for (i = 0; i < nb_rows; i++) {
		FREE_BYTE(Text_nb_orbits[i]);
		}
	FREE_PBYTE(Text_nb_orbits);
	for (i = 0; i < nb_rows; i++) {
		FREE_BYTE(Text_orbit_length_sum[i]);
		}
	FREE_PBYTE(Text_orbit_length_sum);
	for (i = 0; i < nb_rows; i++) {
		FREE_BYTE(Text_schreier_vector_length_sum[i]);
		}
	FREE_PBYTE(Text_schreier_vector_length_sum);
	for (i = 0; i < nb_rows; i++) {
		FREE_BYTE(Text_binomial[i]);
		}
	FREE_PBYTE(Text_binomial);
	
}

void generator::write_file(ofstream &fp,
		INT depth_completed, INT verbose_level)
{
	memory_object *m;
	INT f_v = (verbose_level >= 1);
	INT size0;
	INT nb_group_elements = 0;
	
	
	if (f_v) {
		cout << "generator::write_file "
				"depth_completed=" << depth_completed << endl;
		}
	size0 = calc_size_on_file(depth_completed, 0 /*verbose_level*/);
	if (f_v) {
		cout << "size on file = " << size0 << endl;
		}
	
	if (size0 > 100 * ONE_MILLION) {
		cout << "generator::write_file" << endl;
		cout << "size on file = " << size0 << endl;
		cout << "the size is very big" << endl;
		}
	
	m = new memory_object;
	m->alloc(size0, 0);
	m->used_length = 0;

	write_memory_object(depth_completed, m,
			nb_group_elements, 0 /*verbose_level*/);

	INT size;
	size = m->used_length;

	
	fp.write((char *) &depth_completed, sizeof(INT));
	fp.write((char *) &size, sizeof(INT));
	fp.write(m->char_pointer, size);
	
	delete m;
	
	if (f_v) {
		cout << "generator::write_file done" << endl;
		}
}

void generator::read_file(ifstream &fp,
		INT &depth_completed, INT verbose_level)
{
	memory_object *m;
	INT f_v = (verbose_level >= 1);
	INT size;
	INT nb_group_elements;
	
	if (f_v) {
		cout << "generator::read_file" << endl;
		}


	fp.read((char *) &depth_completed, sizeof(INT));
	fp.read((char *) &size, sizeof(INT));

	if (f_v) {
		cout << "generator::read_file size = " << size << endl;
		}
	
	m = new memory_object;

	m->alloc(size, 0);
	m->used_length = 0;

	fp.read(m->char_pointer, size);
	

	m->used_length = size;
	m->cur_pointer = 0;

	if (f_v) {
		cout << "generator::read_file "
				"before generator_read_memory" << endl;
		}
	read_memory_object(depth_completed, m,
			nb_group_elements, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "generator::read_file "
				"after generator_read_memory" << endl;
		}

	delete m;
	
	if (f_v) {
		cout << "generator::read_file done, "
				"depth_completed=" << depth_completed << endl;
		}
}





