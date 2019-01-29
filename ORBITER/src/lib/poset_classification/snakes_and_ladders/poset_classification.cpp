// poset_classification.C
//
// Anton Betten
// December 29, 2003

#include "foundations/foundations.h"
#include "group_actions/group_actions.h"
#include "poset_classification/poset_classification.h"


namespace orbiter {

int poset_classification::nb_orbits_at_level(int level)
{
	int f, l;

	f = first_poset_orbit_node_at_level[level];
	l = first_poset_orbit_node_at_level[level + 1] - f;
	return l;
}

int poset_classification::nb_flag_orbits_up_at_level(int level)
{
	int f, l, i, F;

	f = first_poset_orbit_node_at_level[level];
	l = nb_orbits_at_level(level);
	F = 0;
	for (i = 0; i < l; i++) {
		F += root[f + i].nb_extensions;
		}
	return F;
}

poset_orbit_node *poset_classification::get_node_ij(
		int level, int node)
{
	int f;

	f = first_poset_orbit_node_at_level[level];
	return root + f + node;
}

int poset_classification::poset_structure_is_contained(
		int *set1, int sz1, int *set2, int sz2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_contained;
	int i, rk1, rk2;

	if (f_v) {
		cout << "poset_structure_is_contained" << endl;
		}
	if (f_vv) {
		cout << "set1: ";
		int_vec_print(cout, set1, sz1);
		cout << " ; ";
		cout << "set2: ";
		int_vec_print(cout, set2, sz2);
		cout << endl;
		}
	if (sz1 > sz2) {
		f_contained = FALSE;
		}
	else {
		if (Poset->f_subspace_lattice) {
			int *B1, *B2;
			int dim = Poset->VS->dimension;

			B1 = NEW_int(sz1 * dim);
			B2 = NEW_int((sz1 + sz2) * dim);

			for (i = 0; i < sz1; i++) {
				unrank_point(B1 + i * dim, set1[i]);
				}
			for (i = 0; i < sz2; i++) {
				unrank_point(B2 + i * dim, set2[i]);
				}

			rk1 = Poset->VS->F->Gauss_easy(B1, sz1, dim);
			if (rk1 != sz1) {
				cout << "poset_structure_is_contained "
						"rk1 != sz1" << endl;
				exit(1);
				}
			
			rk2 = Poset->VS->F->Gauss_easy(B2, sz2, dim);
			if (rk2 != sz2) {
				cout << "poset_structure_is_contained "
						"rk2 != sz2" << endl;
				exit(1);
				}
			int_vec_copy(B1,
					B2 + sz2 * dim,
					sz1 * dim);
			rk2 = Poset->VS->F->Gauss_easy(B2, sz1 + sz2, dim);
			if (rk2 > sz2) {
				f_contained = FALSE;
				}
			else {
				f_contained = TRUE;
				}

			FREE_int(B1);
			FREE_int(B2);
			}
		else {
			f_contained = int_vec_sort_and_test_if_contained(
					set1, sz1, set2, sz2);
			}
		}
	return f_contained;
}

void poset_classification::print_progress_by_extension(
		int size,
		int cur, int prev, int cur_ex,
		int nb_ext_cur, int nb_fuse_cur)
{
	double progress;
	
	
	progress = level_progress(size);
		
	print_level_info(size, prev);
	cout << " **** Upstep extension " << cur_ex << " / "
		<< root[prev].nb_extensions << " with "
		<< nb_ext_cur << " n e w orbits and "
		<< nb_fuse_cur << " fusion nodes. We now have " 
		<< cur - first_poset_orbit_node_at_level[size]
		<< " nodes at level " << size;
		cout << ", ";
	print_progress(progress);
	print_progress_by_level(size);
}

void poset_classification::print_progress(
		int size,
		int cur, int prev,
		int nb_ext_cur, int nb_fuse_cur)
{
	double progress;
	
	
	progress = level_progress(size);
		
	print_level_info(size, prev);
	cout << " **** Upstep finished with " 
		<< nb_ext_cur << " n e w orbits and "
		<< nb_fuse_cur << " fusion nodes. We now have " 
		<< cur - first_poset_orbit_node_at_level[size]
		<< " nodes at level " << size;
		cout << ", ";
	print_progress(progress);
	print_progress_by_level(size);
}

void poset_classification::print_progress(double progress)
{
	double progress0;
	int progress1, progress2;
		
	progress0 = progress * 100.;
	progress2 = (int) (progress0 * 100.);
	progress1 = progress2 / 100;
	progress2 = progress2 % 100;
	cout << "progress: " << progress1 << "."
			<< setw(2) << progress2 << " % " << endl;
}

void poset_classification::print_progress_by_level(int lvl)
{
	int i;
	
	for (i = 0; i < lvl; i++) {
		//remaining = nb_extension_nodes_at_level_total[i] 
		//	- nb_extension_nodes_at_level[i] - nb_fusion_nodes_at_level[i];
		cout << setw(5) << i << " : " << setw(10)
			<< nb_extension_nodes_at_level[i] << " : "
			<< setw(10) << nb_fusion_nodes_at_level[i] << " : " 
			<< setw(10) << nb_extension_nodes_at_level_total[i] << " : " 
			<< setw(10) << nb_unprocessed_nodes_at_level[i];
		cout << endl;
		}
	//print_statistic_on_callbacks();
}

void poset_classification::print_orbit_numbers(int depth)
{
	int nb_nodes, j;
	
	nb_nodes = nb_orbits_at_level(depth);
	cout << "###########################################################"
			"#######################################" << endl;
	print_problem_label();
	cout << "Found " << nb_nodes << " orbits at depth " << depth << endl;
	for (j = 0; j <= depth; j++) {
		cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
		}
	cout << "total: " << first_poset_orbit_node_at_level[depth + 1] << endl;
	//gen->print_statistic_on_callbacks();
	compute_and_print_automorphism_group_orders(depth, cout);
}

void poset_classification::print()
{
	int nb_nodes, j;

	cout << "Poset classification:" << endl;
	print_problem_label();
	cout << "Action:" << endl;
	Poset->A->print_info();
	cout << "Action2:" << endl;
	Poset->A2->print_info();
	cout << "Group order:" << Poset->go << endl;
	cout << "Degree:" << Poset->A2->degree << endl;
	cout << "depth:" << depth << endl;
	nb_nodes = nb_orbits_at_level(depth);
	cout << "Found " << nb_nodes << " orbits at depth " << depth << endl;
	for (j = 0; j <= depth; j++) {
		cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
		}
	cout << "total: " << first_poset_orbit_node_at_level[depth + 1] << endl;
}

void poset_classification::print_statistic_on_callbacks_naked()
{
	cout << Poset->A->nb_times_image_of_called - nb_times_image_of_called0 << "/";
	cout << Poset->A->nb_times_mult_called - nb_times_mult_called0 << "/";
	cout << Poset->A->nb_times_invert_called - nb_times_invert_called0 << "/";
	cout << Poset->A->nb_times_retrieve_called - nb_times_retrieve_called0 << "/";
	cout << Poset->A->nb_times_store_called - nb_times_store_called0;
}

void poset_classification::print_statistic_on_callbacks()
{
	cout << "# of calls to image_of/mult/invert/retrieve/store: ";
	print_statistic_on_callbacks_naked();
	cout << endl;
}

orbit_transversal *poset_classification::get_orbit_transversal(
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	orbit_transversal *T;
	int orbit_at_level;

	if (f_v) {
		cout << "poset_classification::get_orbit_transversal" << endl;
		}
	T = NEW_OBJECT(orbit_transversal);
	T->A = Poset->A;
	T->A2 = Poset->A2;


	T->nb_orbits = nb_orbits_at_level(level);


	if (f_v) {
		cout << "poset_classification::get_orbit_transversal "
				"processing " << T->nb_orbits
				<< " orbit representatives" << endl;
		}


	T->Reps = NEW_OBJECTS(set_and_stabilizer, T->nb_orbits);

	for (orbit_at_level = 0;
			orbit_at_level < T->nb_orbits;
			orbit_at_level++) {

		set_and_stabilizer *SaS;

		SaS = get_set_and_stabilizer(level,
				orbit_at_level, verbose_level);



		T->Reps[orbit_at_level].init_everything(
				Poset->A, Poset->A2, SaS->data, level,
				SaS->Strong_gens, 0 /* verbose_level */);

		SaS->data = NULL;
		SaS->Strong_gens = NULL;

		FREE_OBJECT(SaS);

		}



	if (f_v) {
		cout << "poset_classification::get_orbit_transversal done" << endl;
		}
	return T;
}

set_and_stabilizer *poset_classification::get_set_and_stabilizer(
		int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_and_stabilizer *SaS;

	if (f_v) {
		cout << "poset_classification::get_set_and_stabilizer" << endl;
		}
	SaS = NEW_OBJECT(set_and_stabilizer);
	SaS->init(Poset->A, Poset->A2, 0 /*verbose_level */);
	SaS->allocate_data(level, 0 /* verbose_level */);
	get_set_by_level(level, orbit_at_level, SaS->data);
	get_stabilizer_generators(SaS->Strong_gens,
		level, orbit_at_level, 0 /* verbose_level */);
	SaS->Strong_gens->group_order(SaS->target_go);
	SaS->Stab = SaS->Strong_gens->create_sims(0 /*verbose_level*/);
	if (f_v) {
		cout << "poset_classification::get_set_and_stabilizer done" << endl;
		}
	return SaS;
}

void poset_classification::get_set_by_level(
		int level, int node, int *set)
{
	int size;
	poset_orbit_node *O;
	
	O = get_node_ij(level, node);
	size = O->depth_of_node(this);
	if (size != level) {
		cout << "poset_classification::get_set_by_level "
				"size != level" << endl;
		exit(1);
		}
	//root[n].store_set_to(this, size - 1, set);
	O->store_set_to(this, size - 1, set);
}

void poset_classification::get_set(
		int node, int *set, int &size)
{
	size = root[node].depth_of_node(this);
	root[node].store_set_to(this, size - 1, set);
}

void poset_classification::get_set(
		int level, int orbit, int *set, int &size)
{
	int node;

	node = first_poset_orbit_node_at_level[level] + orbit;
	size = root[node].depth_of_node(this);
	root[node].store_set_to(this, size - 1, set);
}

void poset_classification::print_set_verbose(int node)
{
	root[node].print_set_verbose(this);
}

void poset_classification::print_set_verbose(
		int level, int orbit)
{
	int node;

	node = first_poset_orbit_node_at_level[level] + orbit;
	root[node].print_set_verbose(this);
}

void poset_classification::print_set(int node)
{
	root[node].print_set(this);
}

void poset_classification::print_set(int level, int orbit)
{
	int node;

	node = first_poset_orbit_node_at_level[level] + orbit;
	root[node].print_set(this);
}

int poset_classification::find_poset_orbit_node_for_set(
		int len,
		int *set, int f_tolerant, int verbose_level)
// finds the node that represents s_0,...,s_{len - 1}
{
	int f_v = (verbose_level >= 1);
	int ret;
	
	if (f_v) {
		cout << "poset_classification::find_poset_orbit_node_for_set ";
		int_vec_print(cout, set, len);
		cout << endl;
		}
	if (f_starter) {
		int i, j, h;
		if (len < starter_size) {
			cout << "poset_classification::find_poset_orbit_node_for_set "
					"len < starter_size" << endl;
			cout << "len=" << len << endl;
			exit(1);
			}
		for (i = 0; i < starter_size; i++) {
			for (j = i; j < len; j++) {
				if (set[j] == starter[i]) {
					if (f_v) {
						cout << "found " << i << "-th element "
								"of the starter which is " << starter[i]
							<< " at position " << j << endl;
						}
					break;
					}
				}
			if (j == len) {
				cout << "poset_classification::find_poset_orbit_node_for_set "
						"did not find " << i << "-th element "
						"of the starter" << endl;
				}
			for (h = j; h > i; h--) {
				set[h] = set[h - 1];
				}
			set[i] = starter[i];
			}
		int from = starter_size;
		int node = starter_size;
		ret = find_poset_orbit_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
		}
	else {
		int from = 0;
		int node = 0;
		ret = find_poset_orbit_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
		}
	if (ret == -1) {
		if (f_tolerant) {
			if (f_v) {
				cout << "poset_classification::find_poset_orbit_node_for_set ";
				int_vec_print(cout, set, len);
				cout << " extension not found, "
						"we are tolerant, returnning -1" << endl;
				}
			return -1;
			}
		else {
			cout << "poset_classification::find_poset_orbit_node_for_set "
					"we should not be here" << endl;
			exit(1);
			}
		}
	return ret;
	
}

int poset_classification::find_poset_orbit_node_for_set_basic(
		int from,
		int node, int len, int *set, int f_tolerant,
		int verbose_level)
{
	int i, j, pt;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 1);

	if (f_vv) {
		cout << "poset_classification::"
				"find_poset_orbit_node_for_set_basic "
				"looking for set ";
		int_vec_print(cout, set, len);
		cout << endl;
		cout << "node=" << node << endl;
		cout << "from=" << from << endl;
		cout << "len=" << len << endl;
		cout << "f_tolerant=" << f_tolerant << endl;
		}
	for (i = from; i < len; i++) {
		pt = set[i];
		if (f_vv) {
			cout << "pt=" << pt << endl;
			cout << "calling root[node].find_extension_from_point" << endl;
			}
		j = root[node].find_extension_from_point(this, pt, FALSE);
		if (j == -1) {
			if (f_v) {
				cout << "poset_classification::"
						"find_poset_orbit_node_for_set_basic "
						"depth " << i << " no extension for point "
						<< pt << " found" << endl;
				}
			if (f_tolerant) {
				if (f_v) {
					cout << "poset_classification::"
							"find_poset_orbit_node_for_set_basic "
							"since we are tolerant, we return -1" << endl;
					}
				return -1;
				}
			else {
				cout << "poset_classification::"
						"find_poset_orbit_node_for_set_basic "
						"failure in find_extension_from_point" << endl;
				int_vec_print(cout, set, len);
				cout << endl;
				cout << "node=" << node << endl;
				cout << "from=" << from << endl;
				cout << "i=" << i << endl;
				cout << "pt=" << pt << endl;
				root[node].print_extensions(this);
				exit(1);
				}
			}
		if (root[node].E[j].pt != pt) {
			cout << "poset_classification::"
					"find_poset_orbit_node_for_set_basic "
					"root[node].E[j].pt != pt" << endl;
			exit(1);
			}
		if (root[node].E[j].type != EXTENSION_TYPE_EXTENSION && 
			root[node].E[j].type != EXTENSION_TYPE_PROCESSING) {
			cout << "poset_classification::"
					"find_poset_orbit_node_for_set_basic "
					"root[node].E[j].type != "
					"EXTENSION_TYPE_EXTENSION" << endl;
			cout << "root[node].E[j].type="
					<< root[node].E[j].type << " = ";
			print_extension_type(cout, root[node].E[j].type);
			cout << endl;
			cout << "poset_classification::"
					"find_poset_orbit_node_for_set_basic "
					"looking for set ";
			int_vec_print(cout, set, len);
			cout << endl;
			cout << "node=" << node << endl;
			cout << "from=" << from << endl;
			cout << "i=" << i << endl;
			cout << "node=" << node << endl;
			cout << "f_tolerant=" << f_tolerant << endl;
			cout << "node=" << node << endl;
			cout << "pt=" << pt << endl;
			cout << "j=" << j << endl;
			exit(1);
			}
		node = root[node].E[j].data;
		if (f_v) {
			cout << "depth " << i << " extension " << j
					<< " n e w node " << node << endl;
			}
		}
	return node;
}

void poset_classification::poset_orbit_node_depth_breadth_perm_and_inverse(
	int max_depth,
	int *&perm, int *&perm_inv, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx = 0;
	int N;

	if (f_v) {
		cout << "poset_classification::poset_orbit_node_"
				"depth_breadth_perm_and_inverse" << endl;
		cout << "max_depth = " << max_depth << endl;
		}

	N = first_poset_orbit_node_at_level[max_depth + 1];
	if (f_v) {
		cout << "N = first_poset_orbit_node_at_level[max_depth + 1] = "
				<< N << endl;
		}
	
	perm = NEW_int(N);
	perm_inv = NEW_int(N);
	
	if (f_v) {
		cout << "calling root->poset_orbit_node_"
				"depth_breadth_perm_and_inverse" << endl;
		}
	root->poset_orbit_node_depth_breadth_perm_and_inverse(
			this,
			max_depth, idx, 0, 0, perm, perm_inv);
	
}

int poset_classification::count_extension_nodes_at_level(int lvl)
{
	int prev;

	nb_extension_nodes_at_level_total[lvl] = 0;
	for (prev = first_poset_orbit_node_at_level[lvl];
			prev < first_poset_orbit_node_at_level[lvl + 1];
			prev++) {
			
		nb_extension_nodes_at_level_total[lvl] +=
				root[prev].nb_extensions;
		
		}
	nb_unprocessed_nodes_at_level[lvl] =
			nb_extension_nodes_at_level_total[lvl];
	nb_fusion_nodes_at_level[lvl] = 0;
	nb_extension_nodes_at_level[lvl] = 0;
	return nb_extension_nodes_at_level_total[lvl];
}

double poset_classification::level_progress(int lvl)
{
	return
		((double)(nb_fusion_nodes_at_level[lvl] +
				nb_extension_nodes_at_level[lvl])) /
			(double) nb_extension_nodes_at_level_total[lvl];
}



void poset_classification::count_automorphism_group_orders(
	int lvl, int &nb_agos,
	longinteger_object *&agos, int *&multiplicities,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, l, j, c, h, f_added;
	longinteger_object ago;
	longinteger_object *tmp_agos;
	int *tmp_multiplicities;
	longinteger_domain D;
	
	l = nb_orbits_at_level(lvl);
	if (f_v) {
		cout << "collecting the automorphism group orders of "
				<< l << " orbits" << endl;
		}
	nb_agos = 0;
	agos = NULL;
	multiplicities = NULL;
	for (i = 0; i < l; i++) {
		get_stabilizer_order(lvl, i, ago);
		f_added = FALSE;
		for (j = 0; j < nb_agos; j++) {
			c = D.compare_unsigned(ago, agos[j]);
			//cout << "comparing " << ago << " with " << agos[j]
			// << " yields " << c << endl;
			if (c >= 0) {
				if (c == 0) {
					multiplicities[j]++;
					}
				else {
					tmp_agos = agos;
					tmp_multiplicities = multiplicities;
					agos = NEW_OBJECTS(longinteger_object, nb_agos + 1);
					multiplicities = NEW_int(nb_agos + 1);
					for (h = 0; h < j; h++) {
						tmp_agos[h].swap_with(agos[h]);
						multiplicities[h] = tmp_multiplicities[h];
						}
					ago.swap_with(agos[j]);
					multiplicities[j] = 1;
					for (h = j; h < nb_agos; h++) {
						tmp_agos[h].swap_with(agos[h + 1]);
						multiplicities[h + 1] = tmp_multiplicities[h];
						}
					nb_agos++;
					if (tmp_agos) {
						FREE_OBJECTS(tmp_agos);
						FREE_int(tmp_multiplicities);
						}
					}
				f_added = TRUE;
				break;
				}
			}
		if (!f_added) {
			// add at the end (including the case that the list is empty)
			tmp_agos = agos;
			tmp_multiplicities = multiplicities;
			agos = NEW_OBJECTS(longinteger_object, nb_agos + 1);
			multiplicities = NEW_int(nb_agos + 1);
			for (h = 0; h < nb_agos; h++) {
				tmp_agos[h].swap_with(agos[h]);
				multiplicities[h] = tmp_multiplicities[h];
				}
			ago.swap_with(agos[nb_agos]);
			multiplicities[nb_agos] = 1;
			nb_agos++;
			if (tmp_agos) {
				FREE_OBJECTS(tmp_agos);
				FREE_int(tmp_multiplicities);
				}
			}
		}
}

void poset_classification::compute_and_print_automorphism_group_orders(
		int lvl, ostream &ost)
{

	int j, nb_agos;
	longinteger_object *agos;
	int *multiplicities;
	int N, r, h;
	longinteger_object S, S1, Q;
	longinteger_domain D;
	
	count_automorphism_group_orders(lvl, nb_agos, agos,
			multiplicities, FALSE);
	S.create(0);
	N = 0;
	for (j = 0; j < nb_agos; j++) {
		N += multiplicities[j];
		for (h = 0; h < multiplicities[j]; h++) {
			D.add(S, agos[j], S1);
			S1.assign_to(S);
			}
		}
	D.integral_division_by_int(S, N, Q, r);
	

	ost << "(";
	for (j = 0; j < nb_agos; j++) {
		ost << agos[j];
		if (multiplicities[j] == 1) {
			}
		else if (multiplicities[j] >= 10) {
			ost << "^{" << multiplicities[j] << "}";
			}
		else  {
			ost << "^" << multiplicities[j];
			}
		if (j < nb_agos - 1) {
			ost << ", ";
			}
		}
	ost << ") average is " << Q << " + " << r << " / " << N << endl;
	if (nb_agos) {
		FREE_OBJECTS(agos);
		FREE_int(multiplicities);
		}

}

void poset_classification::stabilizer_order(int node, longinteger_object &go)
{
	if (root[node].nb_strong_generators) {
		go.create_product(Poset->A->base_len, root[node].tl);
		}
	else {
		go.create(1);
		}
}

#if 0
int poset_classification::check_the_set(int len, int *S, int verbose_level)
// used by lookahead_first_at_level
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "poset_classification::check_the_set" << endl;
		}
	if (!f_candidate_check_func) {
		return TRUE;
		}
	
	if (f_v) {
		cout << "poset_classification::check_the_set checking set: ";
		int_set_print(cout, S, len);
		cout << endl;
		}
	if (!(*candidate_check_func)(len, S,
			candidate_check_data, verbose_level - 1)) {
		if (f_v) {
			cout << "the set is not accepted" << endl;
			}
		return FALSE;
		}
	if (f_v) {
		cout << "the set is accepted" << endl;
		}
	return TRUE;
}

int poset_classification::check_the_set_incrementally(int len,
		int *S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (!f_candidate_incremental_check_func) {
		return check_the_set(len, S, verbose_level);
		}
	
	if (f_v) {
		cout << "poset_classification::check_the_set_incrementally checking set: ";
		int_set_print(cout, S, len);
		cout << endl;
		}
	if (!(*candidate_incremental_check_func)(len, S,
			candidate_incremental_check_data, verbose_level - 1)) {
		if (f_v) {
			cout << "the set is not accepted" << endl;
			}
		return FALSE;
		}
	if (f_v) {
		cout << "the set is accepted" << endl;
		}
	return TRUE;
}
#endif


void poset_classification::orbit_length(int orbit_at_level,
		int level, longinteger_object &len)
// uses poset_classification::go for the group order
{
	longinteger_domain D;
	longinteger_object stab_order, quo, rem;

	get_stabilizer_order(level, orbit_at_level, stab_order);
	D.integral_division(Poset->go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "poset_classification::orbit_length stabilizer order does "
				"not divide group order" << endl;
		exit(1);
		}
}

void poset_classification::get_orbit_length_and_stabilizer_order(
		int node,
		int level, longinteger_object &stab_order,
		longinteger_object &len)
// uses poset_classification::go for the group order
{
	longinteger_domain D;
	longinteger_object quo, rem;

	get_stabilizer_order(level, node, stab_order);
	D.integral_division(Poset->go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "poset_classification::orbit_length "
				"stabilizer order "
				"does not divide group order" << endl;
		exit(1);
		}
}

int poset_classification::orbit_length_as_int(
		int orbit_at_level, int level)
{
	longinteger_object len;

	orbit_length(orbit_at_level, level, len);
	return len.as_int();
	
}


void poset_classification::print_representatives_at_level(
		int lvl)
{
	int i, f, l;
	
	f = first_poset_orbit_node_at_level[lvl];
	l = nb_orbits_at_level(lvl);
	cout << "The " << l << " representatives at level "
			<< lvl << " are:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " / " << l << " : ";
		root[f + i].print_set(this);
		cout << endl;
		}
}

void poset_classification::print_lex_rank(int *set, int sz)
{
	int r1, r2;
	int n;
	
	n = Poset->A2->degree;
	r1 = rank_subset(set, sz, n);
	r2 = rank_k_subset(set, n, sz);

	cout << "lex rank = " << r1 << " lex rank as "
			<< sz << "-subset = " << r2;
}


void poset_classification::print_problem_label()
{
	if (problem_label[0]) {
		cout << problem_label << " ";
		}
}

void poset_classification::print_level_info(int prev_level, int prev)
{
	int t1, dt;
	
	t1 = os_ticks();
	//cout << "poset_classification::print_level_info t0=" << t0 << endl;
	//cout << "poset_classification::print_level_info t1=" << t1 << endl;
	dt = t1 - t0;
	//cout << "poset_classification::print_level_info dt=" << dt << endl;
	
	cout << "Time ";
	time_check_delta(cout, dt);
	print_problem_label();
	cout << " : Level " << prev_level << " Node " << prev << " = "
		<< prev - first_poset_orbit_node_at_level[prev_level]
		<< " / " 
		<< nb_orbits_at_level(prev_level)
		<< " : ";
}

void poset_classification::print_level_extension_info(
	int prev_level,
	int prev, int cur_extension)
{
#if 0
	int t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	
	cout << "Time ";
	time_check_delta(cout, dt);
	print_problem_label();
#endif
	cout << "Level " << prev_level << " Node " << prev << " = "
		<< prev - first_poset_orbit_node_at_level[prev_level]
		<< " / " 
		<< nb_orbits_at_level(prev_level)
		<< " Extension " << cur_extension 
		<< " / " 
		<< root[prev].nb_extensions 
		<< " : ";
}

void poset_classification::print_level_extension_coset_info(
	int prev_level,
	int prev, int cur_extension, int coset, int nb_cosets)
{
#if 0
	int t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	
	cout << "Time ";
	time_check_delta(cout, dt);
	print_problem_label();
#endif
	cout << "Level " << prev_level << " Node " << prev << " = "
		<< prev - first_poset_orbit_node_at_level[prev_level]
		<< " / " 
		<< nb_orbits_at_level(prev_level)
		<< " Extension " << cur_extension 
		<< " / " 
		<< root[prev].nb_extensions 
		<< " : " 
		<< "Coset " << coset << " / " << nb_cosets << " : ";
}

void poset_classification::recreate_schreier_vectors_up_to_level(
		int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "poset_classification::recreate_"
				"schreier_vectors_up_to_level "
				"creating Schreier vectors up to "
				"level " << lvl << endl;
		}
	for (i = 0; i <= lvl; i++) {
		if (f_v) {
			cout << "poset_classification::recreate_"
					"schreier_vectors_up_to_level "
					"creating Schreier vectors at "
					"level " << i << endl;
			}
		recreate_schreier_vectors_at_level(i, 0 /*verbose_level*/);
		}
}

void poset_classification::recreate_schreier_vectors_at_level(
		int i,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;//(verbose_level >= 2);
	int f, cur, l, prev, u;
	int f_recreate_extensions = FALSE;
	int f_dont_keep_sv = FALSE;

	f = first_poset_orbit_node_at_level[i];
	cur = first_poset_orbit_node_at_level[i + 1];
	l = cur - f;

	if (f_v) {
		cout << "creating Schreier vectors at depth " << i
				<< " for " << l << " orbits" << endl;
		}
	if (f_v) {
		cout << "poset_classification::recreate_"
				"schreier_vectors_at_level "
				"Testing if a schreier vector file exists" << endl;
		}
	if (test_sv_level_file_binary(i, fname_base)) {

		if (f_v) {
			cout << "poset_classification::recreate_"
					"schreier_vectors_at_level "
					"Yes, a schreier vector file exists. "
					"We will read this file" << endl;
			}

		read_sv_level_file_binary(i, fname_base, FALSE, 0, 0, 
			f_recreate_extensions, f_dont_keep_sv, 
			verbose_level);
		if (f_v) {
			cout << "read Schreier vectors at depth " << i
					<< " from file" << endl;
			}
		return;
		}


	if (f_v) {
		cout << "poset_classification::recreate_"
				"schreier_vectors_at_level "
				"No, a schreier vector file does not exists. "
				"We will create such a file now" << endl;
		}



	for (u = 0; u < l; u++) {
			
		prev = f + u;
			
		if (f_v && !f_vv) {
			cout << ".";
			if (((u + 1) % 50) == 0) {
				cout << "; " << u + 1 << " / " << l << endl;
				}
			if (((u + 1) % 1000) == 0)
				cout << " " << u + 1 << endl;
			}
		else if (f_vv) {
			cout << "poset_classification::recreate_"
					"schreier_vectors_at_level "
				<< i << " node " << u << " / " << l << endl;
			}
			
		root[prev].compute_schreier_vector(this, i,
				0 /*verbose_level - 1*/);
		}
	write_sv_level_file_binary(i,
			fname_base, FALSE, 0, 0, verbose_level);
	if (f_v) {
		cout << "poset_classification::recreate_"
				"schreier_vectors_at_level "
				"Written a file with Schreier "
				"vectors at depth " << i << endl;
		}
	if (f_v) {
		cout << endl;
		}
		
}

void poset_classification::print_node(int node)
{
	cout << "poset_classification::print_node "
			"node " << node << ":" << endl;
	root[node].print_node(this);
}

void poset_classification::print_tree()
{
	int i;
	
	cout << "poset_classification::print_tree "
			"nb_poset_orbit_nodes_used="
			<< nb_poset_orbit_nodes_used << endl;
	for (i = 0; i < nb_poset_orbit_nodes_used; i++) {
		print_node(i);
		}
}

void poset_classification::get_table_of_nodes(int *&Table,
		int &nb_rows, int &nb_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "poset_classification::get_table_of_nodes "
				"nb_poset_orbit_nodes_used="
				<< nb_poset_orbit_nodes_used << endl;
		}
	nb_rows = nb_poset_orbit_nodes_used;
	nb_cols = 6;
	Table = NEW_int(nb_poset_orbit_nodes_used * nb_cols);
		
	for (i = 0; i < nb_poset_orbit_nodes_used; i++) {

		if (f_v) {
			cout << "poset_classification::get_table_of_nodes "
					"node " << i
					<< " / " << nb_poset_orbit_nodes_used << endl;
			}

		Table[i * nb_cols + 0] = root[i].get_level(this);
		Table[i * nb_cols + 1] = root[i].get_node_in_level(this);
		Table[i * nb_cols + 2] = root[i].pt;

		longinteger_object go;
			
		root[i].get_stabilizer_order(this, go);
		Table[i * nb_cols + 3] = go.as_int();
		Table[i * nb_cols + 4] =
				root[i].get_nb_of_live_points();
		Table[i * nb_cols + 5] =
				root[i].get_nb_of_orbits_under_stabilizer();
		}
	if (f_v) {
		cout << "poset_classification::get_table_of_nodes "
				"done" << endl;
		}
}

int poset_classification::count_live_points(
		int level,
		int node_local, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_v2 = FALSE;
	int node;
	//int *osv;
	int nb_points;
	//int *pt_list;
	//int *pt_list2;
	//int a, i, j;
	
	if (f_v) {
		cout << "poset_classification::count_live_points" << endl;
		}
	node = first_poset_orbit_node_at_level[level] + node_local;
	if (root[node].Schreier_vector == NULL) {
		root[node].compute_schreier_vector(this, 
			level, verbose_level - 2);
		}
#if 0
	osv = root[node].sv;
	nb_points = osv[0];
	pt_list = osv + 1;
	if (f_v2) {
		cout << "pt_list of size " << nb_points << " : ";
		int_vec_print(cout, pt_list, nb_points);
		cout << endl;
		}
#else
	nb_points = root[node].get_nb_of_live_points();
#endif


#if 0
	pt_list2 = NEW_int(nb_points);
	a = root[node].pt;
	j = 0;
	for (i = 0; i < nb_points; i++) {
		if (pt_list[i] > a) {
			pt_list2[j++] = pt_list[i]; // why is this needed?
			}
		}
	nb_points = j;
	FREE_int(pt_list2);
#endif

	return nb_points;
}

void poset_classification::find_automorphism_group_of_order(
		int level, int order)
{
	int nb_nodes, node, i, j, elt_order;
	longinteger_object ago;
	int set[300];
	
	nb_nodes = nb_orbits_at_level(level);
	for (i = 0; i < nb_nodes; i++) {
		node = first_poset_orbit_node_at_level[level] + i;
		if (root[node].nb_strong_generators == 0) {
			ago.create(1);
			}
		else {
			ago.create_product(Poset->A->base_len, root[node].tl);
			}
		if (ago.as_int() == order) {
			cout << "found a node whose automorphism group is order "
					<< order << endl;
			cout << "the node is # " << i << " at level "
					<< level << endl;
			get_set(first_poset_orbit_node_at_level[level] + i,
					set, level);
			int_vec_print(cout, set, level);
			cout << endl;
			
			strong_generators *Strong_gens;
			
			get_stabilizer_generators(Strong_gens,
				level, i, 0  /* verbose_level */);
				
			for (j = 0; j < Strong_gens->gens->len; j++) {
				elt_order =
						Poset->A->element_order(Strong_gens->gens->ith(j));
				cout << "poset_classification " << j << " of order "
						<< elt_order << ":" << endl;
				if (order == elt_order) {
					cout << "CYCLIC" << endl;
					}
				Poset->A->element_print(
						Strong_gens->gens->ith(j), cout);
				Poset->A->element_print_as_permutation(
						Strong_gens->gens->ith(j), cout);
				}
			FREE_OBJECT(Strong_gens);
			}
		}
}

void poset_classification::get_stabilizer_order(int level,
		int orbit_at_level, longinteger_object &go)
{
	poset_orbit_node *O;
	int nd;

	nd = first_poset_orbit_node_at_level[level] + orbit_at_level;
	O = root + nd;

	if (O->nb_strong_generators == 0) {
		go.create(1);
		}
	else {
		longinteger_domain D;

		D.multiply_up(go, O->tl, Poset->A->base_len);
		}
}

void poset_classification::get_stabilizer_group(
	group *&G,
	int level, int orbit_at_level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	poset_orbit_node *O;
	int node;

	if (f_v) {
		cout << "poset_classification::"
				"get_stabilizer_group level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
		}
	G = NEW_OBJECT(group);
	node = first_poset_orbit_node_at_level[level] + orbit_at_level;
	O = root + node;

	G->init(Poset->A);
	if (f_vv) {
		cout << "poset_classification::"
				"get_stabilizer_group before "
				"G->init_strong_generators_by_hdl" << endl;
		}
	G->init_strong_generators_by_hdl(O->nb_strong_generators,
			O->hdl_strong_generators, O->tl, FALSE);
	G->schreier_sims(0);
	
	if (f_v) {
		cout << "poset_classification::"
				"get_stabilizer_group level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
		}
}

void poset_classification::get_stabilizer_generators(
	strong_generators *&gens,
	int level, int orbit_at_level, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "poset_classification::"
				"get_stabilizer_generators level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
		}
	group *G;

	get_stabilizer_group(G,
			level, orbit_at_level, verbose_level - 1);

	gens = NEW_OBJECT(strong_generators);

	gens->init_from_sims(G->S, 0 /* verbose_level */);
	FREE_OBJECT(G);
	if (f_v) {
		cout << "poset_classification::"
				"get_stabilizer_generators level=" << level
				<< " orbit_at_level=" << orbit_at_level
				<< " done" << endl;
		}

}

void poset_classification::change_extension_type(int level,
		int node, int cur_ext, int type, int verbose_level)
{
	if (type == EXTENSION_TYPE_EXTENSION) {
		// extension node
		if (root[node].E[cur_ext].type != EXTENSION_TYPE_UNPROCESSED && 
			root[node].E[cur_ext].type != EXTENSION_TYPE_PROCESSING) {
			cout << "poset_classification::change_extension_type trying to install "
					"extension node, fatal: root[node].E[cur_ext].type != "
					"EXTENSION_TYPE_UNPROCESSED && root[node].E[cur_ext].type "
					"!= EXTENSION_TYPE_PROCESSING" << endl;
			cout << "root[node].ext[cur_ext].type="
					<< root[node].E[cur_ext].type << endl;
			exit(1);
			}
		nb_extension_nodes_at_level[level]++;
		nb_unprocessed_nodes_at_level[level]--;
		root[node].E[cur_ext].type = EXTENSION_TYPE_EXTENSION;
		}
	else if (type == EXTENSION_TYPE_FUSION) {
		// fusion
		if (root[node].E[cur_ext].type != EXTENSION_TYPE_UNPROCESSED) {
			cout << "poset_classification::change_extension_type trying to install "
					"fusion node, fatal: root[node].E[cur_ext].type != "
					"EXTENSION_TYPE_UNPROCESSED" << endl;
			cout << "root[node].ext[cur_ext].type="
					<< root[node].E[cur_ext].type << endl;
			exit(1);
			}
		nb_fusion_nodes_at_level[level]++;
		nb_unprocessed_nodes_at_level[level]--;
		root[node].E[cur_ext].type = EXTENSION_TYPE_FUSION;
		}
}

void poset_classification::orbit_element_unrank(
		int depth,
		int orbit_idx, int rank, int *set,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt1;
	int *Elt2;
	int *the_set;
	poset_orbit_node *O;


	if (f_v) {
		cout << "poset_classification::orbit_element_unrank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx
				<< " rank=" << rank << endl;
		}

	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	Elt2 = NEW_int(Poset->A->elt_size_in_int);
	the_set = NEW_int(depth);
	
	O = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];
	coset_unrank(depth, orbit_idx, rank, Elt1,
			0/*verbose_level*/);

	Poset->A->element_invert(Elt1, Elt2, 0);
	O->store_set_to(this, depth - 1, the_set);
	Poset->A2->map_a_set(the_set, set, depth, Elt2,
			0/*verbose_level*/);

	FREE_int(the_set);
	FREE_int(Elt1);
	FREE_int(Elt2);
	if (f_v) {
		cout << "poset_classification::orbit_element_unrank ";
		int_vec_print(cout, set, depth);
		cout << endl;
		}
}

void poset_classification::orbit_element_rank(
	int depth,
	int &orbit_idx, int &rank, int *set,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *Elt1;
	int *the_set;
	int *canonical_set;
	int i;


	if (f_v) {
		cout << "poset_classification::orbit_element_rank "
				"depth=" << depth << " ";
		int_vec_print(cout, set, depth);
		cout << endl;
		}

	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	the_set = NEW_int(depth);
	canonical_set = NEW_int(depth);
	for (i = 0; i < depth; i++) {
		the_set[i] = set[i];
		}
	
	orbit_idx = trace_set(the_set, depth, depth, 
		canonical_set, Elt1, 
		//f_implicit_fusion, 
		verbose_level - 3);

	// now Elt1 is the transporter element that moves 
	// the given set to the orbit representative

	if (f_vv) {
		cout << "poset_classification::orbit_element_rank "
				"after trace_set, "
				"orbit_idx = " << orbit_idx << endl;
		cout << "transporter:" << endl;
		Poset->A->element_print_quick(Elt1, cout);
		cout << "as permutation:" << endl;
		Poset->A2->element_print_as_permutation(Elt1, cout);
		}
	if (f_v) {
		cout << "calling coset_rank" << endl;
		}
	rank = coset_rank(depth, orbit_idx, Elt1, verbose_level);
	if (f_v) {
		cout << "after coset_rank, rank=" << rank << endl;
		}
		
	FREE_int(Elt1);
	FREE_int(the_set);
	FREE_int(canonical_set);
	if (f_v) {
		cout << "poset_classification::orbit_element_rank "
				"orbit_idx="
				<< orbit_idx << " rank=" << rank << endl;
		}
}

void poset_classification::coset_unrank(
		int depth, int orbit_idx,
		int rank, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *the_set;
	group *G1, *G2;
	int *Elt_gk;
	longinteger_object G_order, U_order;
	poset_orbit_node *O1, *O2;

	if (f_v) {
		cout << "poset_classification::coset_unrank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		Poset->A->print_info();
		cout << "action A2:" << endl;
		Poset->A2->print_info();
		}

	O1 = &root[0];
	O2 = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];


	
	G1 = NEW_OBJECT(group);
	G2 = NEW_OBJECT(group);
	the_set = NEW_int(depth);
	Elt_gk = NEW_int(Poset->A->elt_size_in_int);
	
	O2->store_set_to(this, depth - 1, the_set);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		int_vec_print(cout, the_set, depth);
		cout << endl;
		}
	
	O1->get_stabilizer(this,
			*G1, G_order, verbose_level - 2);
	O2->get_stabilizer(this,
			*G2, U_order, verbose_level - 2);


	Poset->A->coset_unrank(G1->S, G2->S, rank, Elt, verbose_level);

	FREE_OBJECT(G1);
	FREE_OBJECT(G2);
	FREE_int(the_set);
	FREE_int(Elt_gk);

}

int poset_classification::coset_rank(
		int depth, int orbit_idx,
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rank;
	int *the_set;
	group *G1, *G2;
	int *Elt_gk;
	longinteger_object G_order, U_order;
	poset_orbit_node *O1, *O2;

	if (f_v) {
		cout << "poset_classification::coset_rank "
				"depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		Poset->A->print_info();
		cout << "action A2:" << endl;
		Poset->A2->print_info();
		}

	O1 = &root[0];
	O2 = &root[first_poset_orbit_node_at_level[depth] + orbit_idx];


	
	G1 = NEW_OBJECT(group);
	G2 = NEW_OBJECT(group);
	the_set = NEW_int(depth);
	Elt_gk = NEW_int(Poset->A->elt_size_in_int);
	
	O2->store_set_to(this, depth - 1, the_set);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		int_vec_print(cout, the_set, depth);
		cout << endl;
		}
	
	O1->get_stabilizer(this, *G1, G_order, verbose_level - 2);
	O2->get_stabilizer(this, *G2, U_order, verbose_level - 2);


	rank = Poset->A->coset_rank(G1->S, G2->S, Elt, verbose_level);

	FREE_OBJECT(G1);
	FREE_OBJECT(G2);
	FREE_int(the_set);
	FREE_int(Elt_gk);
	
	return rank;
}

void poset_classification::list_all_orbits_at_level(
	int depth,
	int f_has_print_function, 
	void (*print_function)(ostream &ost,
			int len, int *S, void *data),
	void *print_function_data, 
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int l, i;

	l = nb_orbits_at_level(depth);

	cout << "poset_classification::list_all_orbits_at_level "
			"listing all orbits "
			"at depth " << depth << ":" << endl;
	for (i = 0; i < l; i++) {
		cout << "poset_classification::list_all_orbits_at_level "
			"listing orbit "
			<< i << " / " << l << endl;
		list_whole_orbit(depth, i, 
			f_has_print_function, print_function, print_function_data, 
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
		}
}

void poset_classification::compute_integer_property_of_selected_list_of_orbits(
	int depth,
	int nb_orbits, int *Orbit_idx, 
	int (*compute_function)(int len, int *S, void *data), 
	void *compute_function_data,
	int *&Data)
{
	int l, i, j, d;
	int *set;

	set = NEW_int(depth);
	l = nb_orbits_at_level(depth);

	Data = NEW_int(nb_orbits);
	
	cout << "computing integer property for a set of "
			<< nb_orbits << " orbits at depth " << depth << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {
		i = Orbit_idx[j];
		if (i >= l) {
			cout << "orbit idx is out of range" << endl;
			exit(1);
			}
		cout << "Orbit " << j << " / " << nb_orbits
				<< " which is no " << i << ":" << endl;

		get_set_by_level(depth, i, set);

		d = (*compute_function)(depth, set, compute_function_data);
		Data[j] = d;
		}

	FREE_int(set);
}

void poset_classification::list_selected_set_of_orbits_at_level(
	int depth,
	int nb_orbits, int *Orbit_idx, 
	int f_has_print_function, 
	void (*print_function)(ostream &ost,
			int len, int *S, void *data),
	void *print_function_data, 
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int l, i, j;

	l = nb_orbits_at_level(depth);

	cout << "listing a set of " << nb_orbits
			<< " orbits at depth " << depth << ":" << endl;
	for (j = 0; j < nb_orbits; j++) {
		i = Orbit_idx[j];
		if (i >= l) {
			cout << "orbit idx is out of range" << endl;
			exit(1);
			}
		cout << "Orbit " << j << " / " << nb_orbits
				<< " which is no " << i << ":" << endl;
		list_whole_orbit(depth, i, 
			f_has_print_function, print_function, print_function_data, 
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
		}
}

void poset_classification::test_property(int depth, 
	int (*test_property_function)(int len, int *S, void *data), 
	void *test_property_data, 
	int &nb, int *&Orbit_idx)
{
	int N, i;
	int *set;

	set = NEW_int(depth);
	N = nb_orbits_at_level(depth);
	Orbit_idx = NEW_int(N);
	nb = 0;
	for (i = 0; i < N; i++) {
		get_set_by_level(depth, i, set);
		if ((*test_property_function)(depth, set, test_property_data)) {
			Orbit_idx[nb++] = i;
			}
		}
}

#if 0
void poset_classification::print_schreier_vectors_at_depth(
		int depth, int verbose_level)
{
	int i, l;

	l = nb_orbits_at_level(depth);
	for (i = 0; i < l; i++) {
		print_schreier_vector(depth, i, verbose_level);
		}
}

void poset_classification::print_schreier_vector(int depth,
		int orbit_idx, int verbose_level)
{
	int *set;
	int len;
	//strong_generators *Strong_gens;
	longinteger_object Len, L, go;
	//longinteger_domain D;
	
	set = NEW_int(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_int(orbit_idx, depth);
	L.create(len);
	
	get_stabilizer_order(depth, orbit_idx, go);


	cout << "orbit " << orbit_idx << " / " << nb_orbits_at_level(depth)
			<< " (=node " << first_poset_orbit_node_at_level[depth] + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	int_set_print(cout, set, depth);
	cout << "_" << go << endl;

	cout << "schreier tree:" << endl;

	int *sv;


	sv = root[first_poset_orbit_node_at_level[depth] + orbit_idx].sv;

	if (sv == NULL) {
		cout << "No schreier vector available" << endl;
		}

	schreier_vector_print_tree(sv, 0 /*verbose_level */);
}
#endif

void poset_classification::list_whole_orbit(
	int depth, int orbit_idx,
	int f_has_print_function, 
	void (*print_function)(ostream &ost,
			int len, int *S, void *data),
	void *print_function_data, 
	int f_show_orbit_decomposition, int f_show_stab,
	int f_save_stab, int f_show_whole_orbit)
{
	int *set;
	int rank, len;
	strong_generators *Strong_gens;
	longinteger_object Len, L, go;
	longinteger_domain D;
	
	set = NEW_int(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_int(orbit_idx, depth);
	L.create(len);
	
	get_stabilizer_order(depth, orbit_idx, go);


	cout << "poset_classification::list_whole_orbit "
			"orbit " << orbit_idx
			<< " / " << nb_orbits_at_level(depth) << " (=node "
			<< first_poset_orbit_node_at_level[depth] + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	int_set_print(cout, set, depth);
	cout << "_" << go << " ";

	print_lex_rank(set, depth);
	cout << endl;

	if (f_has_print_function) {
		(*print_function)(cout, depth, set, print_function_data);
		}

	get_stabilizer_generators(Strong_gens,
		depth, orbit_idx, 0 /* verbose_level*/);


	if (f_show_orbit_decomposition) {
		if (Poset->f_subset_lattice) {
			cout << "poset_classification::list_whole_orbit "
					"orbits on the set:" << endl;
			Strong_gens->compute_and_print_orbits_on_a_given_set(
					Poset->A2, set, depth, 0 /* verbose_level*/);
			}
		else {
			cout << "subspace_lattice not yet implemented" << endl;
		}
	
		cout << "poset_classification::list_whole_orbit "
				"orbits in the original "
				"action on the whole space:" << endl;
		Strong_gens->compute_and_print_orbits(Poset->A,
				0 /* verbose_level*/);
		}
	
	if (f_show_stab) {
		cout << "The stabilizer is generated by:" << endl;
		Strong_gens->print_generators();
		}

	if (f_save_stab) {
		char fname[1000];

		sprintf(fname, "%s_stab_%d_%d.bin",
				fname_base, depth, orbit_idx);
		cout << "saving stabilizer poset_classifications "
				"to file " << fname << endl;
		Strong_gens->write_file(fname, verbose_level);
		}


	if (f_show_whole_orbit) {
		int max_len;
		if (len > 1000) {
			max_len = 10;
			}
		else {
			max_len = len;
			}

		if (D.compare(L, Len) != 0) {
			cout << "orbit is too long to show" << endl;
			}
		else {
			for (rank = 0; rank < max_len; rank++) {
				orbit_element_unrank(depth, orbit_idx,
						rank, set, 0 /* verbose_level */);
				cout << setw(5) << rank << " : ";
				int_set_print(cout, set, depth);
				cout << endl;
				}
			if (max_len < len) {
				cout << "output truncated" << endl;
			}
			}
		}

	FREE_int(set);
	FREE_OBJECT(Strong_gens);
}

void poset_classification::get_whole_orbit(
	int depth, int orbit_idx,
	int *&Orbit, int &orbit_length, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rank;
	longinteger_object Len, L, go;
	longinteger_domain D;

	if (f_v) {
		cout << "poset_classification::get_whole_orbit" << endl;
	}
	poset_classification::orbit_length(orbit_idx, depth, Len);
	orbit_length = orbit_length_as_int(orbit_idx, depth);
	L.create(orbit_length);
	if (D.compare(L, Len) != 0) {
		cout << "poset_classification::get_whole_orbit "
				"orbit is too long" << endl;
		exit(1);
		}

	Orbit = NEW_int(orbit_length * depth);
	for (rank = 0; rank < orbit_length; rank++) {
		orbit_element_unrank(depth, orbit_idx,
				rank,
				Orbit + rank * depth,
				0 /* verbose_level */);
		}
	if (f_v) {
		cout << "poset_classification::get_whole_orbit done" << endl;
	}
}

void poset_classification::print_extensions_at_level(
		ostream &ost, int lvl)
{
	int i, node;
	int fst, len;
	poset_orbit_node *O;
	
	ost << "extensions at level " << lvl << ":" << endl;
	fst = first_poset_orbit_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);
	ost << "there are " << len << " nodes at level " << lvl << ":" << endl;
	for (i = 0; i < len; i++) {
		node = fst + i;
		O = root + node;
		ost << "Node " << i << " / " << len << " = " << node << ":" << endl;
		O->print_extensions(ost);
		}
}

void poset_classification::map_to_canonical_k_subset(
	int *the_set,
	int set_size, int subset_size, int subset_rk,
	int *reduced_set, int *transporter, int &local_idx,
	int verbose_level)
// fills reduced_set[set_size - subset_size], transporter and local_idx
// local_idx is the index of the orbit that the subset belongs to 
// (in the list of orbit of subsets of size subset_size)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset" << endl;
		}
	int *our_set;
	int *subset;
	int *canonical_subset;
	int *Elt1;
	int i, j, k; //, idx;
	int reduced_set_size;
	//int f_implicit_fusion = TRUE;
	
	our_set = NEW_int(set_size);
	subset = NEW_int(set_size);
	canonical_subset = NEW_int(set_size);
	Elt1 = NEW_int(Poset->A->elt_size_in_int);
	reduced_set_size = set_size - subset_size;

	// unrank the k-subset and its complement to our_set[set_size]:
	unrank_k_subset(subset_rk, our_set, set_size, subset_size);
	j = 0;
	k = 0;
	for (i = 0; i < set_size; i++) {
		if (j < subset_size && our_set[j] == i) {
			j++;
			continue;
			}
		our_set[subset_size + k] = i;
		k++;
		}
	for (i = 0; i < set_size; i++) {
		subset[i] = the_set[our_set[i]];
		set[0][i] = subset[i];
		}
	
	Poset->A->element_one(
			poset_classification::transporter->ith(0), FALSE);


	// trace the subset:
	
	if (f_vv) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset "
				"before trace_set" << endl;
		}
	local_idx = trace_set(
		subset, set_size, subset_size,
		canonical_subset, Elt1, 
		verbose_level - 3);

	//idx = local_idx + f;

	if (f_vv) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset "
				"after trace_set local_idx=" << local_idx << endl;
		}
	if (FALSE) {
		cout << "the transporter is" << endl;
		Poset->A->element_print(Elt1, cout);
		cout << endl;
		}
	Poset->A->element_move(Elt1, transporter, FALSE);
	for (i = 0; i < reduced_set_size; i++) {
		reduced_set[i] = canonical_subset[subset_size + i];
		}
	if (FALSE) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset reduced set = ";
		int_vec_print(cout, reduced_set, reduced_set_size);
		cout << endl;
		}
	FREE_int(Elt1);
	FREE_int(our_set);
	FREE_int(subset);
	FREE_int(canonical_subset);
	
	if (f_v) {
		cout << "poset_classification::"
				"map_to_canonical_k_subset done" << endl;
		}
}

void poset_classification::get_representative_of_subset_orbit(
	int *set, int size, int local_orbit_no, 
	strong_generators *&Strong_gens, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int fst, node, sz;
	poset_orbit_node *O;

	if (f_v) {
		cout << "poset_classification::get_representative_"
				"of_subset_orbit "
				"verbose_level=" << verbose_level << endl;
		}
	fst = first_poset_orbit_node_at_level[size];
	node = fst + local_orbit_no;
	if (f_vv) {
		cout << "poset_classification::get_representative_"
				"of_subset_orbit "
				"before get_set" << endl;
		}
	get_set(node, set, sz);
	if (sz != size) {
		cout << "get_representative_of_subset_orbit: "
				"sz != size" << endl;
		exit(1);
		}
	O = root + node;
	if (f_vv) {
		cout << "poset_classification::get_representative_"
				"of_subset_orbit "
				"before get_stabilizer_poset_classifications" << endl;
		}
	O->get_stabilizer_generators(this, Strong_gens, 0);
	if (f_v) {
		cout << "poset_classification::get_representative_"
				"of_subset_orbit done" << endl;
		}
}

void poset_classification::print_fusion_nodes(int depth)
{
	int i, f, l, j, h;

	for (i = 0; i <= depth; i++) {
		f = first_poset_orbit_node_at_level[i];
		l = nb_orbits_at_level(i);
		for (j = 0; j < l; j++) {
			poset_orbit_node *O;

			O = &root[f + j];
			for (h = 0; h < O->nb_extensions; h++) {
				extension *E = O->E + h;

				if (E->type == EXTENSION_TYPE_FUSION) {
					cout << "fusion (" << f + j << "/" << h
							<< ") -> (" << E->data1 << "/"
							<< E->data2 << ")" << endl;
					}
				}
			}
		}
}

void poset_classification::find_interesting_k_subsets(
	int *the_set, int n, int k,
	int *&interesting_sets, int &nb_interesting_sets,
	int &orbit_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	classify *C;
	int j, t, f, l, l_min, t_min = 0;

	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"n = " << n << " k = " << k << endl;
		}
	

	classify_k_subsets(the_set, n, k, C, verbose_level);


	if (f_v) {
		C->print_naked(FALSE);
		cout << endl;
		}

	l_min = INT_MAX;
	f = 0;
	for (t = 0; t < C->nb_types; t++) {
		f = C->type_first[t];
		l = C->type_len[t];
		if (l < l_min) {
			l_min = l;
			t_min = t;
			}
		}
	interesting_sets = NEW_int(l_min);
	nb_interesting_sets = l_min;
	for (j = 0; j < l_min; j++) {
		interesting_sets[j] = C->sorting_perm_inv[f + j];
		}
	orbit_idx = C->data_sorted[f];
	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"l_min = " << l_min << " t_min = " << t_min
				<< " orbit_idx = " << orbit_idx << endl;
		}
	if (f_v) {
		cout << "interesting set of size "
				<< nb_interesting_sets << " : ";
		int_vec_print(cout, interesting_sets, nb_interesting_sets);
		cout << endl;
		}

	FREE_OBJECT(C);
	
	if (f_v) {
		cout << "poset_classification::find_interesting_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
		}
}

void poset_classification::classify_k_subsets(
		int *the_set, int n, int k,
		classify *&C, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int nCk;
	int *isotype;

	if (f_v) {
		cout << "poset_classification::classify_k_subsets "
				"n = " << n << " k = " << k << endl;
		}
	
	trace_all_k_subsets(the_set, n, k, nCk,
			isotype, verbose_level);
	
	C = NEW_OBJECT(classify);

	C->init(isotype, nCk, FALSE, 0);

	if (f_v) {
		cout << "poset_classification::classify_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
		}
}

void poset_classification::trace_all_k_subsets(
		int *the_set,
		int n, int k, int &nCk, int *&isotype,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int *index_set;
	int *subset;
	int *canonical_subset;
	int *Elt;
	int subset_rk, local_idx, i;
	//int f_implicit_fusion = TRUE;

	nCk = int_n_choose_k(n, k);
	if (f_v) {
		cout << "poset_classification::trace_all_k_subsets "
				"n = " << n << " k = " << k
				<< " nCk = " << nCk << endl;
		}

	Elt = NEW_int(Poset->A->elt_size_in_int);

	index_set = NEW_int(k);
	subset = NEW_int(k);
	canonical_subset = NEW_int(k);
	isotype = NEW_int(nCk);
	
	int_vec_zero(isotype, nCk);

	first_k_subset(index_set, n, k);
	subset_rk = 0;

	while (TRUE) {
		if (f_vv && ((subset_rk % 100) == 0)) {
			cout << "poset_classification::trace_all_k_subsets "
					"k=" << k
				<< " testing set " << subset_rk << " / " << nCk 
				<< " = " << 100. * (double) subset_rk /
				(double) nCk << " % : ";
			int_vec_print(cout, index_set, k);
			cout << endl;
			}
		for (i = 0; i < k; i++) {
			subset[i] = the_set[index_set[i]];
			}
		int_vec_copy(subset, set[0], k);

		if (FALSE /*f_v2*/) {
			cout << "poset_classification::trace_all_k_subsets "
					"corresponding to set ";
			int_vec_print(cout, subset, k);
			cout << endl;
			}
		Poset->A->element_one(transporter->ith(0), 0);
		
		if (k == 0) {
			isotype[0] = 0;
			}
		else {

			if (FALSE) {
				cout << "poset_classification::trace_all_k_subsets "
						"before trace_set" << endl;
				}
			local_idx = trace_set(subset, k, k, 
				canonical_subset, Elt, 
				//f_implicit_fusion, 
				0 /*verbose_level - 3*/);
			if (FALSE) {
				cout << "poset_classification::trace_all_k_subsets "
						"after trace_set, local_idx = "
						<< local_idx << endl;
				}
			
			if (FALSE /*f_vvv*/) {
				cout << "poset_classification::trace_all_k_subsets "
						"local_idx=" << local_idx << endl;
				}
			isotype[subset_rk] = local_idx;
			if (FALSE) {
				cout << "poset_classification::trace_all_k_subsets "
						"the transporter is" << endl;
				Poset->A->element_print(Elt, cout);
				cout << endl;
				}

			}
		subset_rk++;
		if (!next_k_subset(index_set, n, k)) {
			break;
			}
		}
	if (subset_rk != nCk) {
		cout << "poset_classification::trace_all_k_subsets "
				"subset_rk != nCk" << endl;
		exit(1);
		}


	FREE_int(index_set);
	FREE_int(subset);
	FREE_int(canonical_subset);
	FREE_int(Elt);
	if (f_v) {
		cout << "poset_classification::trace_all_k_subsets done" << endl;
		}
}

void poset_classification::get_orbit_representatives(
		int level,
		int &nb_orbits, int *&Orbit_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "poset_classification::get_orbit_"
				"representatives" << endl;
		}
	nb_orbits = nb_orbits_at_level(level);
	if (f_v) {
		cout << "orbits_on_k_sets: we found " << nb_orbits
				<< " orbits on " << level << "-sets" << endl;
		}
	Orbit_reps = NEW_int(nb_orbits * level);
	for (i = 0; i < nb_orbits; i++) {
		get_set_by_level(level, i, Orbit_reps + i * level);
		}
	
	if (f_v) {
		cout << "poset_classification::get_orbit_"
				"representatives done" << endl;
		}
}

void poset_classification::unrank_point(int *v, int rk)
{
	Poset->unrank_point(v, rk);
}

int poset_classification::rank_point(int *v)
{
	int rk;

	rk = Poset->rank_point(v);
	return rk;
}

void poset_classification::unrank_basis(
		int *Basis, int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		unrank_point(Basis + i * Poset->VS->dimension, S[i]);
	}
}

void poset_classification::rank_basis(
		int *Basis, int *S, int len)
{
	int i;

	for (i = 0; i < len; i++) {
		S[i] = rank_point(Basis + i * Poset->VS->dimension);
	}
}

}



