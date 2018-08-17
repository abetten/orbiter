// generator.C
//
// Anton Betten
// December 29, 2003

#include "foundations/foundations.h"
#include "groups_and_group_actions/groups_and_group_actions.h"
#include "poset_classification/poset_classification.h"


INT generator::nb_orbits_at_level(INT level)
{
	INT f, l;

	f = first_oracle_node_at_level[level];
	l = first_oracle_node_at_level[level + 1] - f;
	return l;
}

INT generator::nb_flag_orbits_up_at_level(INT level)
{
	INT f, l, i, F;

	f = first_oracle_node_at_level[level];
	l = nb_orbits_at_level(level);
	F = 0;
	for (i = 0; i < l; i++) {
		F += root[f + i].nb_extensions;
		}
	return F;
}

oracle *generator::get_node_ij(INT level, INT node)
{
	INT f;

	f = first_oracle_node_at_level[level];
	return root + f + node;
}

INT generator::poset_structure_is_contained(
		INT *set1, INT sz1, INT *set2, INT sz2, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_contained;
	INT i, rk1, rk2;

	if (f_v) {
		cout << "poset_structure_is_contained" << endl;
		}
	if (f_vv) {
		cout << "set1: ";
		INT_vec_print(cout, set1, sz1);
		cout << " ; ";
		cout << "set2: ";
		INT_vec_print(cout, set2, sz2);
		cout << endl;
		}
	if (sz1 > sz2) {
		f_contained = FALSE;
		}
	else {
		if (f_on_subspaces) {
			INT *B1, *B2;

			B1 = NEW_INT(sz1 * vector_space_dimension);
			B2 = NEW_INT((sz1 + sz2) * vector_space_dimension);

			for (i = 0; i < sz1; i++) {
				unrank_point(B1 + i * vector_space_dimension, set1[i]);
				//(*unrank_point_func)(B1 + i * vector_space_dimension,
				//set1[i], rank_point_data);
				}
			for (i = 0; i < sz2; i++) {
				unrank_point(B2 + i * vector_space_dimension, set2[i]);
				//(*unrank_point_func)(B2 + i * vector_space_dimension,
				//set2[i], rank_point_data);
				}

			rk1 = F->Gauss_easy(B1, sz1, vector_space_dimension);
			if (rk1 != sz1) {
				cout << "poset_structure_is_contained rk1 != sz1" << endl;
				exit(1);
				}
			
			rk2 = F->Gauss_easy(B2, sz2, vector_space_dimension);
			if (rk2 != sz2) {
				cout << "poset_structure_is_contained rk2 != sz2" << endl;
				exit(1);
				}
			INT_vec_copy(B1,
					B2 + sz2 * vector_space_dimension,
					sz1 * vector_space_dimension);
			rk2 = F->Gauss_easy(B2, sz1 + sz2,
					vector_space_dimension);
			if (rk2 > sz2) {
				f_contained = FALSE;
				}
			else {
				f_contained = TRUE;
				}

			FREE_INT(B1);
			FREE_INT(B2);
			}
		else {
			f_contained = INT_vec_sort_and_test_if_contained(
					set1, sz1, set2, sz2);
			}
		}
	return f_contained;
}

void generator::print_progress_by_extension(INT size,
		INT cur, INT prev, INT cur_ex, INT nb_ext_cur, INT nb_fuse_cur)
{
	double progress;
	
	
	progress = level_progress(size);
		
	print_level_info(size + 1, prev);
	cout << " **** Upstep extension " << cur_ex << " / "
		<< root[prev].nb_extensions << " with "
		<< nb_ext_cur << " new orbits and " 
		<< nb_fuse_cur << " fusion nodes. We now have " 
		<< cur - first_oracle_node_at_level[size + 1] 
		<< " nodes at level " << size + 1;
		cout << ", ";
	print_progress(size + 1, progress);
	print_progress_by_level(size + 1);
}

void generator::print_progress(INT size,
		INT cur, INT prev, INT nb_ext_cur, INT nb_fuse_cur)
{
	double progress;
	
	
	progress = level_progress(size);
		
	print_level_info(size + 1, prev);
	cout << " **** Upstep finished with " 
		<< nb_ext_cur << " new orbits and " 
		<< nb_fuse_cur << " fusion nodes. We now have " 
		<< cur - first_oracle_node_at_level[size + 1] 
		<< " nodes at level " << size + 1;
		cout << ", ";
	print_progress(size + 1, progress);
	print_progress_by_level(size + 1);
}

void generator::print_progress(INT lvl, double progress)
{
	double progress0;
	INT progress1, progress2;
		
	progress0 = progress * 100.;
	progress2 = (INT) (progress0 * 100.);
	progress1 = progress2 / 100;
	progress2 = progress2 % 100;
	cout << "progress: " << progress1 << "."
			<< setw(2) << progress2 << " % " << endl;
}

void generator::print_progress_by_level(INT lvl)
{
	INT i;
	
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

void generator::print_orbit_numbers(INT depth)
{
	INT nb_nodes, j;
	
	nb_nodes = nb_orbits_at_level(depth);
	cout << "###########################################################"
			"#######################################" << endl;
	print_problem_label();
	cout << "Found " << nb_nodes << " orbits at depth " << depth << endl;
	for (j = 0; j <= depth; j++) {
		cout << j << " : " << nb_orbits_at_level(j) << " orbits" << endl;
		}
	cout << "total: " << first_oracle_node_at_level[depth + 1] << endl;
	//gen->print_statistic_on_callbacks();
	compute_and_print_automorphism_group_orders(depth, cout);
}


void generator::print_statistic_on_callbacks_naked()
{
	cout << A->nb_times_image_of_called - nb_times_image_of_called0 << "/";
	cout << A->nb_times_mult_called - nb_times_mult_called0 << "/";
	cout << A->nb_times_invert_called - nb_times_invert_called0 << "/";
	cout << A->nb_times_retrieve_called - nb_times_retrieve_called0 << "/";
	cout << A->nb_times_store_called - nb_times_store_called0;
}

void generator::print_statistic_on_callbacks()
{
	cout << "# of calls to image_of/mult/invert/retrieve/store: ";
	print_statistic_on_callbacks_naked();
	cout << endl;
}

set_and_stabilizer *generator::get_set_and_stabilizer(
		INT level, INT orbit_at_level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	set_and_stabilizer *SaS;

	if (f_v) {
		cout << "generator::get_set_and_stabilizer" << endl;
		}
	SaS = new set_and_stabilizer;
	SaS->init(A, A2, 0 /*verbose_level */);
	SaS->allocate_data(level, 0 /* verbose_level */);
	get_set_by_level(level, orbit_at_level, SaS->data);
	get_stabilizer_generators(SaS->Strong_gens,  
		level, orbit_at_level, 0 /* verbose_level */);
	SaS->Strong_gens->group_order(SaS->target_go);
	SaS->Stab = SaS->Strong_gens->create_sims(0 /*verbose_level*/);
	if (f_v) {
		cout << "generator::get_set_and_stabilizer done" << endl;
		}
	return SaS;
}

void generator::get_set_by_level(INT level, INT node, INT *set)
{
	INT size;
	oracle *O;
	
	O = get_node_ij(level, node);
	//n = first_oracle_node_at_level[level] + node;
	size = O->depth_of_node(this);
	//size = root[n].depth_of_node(this);
	if (size != level) {
		cout << "generator::get_set_by_level size != level" << endl;
		exit(1);
		}
	//root[n].store_set_to(this, size - 1, set);
	O->store_set_to(this, size - 1, set);
}

void generator::get_set(INT node, INT *set, INT &size)
{
	size = root[node].depth_of_node(this);
	root[node].store_set_to(this, size - 1, set);
}

void generator::get_set(INT level, INT orbit, INT *set, INT &size)
{
	INT node;

	node = first_oracle_node_at_level[level] + orbit;
	size = root[node].depth_of_node(this);
	root[node].store_set_to(this, size - 1, set);
}

void generator::print_set_verbose(INT node)
{
	root[node].print_set_verbose(this);
}

void generator::print_set_verbose(INT level, INT orbit)
{
	INT node;

	node = first_oracle_node_at_level[level] + orbit;
	root[node].print_set_verbose(this);
}

void generator::print_set(INT node)
{
	root[node].print_set(this);
}

void generator::print_set(INT level, INT orbit)
{
	INT node;

	node = first_oracle_node_at_level[level] + orbit;
	root[node].print_set(this);
}

INT generator::find_oracle_node_for_set(INT len,
		INT *set, INT f_tolerant, INT verbose_level)
// finds the node that represents s_0,...,s_{len - 1}
{
	INT f_v = (verbose_level >= 1);
	INT ret;
	
	if (f_v) {
		cout << "generator::find_oracle_node_for_set ";
		INT_vec_print(cout, set, len);
		cout << endl;
		}
	if (f_starter) {
		INT i, j, h;
		if (len < starter_size) {
			cout << "generator::find_oracle_node_for_set "
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
				cout << "generator::find_oracle_node_for_set "
						"did not find " << i << "-th element "
						"of the starter" << endl;
				}
			for (h = j; h > i; h--) {
				set[h] = set[h - 1];
				}
			set[i] = starter[i];
			}
		INT from = starter_size;
		INT node = starter_size;
		ret = find_oracle_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
		}
	else {
		INT from = 0;
		INT node = 0;
		ret = find_oracle_node_for_set_basic(from,
				node, len, set, f_tolerant, verbose_level);
		}
	if (ret == -1) {
		if (f_tolerant) {
			if (f_v) {
				cout << "generator::find_oracle_node_for_set ";
				INT_vec_print(cout, set, len);
				cout << " extension not found, "
						"we are tolerant, returnning -1" << endl;
				}
			return -1;
			}
		else {
			cout << "generator::find_oracle_node_for_set "
					"we should not be here" << endl;
			exit(1);
			}
		}
	return ret;
	
}

INT generator::find_oracle_node_for_set_basic(INT from,
		INT node, INT len, INT *set, INT f_tolerant,
		INT verbose_level)
{
	INT i, j, pt;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 1);

	if (f_vv) {
		cout << "generator::find_oracle_node_for_set_basic "
				"looking for set ";
		INT_vec_print(cout, set, len);
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
				cout << "generator::find_oracle_node_for_set_basic "
						"depth " << i << " no extension for point "
						<< pt << " found" << endl;
				}
			if (f_tolerant) {
				if (f_v) {
					cout << "generator::find_oracle_node_for_set_basic "
							"since we are tolerant, we return -1" << endl;
					}
				return -1;
				}
			else {
				cout << "generator::find_oracle_node_for_set_basic "
						"failure in find_extension_from_point" << endl;
				INT_vec_print(cout, set, len);
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
			cout << "generator::find_oracle_node_for_set() "
					"root[node].E[j].pt != pt" << endl;
			exit(1);
			}
		if (root[node].E[j].type != EXTENSION_TYPE_EXTENSION && 
			root[node].E[j].type != EXTENSION_TYPE_PROCESSING) {
			cout << "generator::find_oracle_node_for_set() "
					"root[node].E[j].type != "
					"EXTENSION_TYPE_EXTENSION" << endl;
			cout << "root[node].E[j].type="
					<< root[node].E[j].type << " = ";
			print_extension_type(cout, root[node].E[j].type);
			cout << endl;
			cout << "generator::find_oracle_node_for_set_basic "
					"looking for set ";
			INT_vec_print(cout, set, len);
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
					<< " new node " << node << endl;
			}
		}
	return node;
}

void generator::oracle_depth_breadth_perm_and_inverse(INT max_depth, 
	INT *&perm, INT *&perm_inv, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT idx = 0;
	INT N;

	if (f_v) {
		cout << "generator::oracle_depth_breadth_perm_and_inverse" << endl;
		cout << "max_depth = " << max_depth << endl;
		}

	N = first_oracle_node_at_level[max_depth + 1];
	if (f_v) {
		cout << "N = first_oracle_node_at_level[max_depth + 1] = "
				<< N << endl;
		}
	
	perm = NEW_INT(N);
	perm_inv = NEW_INT(N);
	
	if (f_v) {
		cout << "calling root->oracle_depth_breadth_perm_and_inverse" << endl;
		}
	root->oracle_depth_breadth_perm_and_inverse(this,
			max_depth, idx, 0, 0, perm, perm_inv);
	
}

INT generator::count_extension_nodes_at_level(INT lvl)
{
	INT prev;

	nb_extension_nodes_at_level_total[lvl] = 0;
	for (prev = first_oracle_node_at_level[lvl]; 
			prev < first_oracle_node_at_level[lvl + 1]; prev++) {
			
		nb_extension_nodes_at_level_total[lvl] +=
				root[prev].nb_extensions;
		
		}
	nb_unprocessed_nodes_at_level[lvl] =
			nb_extension_nodes_at_level_total[lvl];
	nb_fusion_nodes_at_level[lvl] = 0;
	nb_extension_nodes_at_level[lvl] = 0;
	return nb_extension_nodes_at_level_total[lvl];
}

double generator::level_progress(INT lvl)
{
	return
		((double)(nb_fusion_nodes_at_level[lvl] +
				nb_extension_nodes_at_level[lvl])) /
			(double) nb_extension_nodes_at_level_total[lvl];
}



void generator::count_automorphism_group_orders(
	INT lvl, INT &nb_agos,
	longinteger_object *&agos, INT *&multiplicities,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, l, j, c, h, f_added;
	longinteger_object ago;
	longinteger_object *tmp_agos;
	INT *tmp_multiplicities;
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
					agos = new longinteger_object[nb_agos + 1];
					multiplicities = NEW_INT(nb_agos + 1);
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
						delete [] tmp_agos;
						FREE_INT(tmp_multiplicities);
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
			agos = new longinteger_object[nb_agos + 1];
			multiplicities = NEW_INT(nb_agos + 1);
			for (h = 0; h < nb_agos; h++) {
				tmp_agos[h].swap_with(agos[h]);
				multiplicities[h] = tmp_multiplicities[h];
				}
			ago.swap_with(agos[nb_agos]);
			multiplicities[nb_agos] = 1;
			nb_agos++;
			if (tmp_agos) {
				delete [] tmp_agos;
				FREE_INT(tmp_multiplicities);
				}
			}
		}
}

void generator::compute_and_print_automorphism_group_orders(
		INT lvl, ostream &ost)
{

	INT j, nb_agos;
	longinteger_object *agos;
	INT *multiplicities;
	INT N, r, h;
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
	D.integral_division_by_INT(S, N, Q, r);
	

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
		delete [] agos;
		FREE_INT(multiplicities);
		}

}

void generator::stabilizer_order(INT node, longinteger_object &go)
{
	if (root[node].nb_strong_generators) {
		go.create_product(A->base_len, root[node].tl);
		}
	else {
		go.create(1);
		}
}

INT generator::check_the_set(INT len, INT *S, INT verbose_level)
// used by lookahead_first_at_level
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "generator::check_the_set" << endl;
		}
	if (!f_candidate_check_func) {
		return TRUE;
		}
	
	if (f_v) {
		cout << "generator::check_the_set checking set: ";
		INT_set_print(cout, S, len);
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

INT generator::check_the_set_incrementally(INT len,
		INT *S, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (!f_candidate_incremental_check_func) {
		return check_the_set(len, S, verbose_level);
		}
	
	if (f_v) {
		cout << "generator::check_the_set_incrementally checking set: ";
		INT_set_print(cout, S, len);
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



void generator::orbit_length(INT orbit_at_level,
		INT level, longinteger_object &len)
// uses generator::go for the group order
{
	longinteger_domain D;
	longinteger_object stab_order, quo, rem;

	get_stabilizer_order(level, orbit_at_level, stab_order);
	D.integral_division(go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "generator::orbit_length stabilizer order does "
				"not divide group order" << endl;
		exit(1);
		}
}

void generator::get_orbit_length_and_stabilizer_order(INT node,
		INT level, longinteger_object &stab_order,
		longinteger_object &len)
// uses generator::go for the group order
{
	longinteger_domain D;
	longinteger_object quo, rem;

	get_stabilizer_order(level, node, stab_order);
	D.integral_division(go, stab_order, len, rem, 0);
	if (!rem.is_zero()) {
		cout << "generator::orbit_length stabilizer order "
				"does not divide group order" << endl;
		exit(1);
		}
}

INT generator::orbit_length_as_INT(INT orbit_at_level, INT level)
{
	longinteger_object len;

	orbit_length(orbit_at_level, level, len);
	return len.as_INT();
	
}


void generator::print_representatives_at_level(INT lvl)
{
	INT i, f, l;
	
	f = first_oracle_node_at_level[lvl];
	l = nb_orbits_at_level(lvl);
	cout << "The " << l << " representatives at level "
			<< lvl << " are:" << endl;
	for (i = 0; i < l; i++) {
		cout << i << " / " << l << " : ";
		root[f + i].print_set(this);
		cout << endl;
		}
}

void generator::print_lex_rank(INT *set, INT sz)
{
	INT r1, r2;
	INT n;
	
	n = A2->degree;
	r1 = rank_subset(set, sz, n);
	r2 = rank_k_subset(set, n, sz);

	cout << "lex rank = " << r1 << " lex rank as "
			<< sz << "-subset = " << r2;
}


void generator::print_problem_label()
{
	if (problem_label[0]) {
		cout << problem_label << " ";
		}
}

void generator::print_level_info(INT i, INT prev)
{
	INT t1, dt;
	
	t1 = os_ticks();
	//cout << "generator::print_level_info t0=" << t0 << endl;
	//cout << "generator::print_level_info t1=" << t1 << endl;
	dt = t1 - t0;
	//cout << "generator::print_level_info dt=" << dt << endl;
	
	cout << "Time ";
	time_check_delta(cout, dt);
	print_problem_label();
	cout << " : Level " << i - 1 << " Node " << prev << " = " 
		<< prev - first_oracle_node_at_level[i - 1] 
		<< " / " 
		<< nb_orbits_at_level(i - 1)
		<< " : ";
}

void generator::print_level_extension_info(INT i, 
	INT prev, INT cur_extension)
{
#if 0
	INT t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	
	cout << "Time ";
	time_check_delta(cout, dt);
	print_problem_label();
#endif
	cout << "Level " << i - 1 << " Node " << prev << " = " 
		<< prev - first_oracle_node_at_level[i - 1] 
		<< " / " 
		<< nb_orbits_at_level(i - 1)
		<< " Extension " << cur_extension 
		<< " / " 
		<< root[prev].nb_extensions 
		<< " : ";
}

void generator::print_level_extension_coset_info(INT i, 
	INT prev, INT cur_extension, INT coset, INT nb_cosets)
{
#if 0
	INT t1, dt;
	
	t1 = os_ticks();
	dt = t1 - t0;
	
	cout << "Time ";
	time_check_delta(cout, dt);
	print_problem_label();
#endif
	cout << "Level " << i - 1 << " Node " << prev << " = " 
		<< prev - first_oracle_node_at_level[i - 1] 
		<< " / " 
		<< nb_orbits_at_level(i - 1)
		<< " Extension " << cur_extension 
		<< " / " 
		<< root[prev].nb_extensions 
		<< " : " 
		<< "Coset " << coset << " / " << nb_cosets << " : ";
}

void generator::recreate_schreier_vectors_up_to_level(INT lvl,
		INT f_compact, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "generator::recreate_schreier_vectors_up_to_level "
				"creating Schreier vectors up to level " << lvl << endl;
		}
	for (i = 0; i <= lvl; i++) {
		if (f_v) {
			cout << "generator::recreate_schreier_vectors_up_to_level "
					"creating Schreier vectors at level " << i << endl;
			}
		recreate_schreier_vectors_at_level(i, f_compact, verbose_level);
		}
}

void generator::recreate_schreier_vectors_at_level(INT i,
		INT f_compact, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f, cur, l, prev, u;
	INT f_recreate_extensions = FALSE;
	INT f_dont_keep_sv = FALSE;

	f = first_oracle_node_at_level[i];
	cur = first_oracle_node_at_level[i + 1];
	l = cur - f;

	if (f_v) {
		cout << "creating Schreier vectors at depth " << i
				<< " for " << l << " orbits" << endl;
		}
	if (f_v) {
		cout << "generator::recreate_schreier_vectors_at_level "
				"Testing if a schreier vector file exists" << endl;
		}
	if (test_sv_level_file_binary(i, fname_base)) {

		if (f_v) {
			cout << "generator::recreate_schreier_vectors_at_level "
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
		cout << "generator::recreate_schreier_vectors_at_level "
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
			cout << "generator::recreate_schreier_vectors_at_level " 
				<< i << " node " << u << " / " << l << endl;
			}
			
		root[prev].compute_schreier_vector(this, i,
				f_compact, verbose_level - 1);
		}
	write_sv_level_file_binary(i, fname_base, FALSE, 0, 0, verbose_level);
	if (f_v) {
		cout << "generator::recreate_schreier_vectors_at_level "
				"Written a file with Schreier vectors at depth " << i << endl;
		}
	if (f_v) {
		cout << endl;
		}
		
}

void generator::print_node(INT node)
{
	cout << "generator::print_node node " << node << ":" << endl;
	root[node].print_node(this);
}

void generator::print_tree()
{
	INT i;
	
	cout << "generator::print_tree nb_oracle_nodes_used="
			<< nb_oracle_nodes_used << endl;
	for (i = 0; i < nb_oracle_nodes_used; i++) {
		print_node(i);
		}
}

void generator::get_table_of_nodes(INT *&Table,
		INT &nb_rows, INT &nb_cols, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "generator::get_table_of_nodes "
				"nb_oracle_nodes_used=" << nb_oracle_nodes_used << endl;
		}
	nb_rows = nb_oracle_nodes_used;
	nb_cols = 6;
	Table = NEW_INT(nb_oracle_nodes_used * nb_cols);
		
	for (i = 0; i < nb_oracle_nodes_used; i++) {

		if (f_v) {
			cout << "generator::get_table_of_nodes node " << i
					<< " / " << nb_oracle_nodes_used << endl;
			}

		Table[i * nb_cols + 0] = root[i].get_level(this);
		Table[i * nb_cols + 1] = root[i].get_node_in_level(this);
		Table[i * nb_cols + 2] = root[i].pt;

		longinteger_object go;
			
		root[i].get_stabilizer_order(this, go);
		Table[i * nb_cols + 3] = go.as_INT();
		Table[i * nb_cols + 4] = root[i].get_nb_of_live_points();
		Table[i * nb_cols + 5] = root[i].get_nb_of_orbits_under_stabilizer();
		}
	if (f_v) {
		cout << "generator::get_table_of_nodes done" << endl;
		}
}

INT generator::count_live_points(INT level,
		INT node_local, INT f_compact, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_v2 = FALSE;
	INT node;
	INT *osv;
	INT nb_points;
	INT *pt_list;
	//INT *pt_list2;
	//INT a, i, j;
	
	if (f_v) {
		cout << "generator::count_live_points" << endl;
		}
	node = first_oracle_node_at_level[level] + node_local;
	if (root[node].sv == NULL) {
		root[node].compute_schreier_vector(this, 
			level, f_compact, verbose_level - 2);
		}
	osv = root[node].sv;
	nb_points = osv[0];
	pt_list = osv + 1;
	if (f_v2) {
		cout << "pt_list of size " << nb_points << " : ";
		INT_vec_print(cout, pt_list, nb_points);
		cout << endl;
		}

#if 0
	pt_list2 = NEW_INT(nb_points);
	a = root[node].pt;
	j = 0;
	for (i = 0; i < nb_points; i++) {
		if (pt_list[i] > a) {
			pt_list2[j++] = pt_list[i]; // why is this needed?
			}
		}
	nb_points = j;
	FREE_INT(pt_list2);
#endif

	return nb_points;
}

void generator::find_automorphism_group_of_order(INT level, INT order)
{
	INT nb_nodes, node, i, j, elt_order;
	longinteger_object ago;
	INT set[300];
	
	nb_nodes = nb_orbits_at_level(level);
	for (i = 0; i < nb_nodes; i++) {
		node = first_oracle_node_at_level[level] + i;
		if (root[node].nb_strong_generators == 0) {
			ago.create(1);
			}
		else {
			ago.create_product(A->base_len, root[node].tl);
			}
		if (ago.as_INT() == order) {
			cout << "found a node whose automorphism group is order "
					<< order << endl;
			cout << "the node is # " << i << " at level "
					<< level << endl;
			get_set(first_oracle_node_at_level[level] + i,
					set, level);
			INT_vec_print(cout, set, level);
			cout << endl;
			
			strong_generators *Strong_gens;
			
			get_stabilizer_generators(Strong_gens,  
				level, i, 0  /* verbose_level */);
				
			for (j = 0; j < Strong_gens->gens->len; j++) {
				elt_order = A->element_order(Strong_gens->gens->ith(j));
				cout << "generator " << j << " of order "
						<< elt_order << ":" << endl;
				if (order == elt_order) {
					cout << "CYCLIC" << endl;
					}
				A->element_print(
						Strong_gens->gens->ith(j), cout);
				A->element_print_as_permutation(
						Strong_gens->gens->ith(j), cout);
				}
			delete Strong_gens;
			}
		}
}

void generator::get_stabilizer_order(INT level,
		INT orbit_at_level, longinteger_object &go)
{
	oracle *O;
	INT nd;

	nd = first_oracle_node_at_level[level] + orbit_at_level;
	O = root + nd;

	if (O->nb_strong_generators == 0) {
		go.create(1);
		}
	else {
		longinteger_domain D;

		D.multiply_up(go, O->tl, A->base_len);
		}
}

void generator::get_stabilizer_group(group *&G,  
	INT level, INT orbit_at_level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	oracle *O;
	INT node;

	if (f_v) {
		cout << "generator::get_stabilizer_group level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
		}
	G = new group;
	node = first_oracle_node_at_level[level] + orbit_at_level;
	O = root + node;

	G->init(A);
	if (f_vv) {
		cout << "generator::get_stabilizer_group before "
				"G->init_strong_generators_by_hdl" << endl;
		}
	G->init_strong_generators_by_hdl(O->nb_strong_generators,
			O->hdl_strong_generators, O->tl, FALSE);
	G->schreier_sims(0);
	
	if (f_v) {
		cout << "generator::get_stabilizer_group level=" << level
				<< " orbit_at_level=" << orbit_at_level << " done" << endl;
		}
}

void generator::get_stabilizer_generators(strong_generators *&gens,  
	INT level, INT orbit_at_level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generator::get_stabilizer_generators level=" << level
				<< " orbit_at_level=" << orbit_at_level << endl;
		}
	group *G;

	get_stabilizer_group(G, level, orbit_at_level, verbose_level - 1);

	gens = new strong_generators;

	gens->init_from_sims(G->S, 0 /* verbose_level */);
	delete G;
	if (f_v) {
		cout << "generator::get_stabilizer_generators level=" << level
				<< " orbit_at_level=" << orbit_at_level << " done" << endl;
		}

}

void generator::change_extension_type(INT level,
		INT node, INT cur_ext, INT type, INT verbose_level)
{
	if (type == EXTENSION_TYPE_EXTENSION) {
		// extension node
		if (root[node].E[cur_ext].type != EXTENSION_TYPE_UNPROCESSED && 
			root[node].E[cur_ext].type != EXTENSION_TYPE_PROCESSING) {
			cout << "generator::change_extension_type trying to install "
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
			cout << "generator::change_extension_type trying to install "
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

void generator::orbit_element_unrank(INT depth,
		INT orbit_idx, INT rank, INT *set,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1;
	INT *Elt2;
	INT *the_set;
	oracle *O;


	if (f_v) {
		cout << "generator::orbit_element_unrank depth=" << depth
				<< " orbit_idx=" << orbit_idx
				<< " rank=" << rank << endl;
		}

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	the_set = NEW_INT(depth);
	
	O = &root[first_oracle_node_at_level[depth] + orbit_idx];
	coset_unrank(depth, orbit_idx, rank, Elt1,
			0/*verbose_level*/);

	A->element_invert(Elt1, Elt2, 0);
	O->store_set_to(this, depth - 1, the_set);
	A2->map_a_set(the_set, set, depth, Elt2,
			0/*verbose_level*/);

	FREE_INT(the_set);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_v) {
		cout << "generator::orbit_element_unrank ";
		INT_vec_print(cout, set, depth);
		cout << endl;
		}
}

void generator::orbit_element_rank(INT depth,
	INT &orbit_idx, INT &rank, INT *set,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt1;
	INT *the_set;
	INT *canonical_set;
	INT i;


	if (f_v) {
		cout << "generator::orbit_element_rank "
				"depth=" << depth << " ";
		INT_vec_print(cout, set, depth);
		cout << endl;
		}

	Elt1 = NEW_INT(A->elt_size_in_INT);
	the_set = NEW_INT(depth);
	canonical_set = NEW_INT(depth);
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
		cout << "generator::orbit_element_rank after trace_set, "
				"orbit_idx = " << orbit_idx << endl;
		cout << "transporter:" << endl;
		A->element_print_quick(Elt1, cout);
		cout << "as permutation:" << endl;
		A2->element_print_as_permutation(Elt1, cout);
		}
	if (f_v) {
		cout << "calling coset_rank" << endl;
		}
	rank = coset_rank(depth, orbit_idx, Elt1, verbose_level);
	if (f_v) {
		cout << "after coset_rank, rank=" << rank << endl;
		}
		
	FREE_INT(Elt1);
	FREE_INT(the_set);
	FREE_INT(canonical_set);
	if (f_v) {
		cout << "generator::orbit_element_rank orbit_idx="
				<< orbit_idx << " rank=" << rank << endl;
		}
}

void generator::coset_unrank(INT depth, INT orbit_idx,
		INT rank, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *the_set;
	group *G1, *G2;
	INT *Elt_gk;
	longinteger_object G_order, U_order;
	oracle *O1, *O2;

	if (f_v) {
		cout << "generator::coset_unrank depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		A->print_info();
		cout << "action A2:" << endl;
		A2->print_info();
		}

	O1 = &root[0];
	O2 = &root[first_oracle_node_at_level[depth] + orbit_idx];


	
	G1 = new group;
	G2 = new group;
	the_set = NEW_INT(depth);
	Elt_gk = NEW_INT(A->elt_size_in_INT);
	
	O2->store_set_to(this, depth - 1, the_set);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		INT_vec_print(cout, the_set, depth);
		cout << endl;
		}
	
	O1->get_stabilizer(this,
			*G1, G_order, verbose_level - 2);
	O2->get_stabilizer(this,
			*G2, U_order, verbose_level - 2);


	A->coset_unrank(G1->S, G2->S, rank, Elt, verbose_level);

	delete G1;
	delete G2;
	FREE_INT(the_set);
	FREE_INT(Elt_gk);

}

INT generator::coset_rank(INT depth, INT orbit_idx,
		INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rank;
	INT *the_set;
	group *G1, *G2;
	INT *Elt_gk;
	longinteger_object G_order, U_order;
	oracle *O1, *O2;

	if (f_v) {
		cout << "generator::coset_rank depth=" << depth
				<< " orbit_idx=" << orbit_idx << endl;
		cout << "action A:" << endl;
		A->print_info();
		cout << "action A2:" << endl;
		A2->print_info();
		}

	O1 = &root[0];
	O2 = &root[first_oracle_node_at_level[depth] + orbit_idx];


	
	G1 = new group;
	G2 = new group;
	the_set = NEW_INT(depth);
	Elt_gk = NEW_INT(A->elt_size_in_INT);
	
	O2->store_set_to(this, depth - 1, the_set);
	
	if (f_v) {
		cout << "the set representing orbit " << orbit_idx 
			<< " at level " << depth << " is ";
		INT_vec_print(cout, the_set, depth);
		cout << endl;
		}
	
	O1->get_stabilizer(this, *G1, G_order, verbose_level - 2);
	O2->get_stabilizer(this, *G2, U_order, verbose_level - 2);


	rank = A->coset_rank(G1->S, G2->S, Elt, verbose_level);

	delete G1;
	delete G2;
	FREE_INT(the_set);
	FREE_INT(Elt_gk);
	
	return rank;
}

void generator::list_all_orbits_at_level(INT depth, 
	INT f_has_print_function, 
	void (*print_function)(INT len, INT *S, void *data), 
	void *print_function_data, 
	INT f_show_orbit_decomposition, INT f_show_stab,
	INT f_save_stab, INT f_show_whole_orbit)
{
	INT l, i;

	l = nb_orbits_at_level(depth);

	cout << "generator::list_all_orbits_at_level listing all orbits "
			"at depth " << depth << ":" << endl;
	for (i = 0; i < l; i++) {
		cout << "generator::list_all_orbits_at_level listing orbit "
			<< i << " / " << l << endl;
		list_whole_orbit(depth, i, 
			f_has_print_function, print_function, print_function_data, 
			f_show_orbit_decomposition, f_show_stab,
			f_save_stab, f_show_whole_orbit);
		}
}

void generator::compute_integer_property_of_selected_list_of_orbits(
	INT depth,
	INT nb_orbits, INT *Orbit_idx, 
	INT (*compute_function)(INT len, INT *S, void *data), 
	void *compute_function_data,
	INT *&Data)
{
	INT l, i, j, d;
	INT *set;

	set = NEW_INT(depth);
	l = nb_orbits_at_level(depth);

	Data = NEW_INT(nb_orbits);
	
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

	FREE_INT(set);
}

void generator::list_selected_set_of_orbits_at_level(INT depth, 
	INT nb_orbits, INT *Orbit_idx, 
	INT f_has_print_function, 
	void (*print_function)(INT len, INT *S, void *data), 
	void *print_function_data, 
	INT f_show_orbit_decomposition, INT f_show_stab,
	INT f_save_stab, INT f_show_whole_orbit)
{
	INT l, i, j;

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

void generator::test_property(INT depth, 
	INT (*test_property_function)(INT len, INT *S, void *data), 
	void *test_property_data, 
	INT &nb, INT *&Orbit_idx)
{
	INT N, i;
	INT *set;

	set = NEW_INT(depth);
	N = nb_orbits_at_level(depth);
	Orbit_idx = NEW_INT(N);
	nb = 0;
	for (i = 0; i < N; i++) {
		get_set_by_level(depth, i, set);
		if ((*test_property_function)(depth, set, test_property_data)) {
			Orbit_idx[nb++] = i;
			}
		}
}

void generator::print_schreier_vectors_at_depth(
		INT depth, INT verbose_level)
{
	INT i, l;

	l = nb_orbits_at_level(depth);
	for (i = 0; i < l; i++) {
		print_schreier_vector(depth, i, verbose_level);
		}
}

void generator::print_schreier_vector(INT depth,
		INT orbit_idx, INT verbose_level)
{
	INT *set;
	INT len;
	//strong_generators *Strong_gens;
	longinteger_object Len, L, go;
	//longinteger_domain D;
	
	set = NEW_INT(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_INT(orbit_idx, depth);
	L.create(len);
	
	get_stabilizer_order(depth, orbit_idx, go);


	cout << "orbit " << orbit_idx << " / " << nb_orbits_at_level(depth)
			<< " (=node " << first_oracle_node_at_level[depth] + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	INT_set_print(cout, set, depth);
	cout << "_" << go << endl;

	cout << "schreier tree:" << endl;

	INT *sv;


	sv = root[first_oracle_node_at_level[depth] + orbit_idx].sv;

	if (sv == NULL) {
		cout << "No schreier vector available" << endl;
		}

	schreier_vector_print_tree(sv, 0 /*verbose_level */);
}

void generator::list_whole_orbit(INT depth, INT orbit_idx, 
	INT f_has_print_function, 
	void (*print_function)(INT len, INT *S, void *data), 
	void *print_function_data, 
	INT f_show_orbit_decomposition, INT f_show_stab,
	INT f_save_stab, INT f_show_whole_orbit)
{
	INT *set;
	INT rank, len;
	strong_generators *Strong_gens;
	longinteger_object Len, L, go;
	longinteger_domain D;
	
	set = NEW_INT(depth);

	orbit_length(orbit_idx, depth, Len);
	len = orbit_length_as_INT(orbit_idx, depth);
	L.create(len);
	
	get_stabilizer_order(depth, orbit_idx, go);


	cout << "generator::list_whole_orbit orbit " << orbit_idx
			<< " / " << nb_orbits_at_level(depth) << " (=node "
			<< first_oracle_node_at_level[depth] + orbit_idx
			<< ") at depth " << depth << " has length " << Len << " : ";

	get_set_by_level(depth, orbit_idx, set);
	INT_set_print(cout, set, depth);
	cout << "_" << go << " ";

	print_lex_rank(set, depth);
	cout << endl;

	if (f_has_print_function) {
		(*print_function)(depth, set, print_function_data);
		}

	get_stabilizer_generators(Strong_gens,  
		depth, orbit_idx, 0 /* verbose_level*/);


	if (f_show_orbit_decomposition) {
		if (!f_on_subspaces) {
			cout << "generator::list_whole_orbit "
					"orbits on the set:" << endl;
			Strong_gens->compute_and_print_orbits_on_a_given_set(
					A2, set, depth, 0 /* verbose_level*/);
			}
	
		cout << "generator::list_whole_orbit orbits in the original "
				"action on the whole space:" << endl;
		Strong_gens->compute_and_print_orbits(A, 0 /* verbose_level*/);
		}
	
	if (f_show_stab) {
		cout << "The stabilizer is generated by:" << endl;
		Strong_gens->print_generators();
		}

	if (f_save_stab) {
		BYTE fname[1000];

		sprintf(fname, "%s_stab_%ld_%ld.bin", fname_base, depth, orbit_idx);
		cout << "saving stabilizer generators to file " << fname << endl;
		Strong_gens->write_file(fname, verbose_level);
		}


	if (f_show_whole_orbit) {
		INT max_len;
		if (len > 10) {
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
				INT_set_print(cout, set, depth);
				cout << endl;
				}	
			}
		}

	FREE_INT(set);
	delete Strong_gens;
}

void generator::print_extensions_at_level(ostream &ost, INT lvl)
{
	INT i, node;
	INT fst, len;
	oracle *O;
	
	ost << "extensions at level " << lvl << ":" << endl;
	fst = first_oracle_node_at_level[lvl];
	len = nb_orbits_at_level(lvl);
	ost << "there are " << len << " nodes at level " << lvl << ":" << endl;
	for (i = 0; i < len; i++) {
		node = fst + i;
		O = root + node;
		ost << "Node " << i << " / " << len << " = " << node << ":" << endl;
		O->print_extensions(ost);
		}
}

void generator::map_to_canonical_k_subset(INT *the_set,
	INT set_size, INT subset_size, INT subset_rk,
	INT *reduced_set, INT *transporter, INT &local_idx,
	INT verbose_level)
// fills reduced_set[set_size - subset_size], transporter and local_idx
// local_idx is the index of the orbit that the subset belongs to 
// (in the list of orbit of subsets of size subset_size)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "generator::map_to_canonical_k_subset" << endl;
		}
	INT *our_set;
	INT *subset;
	INT *canonical_subset;
	INT *Elt1;
	INT i, j, k; //, idx;
	INT reduced_set_size;
	//INT f_implicit_fusion = TRUE;
	
	our_set = NEW_INT(set_size);
	subset = NEW_INT(set_size);
	canonical_subset = NEW_INT(set_size);
	Elt1 = NEW_INT(A->elt_size_in_INT);
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
	
	A->element_one(generator::transporter->ith(0), FALSE);


	// trace the subset:
	//f = first_oracle_node_at_level[subset_size];
	
	if (f_vv) {
		cout << "generator::map_to_canonical_k_subset "
				"before trace_set" << endl;
		}
	local_idx = trace_set(
		subset, set_size, subset_size,
		canonical_subset, Elt1, 
		verbose_level - 3);

	//idx = local_idx + f;

	if (f_vv) {
		cout << "generator::map_to_canonical_k_subset "
				"after trace_set local_idx=" << local_idx << endl;
		}
	if (FALSE) {
		cout << "the transporter is" << endl;
		A->element_print(Elt1, cout);
		cout << endl;
		}
	A->element_move(Elt1, transporter, FALSE);
	for (i = 0; i < reduced_set_size; i++) {
		reduced_set[i] = canonical_subset[subset_size + i];
		}
	if (FALSE) {
		cout << "generator::map_to_canonical_k_subset reduced set = ";
		INT_vec_print(cout, reduced_set, reduced_set_size);
		cout << endl;
		}
	FREE_INT(Elt1);
	FREE_INT(our_set);
	FREE_INT(subset);
	FREE_INT(canonical_subset);
	
	if (f_v) {
		cout << "generator::map_to_canonical_k_subset done" << endl;
		}
}

void generator::get_representative_of_subset_orbit(
	INT *set, INT size, INT local_orbit_no, 
	strong_generators *&Strong_gens, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT fst, node, sz;
	oracle *O;

	if (f_v) {
		cout << "generator::get_representative_of_subset_orbit "
				"verbose_level=" << verbose_level << endl;
		}
	fst = first_oracle_node_at_level[size];
	node = fst + local_orbit_no;
	if (f_vv) {
		cout << "generator::get_representative_of_subset_orbit "
				"before get_set" << endl;
		}
	get_set(node, set, sz);
	if (sz != size) {
		cout << "get_representative_of_subset_orbit: sz != size" << endl;
		exit(1);
		}
	O = root + node;
	if (f_vv) {
		cout << "generator::get_representative_of_subset_orbit "
				"before get_stabilizer_generators" << endl;
		}
	O->get_stabilizer_generators(this, Strong_gens, 0);
	if (f_v) {
		cout << "generator::get_representative_of_subset_orbit done" << endl;
		}
}

void generator::print_fusion_nodes(INT depth)
{
	INT i, f, l, j, h;

	for (i = 0; i <= depth; i++) {
		f = first_oracle_node_at_level[i];
		l = nb_orbits_at_level(i);
		for (j = 0; j < l; j++) {
			oracle *O;

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

void generator::find_interesting_k_subsets(
	INT *the_set, INT n, INT k,
	INT *&interesting_sets, INT &nb_interesting_sets,
	INT &orbit_idx,
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	classify *C;
	INT j, t, f, l, l_min, t_min = 0;

	if (f_v) {
		cout << "generator::find_interesting_k_subsets "
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
	interesting_sets = NEW_INT(l_min);
	nb_interesting_sets = l_min;
	for (j = 0; j < l_min; j++) {
		interesting_sets[j] = C->sorting_perm_inv[f + j];
		}
	orbit_idx = C->data_sorted[f];
	if (f_v) {
		cout << "generator::find_interesting_k_subsets "
				"l_min = " << l_min << " t_min = " << t_min
				<< " orbit_idx = " << orbit_idx << endl;
		}
	if (f_v) {
		cout << "interesting set of size " << nb_interesting_sets << " : ";
		INT_vec_print(cout, interesting_sets, nb_interesting_sets);
		cout << endl;
		}

	delete C;
	
	if (f_v) {
		cout << "generator::find_interesting_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
		}
}

void generator::classify_k_subsets(INT *the_set, INT n, INT k,
		classify *&C, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT nCk;
	INT *isotype;

	if (f_v) {
		cout << "generator::classify_k_subsets "
				"n = " << n << " k = " << k << endl;
		}
	
	trace_all_k_subsets(the_set, n, k, nCk, isotype, verbose_level);
	
	C = new classify;

	C->init(isotype, nCk, FALSE, 0);

	if (f_v) {
		cout << "generator::classify_k_subsets "
				"n = " << n << " k = " << k << " done" << endl;
		}
}

void generator::trace_all_k_subsets(INT *the_set,
		INT n, INT k, INT &nCk, INT *&isotype,
		INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *index_set;
	INT *subset;
	INT *canonical_subset;
	INT *Elt;
	INT subset_rk, local_idx, i;
	//INT f_implicit_fusion = TRUE;

	nCk = INT_n_choose_k(n, k);
	if (f_v) {
		cout << "generator::trace_all_k_subsets "
				"n = " << n << " k = " << k
				<< " nCk = " << nCk << endl;
		}

	Elt = NEW_INT(A->elt_size_in_INT);

	index_set = NEW_INT(k);
	subset = NEW_INT(k);
	canonical_subset = NEW_INT(k);
	isotype = NEW_INT(nCk);
	
	INT_vec_zero(isotype, nCk);

	first_k_subset(index_set, n, k);
	subset_rk = 0;

	while (TRUE) {
		if (f_vv && ((subset_rk % 100) == 0)) {
			cout << "generator::trace_all_k_subsets k=" << k 
				<< " testing set " << subset_rk << " / " << nCk 
				<< " = " << 100. * (double) subset_rk /
				(double) nCk << " % : ";
			INT_vec_print(cout, index_set, k);
			cout << endl;
			}
		for (i = 0; i < k; i++) {
			subset[i] = the_set[index_set[i]];
			}
		INT_vec_copy(subset, set[0], k);

		if (FALSE /*f_v2*/) {
			cout << "generator::trace_all_k_subsets "
					"corresponding to set ";
			INT_vec_print(cout, subset, k);
			cout << endl;
			}
		A->element_one(transporter->ith(0), 0);
		
		if (k == 0) {
			isotype[0] = 0;
			}
		else {

			if (FALSE) {
				cout << "generator::trace_all_k_subsets "
						"before trace_set" << endl;
				}
			local_idx = trace_set(subset, k, k, 
				canonical_subset, Elt, 
				//f_implicit_fusion, 
				0 /*verbose_level - 3*/);
			if (FALSE) {
				cout << "generator::trace_all_k_subsets "
						"after trace_set, local_idx = "
						<< local_idx << endl;
				}
			
			if (FALSE /*f_vvv*/) {
				cout << "generator::trace_all_k_subsets "
						"local_idx=" << local_idx << endl;
				}
			isotype[subset_rk] = local_idx;
			if (FALSE) {
				cout << "generator::trace_all_k_subsets "
						"the transporter is" << endl;
				A->element_print(Elt, cout);
				cout << endl;
				}

			}
		subset_rk++;
		if (!next_k_subset(index_set, n, k)) {
			break;
			}
		}
	if (subset_rk != nCk) {
		cout << "generator::trace_all_k_subsets "
				"subset_rk != nCk" << endl;
		exit(1);
		}


	FREE_INT(index_set);
	FREE_INT(subset);
	FREE_INT(canonical_subset);
	FREE_INT(Elt);
	if (f_v) {
		cout << "generator::trace_all_k_subsets done" << endl;
		}
}

void generator::get_orbit_representatives(INT level,
		INT &nb_orbits, INT *&Orbit_reps, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;

	if (f_v) {
		cout << "generator::get_orbit_representatives" << endl;
		}
	nb_orbits = nb_orbits_at_level(level);
	if (f_v) {
		cout << "orbits_on_k_sets: we found " << nb_orbits
				<< " orbits on " << level << "-sets" << endl;
		}
	Orbit_reps = NEW_INT(nb_orbits * level);
	for (i = 0; i < nb_orbits; i++) {
		get_set_by_level(level, i, Orbit_reps + i * level);
		}
	
	if (f_v) {
		cout << "generator::get_orbit_representatives done" << endl;
		}
}

void generator::unrank_point(INT *v, INT rk)
{
	(*unrank_point_func)(v, rk, rank_point_data);
}

INT generator::rank_point(INT *v)
{
	INT rk;

	rk = (*rank_point_func)(v, rank_point_data);
	return rk;
}

void generator::unrank_basis(INT *Basis, INT *S, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		(*unrank_point_func)(Basis + i * vector_space_dimension,
				S[i], rank_point_data);
	}
}

void generator::rank_basis(INT *Basis, INT *S, INT len)
{
	INT i;

	for (i = 0; i < len; i++) {
		S[i] = (*rank_point_func)(Basis + i * vector_space_dimension,
				rank_point_data);
	}
}


void generator::generate_source_code(INT level, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	BYTE fname[1000];
	BYTE my_prefix[1000];
	INT iso_type;
	INT *rep;
	INT i, j;
	INT /*f,*/ nb_iso;
	INT *set;
	longinteger_object go;

	if (f_v) {
		cout << "generator::generate_source_code" << endl;
		}
	sprintf(my_prefix, "%s_level_%ld", prefix, level);
	sprintf(fname, "%s.C", my_prefix);

	set = NEW_INT(level);
	//f = first_oracle_node_at_level[level];
	nb_iso = nb_orbits_at_level(level);



		
	{
	ofstream fp(fname);

	fp << "INT " << prefix << "_nb_reps = " << nb_iso << ";" << endl;
	fp << "INT " << prefix << "_size = " << level << ";" << endl;
	fp << "INT " << prefix << "_reps[] = {" << endl;
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		get_set_by_level(level, iso_type, set);
		rep = set;
		fp << "\t";
		for (i = 0; i < level; i++) {
			fp << rep[i];
			fp << ", ";
			}
		fp << endl;
		}
	fp << "};" << endl;
	fp << "const BYTE *" << prefix << "_stab_order[] = {" << endl;
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		//rep = The_surface[iso_type]->coeff;

		set_and_stabilizer *SaS;

		SaS = get_set_and_stabilizer(level, iso_type,
				0 /* verbose_level */);
		fp << "\t\"";
		
		SaS->target_go.print_not_scientific(fp);
		fp << "\"," << endl;

		delete SaS;
		}
	fp << "};" << endl;


	fp << "INT " << prefix << "_make_element_size = "
			<< A->make_element_size << ";" << endl;
	
	{
	INT *stab_gens_first;
	INT *stab_gens_len;
	INT fst;

	stab_gens_first = NEW_INT(nb_iso);
	stab_gens_len = NEW_INT(nb_iso);
	fst = 0;
	fp << "INT " << prefix << "_stab_gens[] = {" << endl;
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {

		
		set_and_stabilizer *SaS;


		SaS = get_set_and_stabilizer(level,
				iso_type, 0 /* verbose_level */);

		stab_gens_first[iso_type] = fst;
		stab_gens_len[iso_type] = SaS->Strong_gens->gens->len;
		fst += stab_gens_len[iso_type];

		
		for (j = 0; j < stab_gens_len[iso_type]; j++) {
			if (f_vv) {
				cout << "generator::generate_source_code "
						"before extract_strong_generators_in_order generator "
						<< j << " / " << stab_gens_len[iso_type] << endl;
				}
			fp << "\t";
			A->element_print_for_make_element(
					SaS->Strong_gens->gens->ith(j), fp);
			fp << endl;
			}

		delete SaS;

		}
	fp << "};" << endl;


	fp << "INT " << prefix << "_stab_gens_fst[] = { ";
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		fp << stab_gens_first[iso_type];
		if (iso_type < nb_iso - 1) {
			fp << ", ";
			}
		if (((iso_type + 1) % 10) == 0) {
			fp << endl << "\t";
			}
		}
	fp << "};" << endl;
	
	fp << "INT " << prefix << "_stab_gens_len[] = { ";
	for (iso_type = 0; iso_type < nb_iso; iso_type++) {
		fp << stab_gens_len[iso_type];
		if (iso_type < nb_iso - 1) {
			fp << ", ";
			}
		if (((iso_type + 1) % 10) == 0) {
			fp << endl << "\t";
			}
		}
	fp << "};" << endl;



	
	FREE_INT(stab_gens_first);
	FREE_INT(stab_gens_len);
	}
	}

	cout << "written file " << fname << " of size "
			<< file_size(fname) << endl;
	if (f_v) {
		cout << "generator::generate_source_code done" << endl;
		}
}




