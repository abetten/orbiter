/*
 * sims_main.cpp
 *
 *  Created on: Sep 1, 2019
 *      Author: betten
 */





#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {



void sims::compute_base_orbits(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "sims::compute_base_orbits" << endl;
	}
	if (f_vv) {
		cout << "sims::compute_base_orbits "
				"base_len=" << A->base_len() << endl;
	}
	for (i = A->base_len() - 1; i >= 0; i--) {
		if (FALSE) {
			cout << "sims::compute_base_orbits "
					"level " << i << endl;
		}
		compute_base_orbit(i, 0/*verbose_level - 1*/);
		if (f_vv) {
			cout << "sims::compute_base_orbits level " << i
				<< " base point " << A->base_i(i)
				<< " orbit length " << orbit_len[i] << endl;
		}
	}
	if (f_v) {
		cout << "sims::compute_base_orbits done, orbit_len=";
		Int_vec_print(cout, orbit_len, A->base_len());
		cout << endl;
	}
}

void sims::compute_base_orbits_known_length(int *tl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "sims::compute_base_orbits_known_length: ";
		Int_vec_print(cout, tl, A->base_len());
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
	}
	for (i = A->base_len() - 1; i >= 0; i--) {
		if (f_v) {
			cout << "sims::compute_base_orbits_known_length "
					"computing level " << i << endl;
		}
		compute_base_orbit_known_length(i, tl[i], verbose_level);
		if (f_v) {
			cout << "sims::compute_base_orbits_known_length "
					"level " << i
				<< " base point " << A->base_i(i)
				<< " orbit length " << orbit_len[i]
				<< " has been computed" << endl;
		}
		if (orbit_len[i] != tl[i]) {
			cout << "sims::compute_base_orbits_known_length "
					"orbit_len[i] != tl[i]" << endl;
			cout << "orbit_len[i]=" << orbit_len[i] << endl;
			cout << "tl[i]=" << tl[i] << endl;
			print_generators_at_level_or_below(i);
			exit(1);
		}
		}
	if (f_v) {
		cout << "sims::compute_base_orbits_known_length done" << endl;
	}
}

void sims::extend_base_orbit(int new_gen_idx, int lvl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i, cur, cur_pt, total, total0;
	int next_pt, next_pt_loc, gen_idx, nbg;

	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << " verbose_level = " << verbose_level << endl;
	}
	cur = 0;
	total = total0 = orbit_len[lvl];
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << " orbit_len[lvl] = " << orbit_len[lvl] << endl;
	}
	nbg = nb_gen[lvl];
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << " nbg = " << nbg << endl;
	}
	if (orbit[lvl] == NULL) {
		cout << "sims::extend_base_orbit orbit[lvl] == NULL" << endl;
		exit(1);
	}
	while (cur < total) {
		cur_pt = orbit[lvl][cur];
		if (f_vvv) {
			cout << "sims::extend_base_orbit: cur=" << cur << " total = " << total << " cur_pt=" << cur_pt << endl;
		}
		for (i = 0; i < nbg; i++) {
			if (f_vvv) {
				cout << "sims::extend_base_orbit: applying generator " << i << " / " << nbg << endl;
			}
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			if (f_vvv) {
				cout << "sims::extend_base_orbit: next_pt = " << next_pt << endl;
			}
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (f_vvv) {
				cout << "sims::extend_base_orbit: next_pt_loc = " << next_pt_loc << endl;
			}
			if (f_vvv) {
				cout << "sims::extend_base_orbit "
						"generator " << gen_idx << " maps "
						<< cur_pt << " to " << next_pt << endl;
			}
			if (next_pt_loc < total) {
				continue;
			}
			if (f_vvv) {
				cout << "sims::extend_base_orbit "
						"additional pt " << next_pt << " reached from "
						<< cur_pt << " under generator "
						<< i << endl;
			}
			swap_points(lvl, total, next_pt_loc);
			prev[lvl][total] = cur_pt;
			label[lvl][total] = gen_idx;
			total++;
			if (f_vvv) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				//print_orbit(cur, total - 1);
			}
		}
		cur++;
	}
	orbit_len[lvl] = total;
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << " finished" << endl;
		cout << lvl << "-th base point " << A->base_i(lvl)
			<< " orbit extended to length " << orbit_len[lvl];
		if (FALSE) {
			cout << " { ";
			for (i = 0; i < orbit_len[lvl]; i++) {
				cout << orbit[lvl][i];
				if (i < orbit_len[lvl] - 1)
					cout << ", ";
			}
			cout << " }" << endl;
		}
		else {
			cout << endl;
		}
	}
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << " done" << endl;
	}
}

void sims::compute_base_orbit(int lvl, int verbose_level)
// applies all generators at the given level to compute
// the corresponding basic orbit.
// the generators are the first nb_gen[lvl] in the generator arry
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int pt, pt_loc, cur, cur_pt, i, next_pt, next_pt_loc, gen_idx;

	pt = A->base_i(lvl);
	pt_loc = orbit_inv[lvl][pt];
	if (f_v) {
		cout << "sims::compute_base_orbit: "
				"computing orbit of " << lvl << "-th base point "
				<< pt << " with " << nb_gen[lvl] << " generators" << endl;
	}
	if (pt_loc > 0) {
		swap_points(lvl, 0, pt_loc);
	}
	cur = 0;
	orbit_len[lvl] = 1;
	while (cur < orbit_len[lvl]) {
		cur_pt = orbit[lvl][cur];
		if (FALSE) {
			cout << "sims::compute_base_orbit "
					"applying generator to " << cur_pt << endl;
		}
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (FALSE) {
				cout << "sims::compute_base_orbit "
						"generator " << i << " maps " << cur_pt
						<< " to " << next_pt << endl;
			}
			if (next_pt_loc < orbit_len[lvl]) {
				continue;
			}
			if (FALSE) {
				cout << "additional pt " << next_pt << " reached from "
						<< cur_pt << " under generator "
						<< i << endl;
			}
			swap_points(lvl, orbit_len[lvl], next_pt_loc);
			prev[lvl][orbit_len[lvl]] = cur_pt;
			label[lvl][orbit_len[lvl]] = gen_idx;
			orbit_len[lvl]++;
			if (FALSE) {
				cout << "sims::compute_base_orbit "
						"cur = " << cur << endl;
				cout << "sims::compute_base_orbit "
						"orbit_len[lvl] = " << orbit_len[lvl] << endl;
				//print_orbit(cur, total - 1);
			}
		}
		cur++;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit finished, "
			<< lvl << "-th base orbit of length "
			<< orbit_len[lvl] << endl;
	}
	if (FALSE) {
		cout << "{ ";
		for (i = 0; i < orbit_len[lvl]; i++) {
			cout << orbit[lvl][i];
			if (i < orbit_len[lvl] - 1)
				cout << ", ";
		}
		cout << " }" << endl;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit done" << endl;
	}
}

void sims::compute_base_orbit_known_length(int lvl,
		int target_length, int verbose_level)
// applies all generators at the given level to compute
// the corresponding basic orbit.
// the generators are the first nb_gen[lvl] in the generator arry
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE;
	//int f_vvv = (verbose_level >= 3);
	//int f_v10 = FALSE; // (verbose_level >= 10);
	int pt, pt_loc, cur, cur_pt, i, next_pt, next_pt_loc, gen_idx;
	double progress;

	pt = A->base_i(lvl);
	pt_loc = orbit_inv[lvl][pt];
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length: "
				"computing orbit of " << lvl
			<< "-th base point " << pt
			<< " target_length = " << target_length
			<< " nb_gens=" << nb_gen[lvl] << endl;
	}
	if (target_length > 1000000) {
		f_vv = TRUE;
	}
	if (FALSE) {
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			cout << "sims::compute_base_orbit_known_length "
					"generator " << i << ":" << endl;
			A->element_print_quick(gens.ith(gen_idx), cout);
		}
	}
	if (pt_loc > 0) {
		swap_points(lvl, 0, pt_loc);
	}
	cur = 0;
	orbit_len[lvl] = 1;
	while (cur < orbit_len[lvl] && orbit_len[lvl] < target_length) {
		cur_pt = orbit[lvl][cur];
		if (f_vv) {
			if (target_length) {
				progress = (double) cur / (double) target_length;
			}
			else {
				progress = 0.;
			}
			if (cur % ((1 << 21) - 1) == 0) {
				cout << "sims::compute_base_orbit_known_length "
						"lvl=" << lvl << " cur=" << cur
						<< " orbit_len[lvl]=" << orbit_len[lvl]
						<< " target_length=" << target_length
						<< " progress=" << progress * 100 << "%" << endl;
			}
		}
		if (FALSE) {
			cout << "sims::compute_base_orbit_known_length "
					"applying " << nb_gen[lvl] << " generators to "
					<< cur_pt << " orbit_len[lvl]=" << orbit_len[lvl]
					<< " target_length=" << target_length << endl;
		}
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length "
						"generator " << i << " maps " << cur_pt
						<< " to " << next_pt << endl;
			}
			if (next_pt_loc < orbit_len[lvl]) {
				continue;
			}
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length "
						"additional pt " << next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
			}
			swap_points(lvl, orbit_len[lvl], next_pt_loc);
			prev[lvl][orbit_len[lvl]] = cur_pt;
			label[lvl][orbit_len[lvl]] = gen_idx;
			orbit_len[lvl]++;
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length "
						"cur = " << cur << endl;
				cout << "sims::compute_base_orbit_known_length "
						"orbit_len[lvl] = " << orbit_len[lvl] << endl;
				//print_orbit(cur, total - 1);
			}
		}
		cur++;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length finished, "
				<< lvl << "-th base orbit of length "
				<< orbit_len[lvl] << endl;
	}
	if (FALSE) {
		cout << "{ ";
		for (i = 0; i < orbit_len[lvl]; i++) {
			cout << orbit[lvl][i];
			if (i < orbit_len[lvl] - 1) {
				cout << ", ";
			}
		}
		cout << " }" << endl;
	}
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length done" << endl;
	}
}

int sims::strip_and_add(int *elt, int *residue, int verbose_level)
// returns TRUE if something was added,
// FALSE if element stripped through
{
	int drop_out_level, image;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "sims::strip_and_add" << endl;
	}
	if (strip(elt, residue, drop_out_level,
			image, 0 /*verbose_level*/)) {
		if (f_v) {
			cout << "sims::strip_and_add element strips to the identity, finished" << endl;
		}
		return FALSE;
	}
	if (f_v) {
		cout << "sims::strip_and_add after strip, drop_out_level = "
				<< drop_out_level << " image = " << image << endl;
	}
	if (FALSE) {
		cout << "sims::strip_and_add residue = " << endl;
		A->element_print_quick(residue, cout);
		//A->element_print_as_permutation(residue, cout);
		cout << endl;
	}

	if (f_v) {
		cout << "sims::strip_and_add calling add_generator_at_level "
				<< drop_out_level << endl;
	}
	add_generator_at_level(residue, drop_out_level, verbose_level);
	//add_generator_at_level_only(residue, drop_out_level, verbose_level);
	// !!! this was add_generator_at_level previously

	if (FALSE) {
		cout << "sims::strip_and_add increased set of generators:" << endl;
		gens.print(cout);
		gens.print_as_permutation(cout);
	}
	if (f_v) {
		cout << "sims::strip_and_add finished, final group order is ";
		print_group_order(cout);
		cout << endl;
	}
	if (f_v) {
		cout << "sims::strip_and_add done" << endl;
	}
	return TRUE;
}

int sims::strip(int *elt, int *residue,
		int &drop_out_level, int &image, int verbose_level)
// returns TRUE if the element sifts through
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, bi, j, j_coset;

	if (f_v) {
		cout << "sims::strip" << endl;
		cout << "my_base_len=" << my_base_len << endl;
	}
	if (A == NULL) {
		cout << "sims::strip A==NULL" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "sims::strip A=" << A->label << endl;
		cout << "A->base_len=" << A->base_len() << endl;
	}
	if (f_vv) {
		A->element_print_quick(elt, cout);
		cout << endl;
	}
	A->element_move(elt, strip1, FALSE);
	for (i = 0; i < my_base_len; i++) {
		if (f_v) {
			cout << "sims::strip level " << i << " / " << my_base_len << endl;
			//A->element_print(strip1, cout);
			//cout << endl;
		}
		bi = A->base_i(i);
		if (f_vv) {
			cout << "computing image of " << i
					<< "-th base element " << bi << endl;
		}
		j = A->element_image_of(bi, strip1, verbose_level - 2);
		if (f_v) {
			cout << "sims::strip level " << i
					<< " base point " << bi
					<< " gets mapped to " << j << endl;
		}
		if (f_v) {
			cout << "sims::strip level " << i << " / " << A->base_len() << " before get_orbit_inv j=" << j << endl;
		}
		j_coset = get_orbit_inv(i, j);
		if (f_v) {
			cout << "sims::strip j_coset " << j_coset << endl;
		}
		if (j_coset >= orbit_len[i]) {
			if (f_v) {
				cout << "sims::strip not in the orbit, "
						"dropping out" << endl;
			}
			image = j;
			drop_out_level = i;
			A->element_move(strip1, residue, FALSE);
			if (f_v) {
				cout << "sims::strip returns FALSE, "
						"drop_out_level=" << drop_out_level << endl;
			}
			return FALSE;
		}
		else {
			if (f_v) {
				cout << "sims::strip computing representative "
						"of coset " << j_coset << endl;
			}
			coset_rep_inv(eltrk3, i, j_coset, verbose_level);
			if (FALSE) {
				cout << "sims::strip representative "
						"of coset " << j_coset << " is " << endl;
				A->element_print(eltrk3, cout);
				cout << endl;
			}
			if (FALSE) {
				cout << "sims::strip before element_mult, "
						"strip1=" << endl;
				A->element_print(strip1, cout);
				cout << endl;
			}
			if (FALSE) {
				cout << "sims::strip before element_mult, "
						"cosetrep=" << endl;
				A->element_print(eltrk3, cout);
				cout << endl;
			}
			A->element_mult(strip1, eltrk3, strip2, 0 /*verboe_level*/);
			if (FALSE) {
				cout << "sims::strip before element_move" << endl;
			}
			A->element_move(strip2, strip1, FALSE);
			if (FALSE) {
				cout << "sims::strip after dividing off, "
						"we have strip1= " << endl;
				A->element_print(strip1, cout);
				cout << endl;
			}
		}
	}
	if (f_v) {
		cout << "sims::strip after loop" << endl;
	}
	A->element_move(strip1, residue, FALSE);
	if (f_v) {
		cout << "sims::strip returns TRUE" << endl;
	}
	return TRUE;
}

void sims::add_generator_at_level(int *elt,
		int lvl, int verbose_level)
// add the generator to the array of generators and then extends the
// basic orbits 0,..,lvl using extend_base_orbit
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "sims::add_generator_at_level adding generator at "
				"level " << lvl << " verbose_level = " << verbose_level<< endl;
		print_generator_depth_and_perm();
		if (FALSE) {
			A->element_print_quick(elt, cout);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "sims::add_generator_at_level before add_generator" << endl;
	}
	add_generator(elt, verbose_level);
	if (f_v) {
		cout << "sims::add_generator_at_level after add_generator" << endl;
		print_generator_depth_and_perm();
	}
	for (i = lvl; i >= 0; i--) {
		if (f_v) {
			cout << "sims::add_generator_at_level "
				<< lvl << " calling extend_base_orbit " << i << endl;
		}
		extend_base_orbit(gens.len - 1, i, verbose_level - 1);
	}
	if (f_v) {
		cout << "sims::add_generator_at_level done" << endl;
	}
}

void sims::add_generator_at_level_only(int *elt,
		int lvl, int verbose_level)
// add the generator to the array of generators and then extends the
// basic orbit lvl using extend_base_orbit
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "sims::add_generator_at_level_only level " << lvl << endl;
		if (f_vvv) {
			A->element_print(elt, cout);
			cout << endl;
		}
	}
	add_generator(elt, verbose_level);
	if (f_vvv) {
		print_generator_depth_and_perm();
	}
	extend_base_orbit(gens.len - 1, lvl, verbose_level - 1);
	if (f_v) {
		cout << "sims::add_generator_at_level_only level " << lvl << " done" << endl;
	}
}

void sims::build_up_group_random_process_no_kernel(
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	ring_theory::longinteger_object go, go1;
	sims K;

	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel" << endl;
	}
	old_G->group_order(go);
	if (f_v) {
		cout << "target group order = " << go << endl;
	}
	K.init(A, verbose_level - 2);
	K.init_trivial_group(verbose_level - 1);
	K.group_order(go1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel "
				"kernel group order " << go1 << endl;
	}
	init_trivial_group(verbose_level - 1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel "
				"before build_up_group_random_process" << endl;
	}
	build_up_group_random_process(&K, old_G, go,
		FALSE /* f_override_chose_next_base_point */,
		NULL /* choose_next_base_point_method */,
		verbose_level - 1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel "
				"after build_up_group_random_process" << endl;
	}
}

void sims::extend_group_random_process_no_kernel(
		sims *extending_by_G, ring_theory::longinteger_object &target_go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//longinteger_object go, go1, go2;
	//longinteger_domain D;
	sims K;

	if (f_v) {
		cout << "sims::extend_group_random_process_no_kernel" << endl;
	}
	//group_order(go);
	//extending_by_G->group_order(go1);
	//D.mult(go, go1, go2);
	if (f_v) {
		cout << "target group order = " << target_go << endl;
	}

	K.init(A, verbose_level - 2);
	K.init_trivial_group(verbose_level - 1);
	build_up_group_random_process(
		&K,
		extending_by_G,
		target_go,
		FALSE /* f_override_chose_next_base_point */,
		NULL /* choose_next_base_point_method */,
		verbose_level + 3);
	if (f_v) {
		cout << "sims::extend_group_random_process_no_kernel done" << endl;
	}
}

void sims::build_up_group_random_process(sims *K,
	sims *old_G,
	ring_theory::longinteger_object &target_go,
	int f_override_choose_next_base_point,
	int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	//int f_vvv = (verbose_level >= 1);
	//int f_vvvv = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 6);
	int f_v4 = (verbose_level >= 7);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object go, G_order, K_order, KG_order, quo, rem;
	int drop_out_level, image, cnt, b, c, old_base_len;
	actions::action *GA;
	actions::action *KA;
	int *Elt;

	if (f_v) {
		cout << "sims::build_up_group_random_process verbose_level=" << verbose_level << endl;
	}
	GA = A;
	KA = K->A;
	Elt = NEW_int(A->elt_size_in_int);

	group_order(G_order);
	K->group_order(K_order);
	D.mult(G_order, K_order, KG_order);
	if (f_v) {
		cout << "sims::build_up_group_random_process: "
				"current group order is " << G_order
				<< " target " << target_go << endl;
		cout << "the old_G action " << old_G->A->label
				<< " has base_length = " << old_G->A->base_len()
			<< " and degree " << old_G->A->degree << endl;
		cout << "the kernel action " << KA->label
				<< " has base_length = " << KA->base_len()
			<< " and degree " << KA->degree << endl;
		cout << "the image action has base_length = " << GA->base_len()
			<< " and degree " << GA->degree << endl;
		cout << "current action " << GA->label << endl;
		cout << "current group order = " << G_order << endl;
		cout << "current kernel order = " << K_order << endl;
		cout << "together = " << KG_order << endl;
		cout << "target_go = " << target_go << endl;
	}
	cnt = 0;
	while (TRUE) {

		if (f_vv) {
			cout << "sims::build_up_group_random_process "
					"iteration " << cnt << endl;
		}
		if (cnt > 1000) {
			cout << "sims::build_up_group_random_process "
					"cnt > 1000, something seems to be wrong" << endl;
			test_if_subgroup(old_G, 2);
			exit(1);
		}
		if (f_v4) {
			old_G->A->print_base();
			old_G->print_orbit_len();
		}
		if ((cnt % 2) == 0) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"choosing random schreier generator" << endl;
			}
			random_schreier_generator(Elt, verbose_level - 5);
			A->element_move(Elt, GA->Elt1, 0);
			if (f_v4) {
				cout << "sims::build_up_group_random_process: "
						"random element chosen:" << endl;
				A->element_print_quick(GA->Elt1, cout);
				cout << endl;
			}
		}
		else if ((cnt % 2) == 1) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"choosing random element in the group by "
						"which we extend" << endl;
			}
			old_G->random_element(GA->Elt1, verbose_level - 5);
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"random element chosen, path = ";
				Int_vec_print(cout, old_G->path, old_G->A->base_len());
				cout << endl;
			}
			if (f_v4) {
				GA->element_print_quick(GA->Elt1, cout);
				cout << endl;
			}
		}
		if (f_v4) {
			cout << "sims::build_up_group_random_process: "
					"calling strip:" << endl;
		}
		if (strip(GA->Elt1, GA->Elt2, drop_out_level, image,
				verbose_level - 5)) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"element strips through" << endl;
				if (f_v4) {
					cout << "sims::build_up_group_random_process: "
							"residue = " << endl;
					GA->element_print_quick(GA->Elt2, cout);
					cout << endl;
				}
			}
			//f_added = FALSE;
			if (!GA->element_is_one(GA->Elt2, 0)) {
				if (f_vvv) {
					cout << "sims::build_up_group_random_process: "
							"the residue is not trivial, we need to "
							"choose another base point" << endl;
				}
				if (f_override_choose_next_base_point) {
					b = (*choose_next_base_point_method)(GA,
							GA->Elt2, verbose_level - 5);
				}
				else {
					b = GA->choose_next_base_point_default_method(
							GA->Elt2, verbose_level - 5);
				}

				if (f_vv) {
					cout << "sims::build_up_group_random_process: "
							"suggested next base point " << b << endl;
				}
				if (b == -1) {
					if (f_vv) {
						cout << "sims::build_up_group_random_process: "
								"cannot find next base point" << endl;
					}
					if (K->strip(GA->Elt2, GA->Elt3,
							drop_out_level, image, 0/*verbose_level - 3*/)) {
						if (f_vv) {
							cout << "sims::build_up_group_random_process: "
									"element strips through kernel" << endl;
							if (f_v4) {
								cout << "sims::build_up_group_random_"
										"process: residue = " << endl;
								KA->element_print_quick(GA->Elt3, cout);
								cout << endl;
								K->print(FALSE);
								K->print_basic_orbits();
								cout << "sims::build_up_group_random_"
										"process: residue" << endl;
								KA->element_print_image_of_set(
										GA->Elt3, KA->base_len(), KA->get_base());
								cout << "sims::build_up_group_random_"
										"process: Elt2" << endl;
								KA->element_print_image_of_set(
										GA->Elt2, KA->base_len(), KA->get_base());
							}
						}
						if (!KA->element_is_one(GA->Elt3, FALSE)) {
							cout << "sims::build_up_group_random_process: "
									"element strips through kernel, "
									"residue = " << endl;
							cout << "but the element is not the identity, "
									"something is wrong" << endl;
							GA->element_print(GA->Elt3, cout);
							cout << endl;

							cout << "sims::build_up_group_random_process: "
									"current group order is " << G_order
									<< " target " << target_go << endl;
							cout << "the old_G action " << old_G->A->label
									<< " has base_length = "
									<< old_G->A->base_len()
								<< " and degree " << old_G->A->degree << endl;
							cout << "the kernel action " << KA->label
									<< " has base_length = " << KA->base_len()
								<< " and degree " << KA->degree << endl;
							cout << "the image action has base_length = "
								<< GA->base_len()
								<< " and degree " << GA->degree << endl;
							cout << "current action " << GA->label << endl;
							cout << "current group order = "
								<< G_order << endl;
							cout << "current kernel order = "
								<< K_order << endl;
							cout << "together = " << KG_order << endl;
							cout << "target_go = " << target_go << endl;

							exit(1);
						}
					}
					else {
						if (f_vv) {
							cout << "sims::build_up_group_random_process before K->add_generator_at_level drop_out_level=" << drop_out_level << endl;
						}
						K->add_generator_at_level(GA->Elt3,
								drop_out_level, verbose_level - 3);
						if (f_vvv) {
							cout << "sims::build_up_group_random_process: "
									"the residue has been added as kernel "
									"generator at level " << drop_out_level
									<< endl;
						}
					}
					//f_added = TRUE;
				}
				else {
					if (f_vvv) {
						cout << "sims::build_up_group_random_process: "
								"choosing additional base point " << b << endl;
					}
					old_base_len = GA->base_len();
					GA->Stabilizer_chain->reallocate_base(b);
					if (f_v) {
						cout << "sims::build_up_group_random_process before reallocate_base" << endl;
					}
					reallocate_base(old_base_len, verbose_level - 1);
					if (f_v) {
						cout << "sims::build_up_group_random_process after reallocate_base" << endl;
					}
					if (f_vv) {
						cout << "sims::build_up_group_random_process: "
								"additional base point " << b
							<< " chosen, increased base has length "
							<< GA->base_len() << endl;
						cout << "sims::build_up_group_random_process: "
								"calling add_generator_at_level" << endl;
					}
					if (f_v) {
						cout << "sims::build_up_group_random_process before add_generator_at_level" << endl;
					}
					add_generator_at_level(GA->Elt2,
							GA->base_len() - 1, verbose_level - 3);
					if (f_v) {
						cout << "sims::build_up_group_random_process after add_generator_at_level" << endl;
					}
					if (f_vv) {
						cout << "sims::build_up_group_random_process: "
								"the residue has been added at level "
								<< GA->base_len() - 1 << endl;
					}
				} // if b
			} // if ! element is one
			else {
				if (f_vv) {
					cout << "sims::build_up_group_random_process: "
							"the residue is trivial" << endl;
				}
			}
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"before closure_group" << endl;
			}
			//closure_group(10, verbose_level);
			closure_group(10, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"after closure_group" << endl;
			}
		}
		else {
			//f_added = TRUE;
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"element needs to be inserted at level = "
					<< drop_out_level << " with image "
					<< image << endl;
				if (FALSE) {
					GA->element_print(GA->Elt2, cout);
					cout  << endl;
				}
			}
			if (f_vv) {
				cout << "sims::build_up_group_random_process before add_generator_at_level" << endl;
			}
			add_generator_at_level(GA->Elt2, drop_out_level,
					0/*verbose_level - 3*/);
			if (f_vv) {
				cout << "sims::build_up_group_random_process after add_generator_at_level" << endl;
			}
		}

		if (f_vv) {
			cout << "sims::build_up_group_random_process: "
					"computing group order G" << endl;
		}
		group_order(G_order);
		if (f_vv) {
			cout << "sims::build_up_group_random_process:  "
					"G_order=" << G_order << endl;
		}
		K->group_order(K_order);
		if (f_vv) {
			cout << "sims::build_up_group_random_process:  "
					"K_order=" << K_order << endl;
		}
		//cout << "K tl: ";
		//int_vec_print(cout, K->orbit_len, K->A->base_len);
		//cout << endl;
		//cout << "K action " << K->A->label << endl;
		D.mult(G_order, K_order, KG_order);
		if (f_v /* (f_v && f_added) || f_vv */) {
			cout << "sims::build_up_group_random_process: "
					"current group order is " << KG_order
				<< " = " << G_order << " * " << K_order << endl;
		}
		if (f_vv) {
			print_transversal_lengths();
		}
		if (FALSE) {
			cout << "sims::build_up_group_random_process "
					"before D.compare" << endl;
		}
		c = D.compare(target_go, KG_order);
		if (FALSE) {
			cout << "sims::build_up_group_random_process "
					"after D.compare c=" << c
					<< " cnt=" << cnt << endl;
		}
		cnt++;
		if (c == 0) {
			if (f_v) {
				cout << "sims::build_up_group_random_process: "
						"reached the full group after "
						<< cnt << " iterations" << endl;
			}
			break;
		}
		if (c < 0) {
			if (TRUE) {
				cout << "sims::build_up_group_random_process "
						"overshooting the expected group after "
						<< cnt << " iterations" << endl;
				cout << "current group order is " << KG_order
					<< " = |G| * |K| = " << G_order << " * " << K_order << ", target_go=" << target_go << endl;
			}
			//break;
			exit(1);
		}
	} // while TRUE
	FREE_int(Elt);
	if (f_vv) {
		cout << "sims::build_up_group_random_process finished: "
				"found a group of order " << KG_order
			<< " = " << G_order << " * " << K_order << endl;
		if (f_vvv) {
			cout << "the n e w action has base_length = "
				<< GA->base_len()
				<< " and degree " << GA->degree << endl;
			print_transversal_lengths();
			if (FALSE) {
				print_transversals();
			}
			if (FALSE) {
				print(FALSE);
			}
		}
	}
	if (f_v) {
		cout << "sims::build_up_group_random_process done" << endl;
	}
}

void sims::build_up_group_from_generators(sims *K,
		data_structures_groups::vector_ge *gens,
	int f_target_go, ring_theory::longinteger_object *target_go,
	int f_override_choose_next_base_point,
	int (*choose_next_base_point_method)(actions::action *A,
			int *Elt, int verbose_level),
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	ring_theory::longinteger_domain D;
	ring_theory::longinteger_object G_order, K_order, KG_order;
	int drop_out_level, image, f_added, j;
	int level, base_point, b, old_base_len;
	actions::action *GA;
	actions::action *KA;
	data_structures_groups::vector_ge subset_of_gens;

	GA = A;
	KA = K->A;


	if (f_v) {
		cout << "sims::build_up_group_from_generators base: ";
		Lint_vec_print(cout, GA->get_base(), GA->base_len());
		cout << endl;

#if 0
		cout << "generators:" << endl;
		gens->print(cout);
		cout << endl;
#endif

		if (f_target_go) {
			cout << "sims::build_up_group_from_generators target group order: " << *target_go << endl;
		}
		else {
			cout << "sims::build_up_group_from_generators no target group order given" << endl;
		}
		cout << "sims::build_up_group_from_generators verbose_level=" << verbose_level << endl;
	}
	group_order(G_order);
	K->group_order(K_order);
	D.mult(G_order, K_order, KG_order);
	for (level = GA->base_len() - 1; level >= 0; level--) {
		base_point = GA->base_i(level);
		if (f_vv) {
			cout << "sims::build_up_group_from_generators level " << level << " base point "
					<< base_point << endl;
		}
		GA->find_strong_generators_at_level(
			GA->base_len(), GA->get_base(), level,
			*gens, subset_of_gens, verbose_level - 3);

		{
		schreier O;

		if (f_v) {
			cout << "sims::build_up_group_from_generators calling O.init" << endl;
		}

		O.init(GA, verbose_level - 2);

		if (f_v) {
			cout << "sims::build_up_group_from_generators calling O.init_generators" << endl;
		}
		O.init_generators(subset_of_gens, verbose_level - 2);

		if (f_vvv) {
			cout << "sims::build_up_group_from_generators generators in schreier" << endl;
			O.print_generators();
		}

		if (f_vv) {
			cout << "sims::build_up_group_from_generators computing orbit of point " << base_point << endl;
		}
		O.compute_point_orbit(base_point, 0);
		if (f_vv) {
			cout << "sims::build_up_group_from_generators point " << base_point << " lies in an orbit "
					"of size " << O.orbit_len[0] << endl;
			if (FALSE) {
				O.print(cout);
				O.print_tables(cout, FALSE);
			}
		}
		for (j = 0; j < O.orbit_len[0]; j++) {
			if (FALSE) {
				cout << "sims::build_up_group_from_generators level " << level << " coset rep " << j << endl;
			}
			O.coset_rep(j, 0 /* verbose_level */);
			if (FALSE) {
				GA->element_print(O.cosetrep, cout);
				cout << endl;
			}
			if (strip(O.cosetrep, GA->Elt2 /* residue */,
					drop_out_level, image, 0 /*verbose_level - 1*/)) {
				if (f_vv) {
					cout << "sims::build_up_group_from_generators element strips through" << endl;
					if (FALSE /*f_vvv */) {
						cout << "sims::build_up_group_from_generators residue=" << endl;
						GA->element_print_quick(GA->Elt2, cout);
						cout << endl;
					}
				}
				if (FALSE) {
					cout << "sims::build_up_group_from_generators element strips through." << endl;
					cout << "if it is the identity element, that's OK,"
							<< endl;
					cout << "sims::build_up_group_from_generators otherwise please add another base point,"
							<< endl;
					cout << "sims::build_up_group_from_generators a point which is moved by the residue"
							<< endl;
					GA->element_print(GA->Elt2, cout);
				}
				if (!GA->element_is_one(GA->Elt2, FALSE)) {
					if (f_vvv) {
						cout << "sims::build_up_group_from_generators the residue is not trivial, "
								"we need to chose another base point"
								<< endl;
					}
					if (f_override_choose_next_base_point) {
						b = (*choose_next_base_point_method)(
								GA, GA->Elt2, verbose_level);
					}
					else {
						b = GA->choose_next_base_point_default_method(
								GA->Elt2, verbose_level);
					}
					if (b == -1) {
						if (f_vv) {
							cout << "sims::build_up_group_from_generators: "
									"cannot find next base point" << endl;
						}
						if (K->strip(GA->Elt2, GA->Elt3,
								drop_out_level, image, verbose_level - 3)) {
							if (f_vv) {
								cout << "sims::build_up_group_from_generators element strips through kernel, "
										"residue = " << endl;
								if (f_vv) {
									KA->element_print(GA->Elt3, cout);
									cout << endl;
									}
								K->print(FALSE);
								K->print_basic_orbits();
								cout << "sims::build_up_group_from_generators residue" << endl;
								KA->element_print_image_of_set(
										GA->Elt3, KA->base_len(), KA->get_base());
								cout << "sims::build_up_group_from_generators Elt2" << endl;
								KA->element_print_image_of_set(
										GA->Elt2, KA->base_len(), KA->get_base());
							}
							if (!KA->element_is_one(GA->Elt3, FALSE)) {
								cout << "sims::build_up_group_from_generators but the element is not the identity, "
										"something is wrong" << endl;
								GA->element_print(GA->Elt3, cout);
								cout << endl;
								exit(1);
							}
						}
						K->add_generator_at_level(GA->Elt3,
								drop_out_level, verbose_level - 3);
						if (f_vv) {
							cout << "sims::build_up_group_from_generators the residue has been added as "
									"kernel generator at level "
									<< drop_out_level << endl;
						}
						f_added = TRUE;
					}
					else {
						if (f_vv) {
							cout << "sims::build_up_group_from_generators: "
									"choosing additional base point "
									<< b << endl;
						}
						old_base_len = GA->base_len();
						GA->Stabilizer_chain->reallocate_base(b);
						if (f_vv) {
							//cout << "after reallocate_base 1" << endl;
						}
						reallocate_base(old_base_len, verbose_level - 1);
						if (f_vv) {
							//cout << "after reallocate_base 2" << endl;
						}
						if (f_v) {
							cout << "sims::build_up_group_from_generators additional base point " << b
								<< " chosen, increased base has length "
								<< GA->base_len() << endl;
							cout << "sims::build_up_group_from_generators calling add_generator_at_level" << endl;
						}
						add_generator_at_level(GA->Elt2,
								GA->base_len() - 1, verbose_level - 3);
						if (f_vv) {
							cout << "sims::build_up_group_from_generators the residue has been added at level "
									<< GA->base_len() - 1 << endl;
						}
					} // if b
				} // if ! element is one
				else {
					if (f_vv) {
						cout << "sims::build_up_group_from_generators the residue is trivial" << endl;
					}
				}

				f_added = FALSE;
			}
			else {
				f_added = TRUE;
				if (f_vv) {
					cout << "sims::build_up_group_from_generators before add_generator_at_level" << endl;
				}
				add_generator_at_level(GA->Elt2,
						drop_out_level, 0 /*verbose_level - 1*/);
				if (f_vv) {
					cout << "sims::build_up_group_from_generators after add_generator_at_level" << endl;
				}
			}

			group_order(G_order);
			K->group_order(K_order);
			D.mult(G_order, K_order, KG_order);


			if (f_v && f_added) {
				cout << "sims::build_up_group_from_generators level " << level << " coset " << j
					<< " group of order increased to " << KG_order
					<< " = " << G_order << " * " << K_order << endl;
			}
			if (f_vv) {
				cout << "sims::build_up_group_from_generators level " << level << " coset " << j
					<< " found a group of order " << KG_order
					<< " = " << G_order << " * " << K_order << endl;
			}
		}
		} // end of schreier

		} // next level


	if (f_target_go) {
		int c, cnt;

		cnt = 0;
		while (TRUE) {
			group_order(G_order);
			K->group_order(K_order);
			D.mult(G_order, K_order, KG_order);

			c = D.compare(*target_go, KG_order);
			cnt++;
			if (c == 0) {
				if (f_v) {
					cout << "sims::build_up_group_from_generators reached the full group after "
							<< cnt << " iterations" << endl;
				}
				break;
			}
			if (c < 0) {
				if (TRUE) {
					cout << "sims::build_up_group_from_generators "
							"overshooting the expected group after "
							<< cnt << " iterations" << endl;
					cout << "current group order is " << KG_order
						<< " = " << G_order << " * " << K_order << endl;
				}
				//break;
				exit(1);
			}
			if (cnt > 10000) {
				cout << "sims::build_up_group_from_generators after "
						<< cnt << " iterations, we seem to be having "
						"problems reaching the target group order" << endl;
				cout << "sims::build_up_group_from_generators group order = " << KG_order << endl;
				cout << "sims::build_up_group_from_generators target group order = " << *target_go << endl;
				exit(1);
			}

			if (f_vv) {
				cout << "sims::build_up_group_from_generators "
						"calling closure group" << endl;
			}
			closure_group(10, verbose_level - 2);

		}
	}

	if (f_v) {
		cout << "sims::build_up_group_from_generators finished: "
				"found a group of order " << KG_order
			<< " = " << G_order << " * " << K_order << endl;
		cout << "sims::build_up_group_from_generators the n e w action has base_length = " << GA->base_len()
			<< " and degree " << GA->degree << endl;
		print_transversal_lengths();

#if 0
		if (f_vv) {
			print_transversals();
		}
		if (f_vvv) {
			print(FALSE);
		}
#endif
	}
	if (f_v) {
		cout << "sims::build_up_group_from_generators found a group of order " << G_order << endl;
	}
	if (f_v) {
		cout << "sims::build_up_group_from_generators done" << endl;
	}
}

int sims::closure_group(int nb_times, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	//int f_vvv = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_v3 = (verbose_level >= 6);
	int i, f_extended = FALSE;
	int *Elt1;
	int *Elt2;
	int *Elt3;
	ring_theory::longinteger_object old_go, go, go1;

	if (f_v) {
		cout << "sims::closure_group" << endl;
	}
	if (f_vv) {
		print_transversal_lengths();
		cout << "verbose_level=" << verbose_level << endl;
	}
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);

	group_order(old_go);
	if (f_vv) {
		cout << "sims::closure_group for group of order "
				<< old_go << endl;
	}
	if (old_go.is_one()) {
		FREE_int(Elt1);
		FREE_int(Elt2);
		if (f_v) {
			cout << "sims::closure_group finishing with FALSE "
					"because the old group order is one" << endl;
		}
		return FALSE;
	}
	group_order(go);
	for (i = 0; i < nb_times; i++) {
		if (f_vv) {
			cout << "sims::closure_group loop " << i << " / "
					<< nb_times << " go=" << go << endl;
		}
		if (f_v3) {
			cout << "sims::closure_group "
					"before random_schreier_generator" << endl;
		}
		random_schreier_generator(Elt3, verbose_level /*- 4*/);
		if (f_v3) {
			cout << "sims::closure_group "
					"after random_schreier_generator" << endl;
		}
		group_order(go);
		A->element_move(Elt3, Elt2, 0);
		if (strip_and_add(Elt2, Elt1 /* residue */,
				verbose_level - 3)) {
			group_order(go1);
			if (f_vv) {
				cout << "closure_group: iteration " << i
						<< " the group has been extended, old order "
					<< go << " extended group order " << go1 << endl;
				print_transversal_lengths();
			}
			if (f_v3) {
				cout << "original element:" << endl;
				A->element_print_quick(Elt3, cout);
				cout << "additional generator:" << endl;
				A->element_print_quick(Elt2, cout);
			}
			f_extended = TRUE;
		}
	}
	FREE_int(Elt1);
	FREE_int(Elt2);
	FREE_int(Elt3);
	if (f_extended) {
		if (f_v) {
			cout << "sims::closure_group group order extended from "
					<< old_go << " to " << go1 << endl;
			if (f_vv) {
				print_transversal_lengths();
			}
		}
	}
	else {
		if (f_vv) {
			cout << "sims::closure_group group order stays at "
					<< old_go << endl;
		}
	}
	return f_extended;
}



}}}

