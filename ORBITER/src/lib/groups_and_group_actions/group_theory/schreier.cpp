// schreier.C
//
// Anton Betten
// December 9, 2003

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

schreier::schreier()
{
	A = NULL;
	nb_images = 0;
	images = NULL;
	orbit = NULL;
	orbit_inv = NULL;
	prev = NULL;
	label = NULL;
	orbit_first = NULL;
	orbit_len = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	schreier_gen = NULL;
	schreier_gen1 = NULL;
	cosetrep = NULL;
	cosetrep_tmp = NULL;
	f_print_function = FALSE;
	print_function = NULL;
	print_function_data = NULL;
	nb_orbits = 0;
}

schreier::schreier(action *A)
{
	init(A);
};

schreier::~schreier()
{
	//cout << "in ~schreier()" << endl;
	freeself();
	//cout << "~schreier() finished" << endl;
};

void schreier::freeself()
{
	//cout << "deleting A" << endl;
	if (A) {
		//cout << "deleting orbit" << endl;
		FREE_int(orbit);
		//cout << "deleting orbit_inv" << endl;
		FREE_int(orbit_inv);
		//cout << "deleting prev" << endl;
		FREE_int(prev);
		//cout << "deleting label" << endl;
		FREE_int(label);
		//cout << "deleting orbit_first" << endl;
		FREE_int(orbit_first);
		//cout << "deleting orbit_len" << endl;
		FREE_int(orbit_len);
		//cout << "deleting Elt1" << endl;
		FREE_int(Elt1);
		//cout << "deleting Elt2" << endl;
		FREE_int(Elt2);
		//cout << "deleting Elt3" << endl;
		FREE_int(Elt3);
		//cout << "deleting schreier_gen" << endl;
		FREE_int(schreier_gen);
		//cout << "deleting schreier_gen1" << endl;
		FREE_int(schreier_gen1);
		//cout << "deleting cosetrep" << endl;
		FREE_int(cosetrep);
		//cout << "deleting cosetrep_tmp" << endl;
		FREE_int(cosetrep_tmp);
		//cout << "A = NULL" << endl;
		A = NULL;
		}
	//cout << "deleting images" << endl;
	delete_images();
}

void schreier::delete_images()
{
	int i;
	
	if (images) {
		for (i = 0; i < nb_images; i++) {
			FREE_int(images[i]);
			}
		FREE_pint(images);
		images = NULL;
		nb_images = 0;
		}
}

void schreier::init_images(int nb_images, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	
	if (f_v) {
		cout << "schreier::init_images" << endl;
		}
	if (A == NULL) {
		cout << "schreier::init_images action is NULL" << endl;
		exit(1);
		}
	delete_images();
	schreier::nb_images = nb_images;
	images = NEW_pint(nb_images);
	for (i = 0; i < nb_images; i++) {
		images[i] = NEW_int(2 * A->degree);
		for (j = 0; j < 2 * A->degree; j++) {
			images[i][j] = -1;
			}
		}
	if (f_v) {
		cout << "schreier::init_images done" << endl;
		}
}

void schreier::images_append()
{
	int **new_images = NEW_pint(nb_images + 1);
	int i, j;
	
	new_images[nb_images] = NEW_int(2 * A->degree);
	for (j = 0; j < 2 * A->degree; j++) {
		new_images[nb_images][j] = -1;
		}
	for (i = 0; i < nb_images; i++) {
		new_images[i] = images[i];
		}
	FREE_pint(images);
	images = new_images;
	nb_images++;
}

void schreier::init(action *A)
{
	schreier::A = A;
	orbit = NEW_int(A->degree);
	orbit_inv = NEW_int(A->degree);
	prev = NEW_int(A->degree);
	label = NEW_int(A->degree);
	//orbit_no = NEW_int(A->degree);
	orbit_first = NEW_int(A->degree + 1);
	orbit_len = NEW_int(A->degree);
	gens.init(A);
	gens_inv.init(A);
	initialize_tables();
	init2();
}

void schreier::init2()
{
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	schreier_gen = NEW_int(A->elt_size_in_int);
	schreier_gen1 = NEW_int(A->elt_size_in_int);
	cosetrep = NEW_int(A->elt_size_in_int);
	cosetrep_tmp = NEW_int(A->elt_size_in_int);
}

void schreier::initialize_tables()
{
	int i;
	
	nb_orbits = 0;
	perm_identity(orbit, A->degree);
	perm_identity(orbit_inv, A->degree);
	orbit_first[0] = 0;
	for (i = 0; i < A->degree; i++) {
		prev[i] = -1;
		label[i] = -1;
		//orbit_no[i] = -1;
		}
}

void schreier::init_single_generator(int *elt)
{
	init_generators(1, elt);
}

void schreier::init_generators(vector_ge &generators)
{
	if (generators.len) {
		init_generators(generators.len,
				generators.ith(0));
		}
	else {
		init_generators(generators.len, NULL);
		}
}

void schreier::init_generators(int nb, int *elt)
// elt must point to nb * A->elt_size_in_int
// int's that are
// group elements in int format
{
	int i;
	
	gens.allocate(nb);
	gens_inv.allocate(nb);
	for (i = 0; i < nb; i++) {
		//cout << "schreier::init_generators i = " << i << endl;
		gens.copy_in(i, elt + i * A->elt_size_in_int);
		A->element_invert(elt + i * A->elt_size_in_int,
				gens_inv.ith(i), 0);
		}
	init_images(nb, 0 /* verbose_level */);	
}

void schreier::init_generators_by_hdl(int nb_gen, 
	int *gen_hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	
	if (f_v) {
		cout << "schreier::init_generators_by_hdl" << endl;
		cout << "nb_gen = " << nb_gen << endl;
		cout << "degree = " << A->degree << endl;
		}
	gens.allocate(nb_gen);
	gens_inv.allocate(nb_gen);
	for (i = 0; i < nb_gen; i++) {
		//cout << "schreier::init_generators_by_hdl "
		// "i = " << i << endl;
		A->element_retrieve(gen_hdl[i], gens.ith(i), 0);
		
		//cout << "schreier::init_generators_by_hdl "
		// "generator i = " << i << ":" << endl;
		//A->element_print_quick(gens.ith(i), cout);

		A->element_invert(gens.ith(i), gens_inv.ith(i), 0);
		}
	if (f_vv) {
		cout << "schreier::init_generators_by_hdl "
				"generators:" << endl;
		gens.print(cout);
		}
	if (f_v) {
		cout << "schreier::init_generators_by_hdl "
				"before init_images()" << endl;
		}
	init_images(nb_gen, verbose_level);	
	if (f_v) {
		cout << "schreier::init_generators_by_hdl "
				"done" << endl;
		}
}

int schreier::get_image(int i, int gen_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a;
	
	if (f_v) {
		cout << "schreier::get_image computing image of point "
				<< i << " under generator " << gen_idx << endl;
		}
	if (images == NULL) {
		a = A->element_image_of(i,
				gens.ith(gen_idx), verbose_level - 2);
		//cout << "schreier::get_image"
		// "images == NULL" << endl;
		//exit(1);
		}
	else {
		a = images[gen_idx][i];
		if (a == -1) {
			a = A->element_image_of(i,
					gens.ith(gen_idx), verbose_level - 2);
			if (f_v) {
				cout << "schreier::get_image image of "
						"i=" << i << " is " << a << endl;
				}
			images[gen_idx][i] = a;
			images[gen_idx][A->degree + a] = i;
			}
		}
	if (f_v) {
		cout << "schreier::get_image image of point "
				<< i << " under generator " << gen_idx
				<< " is " << a << endl;
		}
	return a;
}

void schreier::swap_points(int i, int j)
{
	int pi, pj;
	
	pi = orbit[i];
	pj = orbit[j];
	orbit[i] = pj;
	orbit[j] = pi;
	orbit_inv[pi] = j;
	orbit_inv[pj] = i;
}

void schreier::move_point_here(int here, int pt)
{
	int a, loc;
	if (orbit[here] == pt)
		return;
	a = orbit[here];
	loc = orbit_inv[pt];
	orbit[here] = pt;
	orbit[loc] = a;
	orbit_inv[a] = loc;
	orbit_inv[pt] = here;
}

int schreier::orbit_representative(int pt)
{
	int j;
	
	while (TRUE) {
		j = orbit_inv[pt];
		if (prev[j] == -1)
			return pt;
		pt = prev[j];
		}
}

int schreier::depth_in_tree(int j)
// j is a coset, not a point
{
	if (prev[j] == -1) {
		return 0;
		}
	else {
		return depth_in_tree(orbit_inv[prev[j]]) + 1;
		}
}

void schreier::transporter_from_orbit_rep_to_point(int pt, 
	int &orbit_idx, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pos;

	if (f_v) {
		cout << "schreier::transporter_from_orbit_"
				"rep_to_point" << endl;
		}
	pos = orbit_inv[pt];
	orbit_idx = orbit_number(pt); //orbit_no[pos];
	//cout << "lies in orbit " << orbit_idx << endl;
	coset_rep(pos);
	A->element_move(cosetrep, Elt, 0);
	if (f_v) {
		cout << "schreier::transporter_from_orbit_"
				"rep_to_point done" << endl;
		}
}

void schreier::transporter_from_point_to_orbit_rep(int pt, 
	int &orbit_idx, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pos;

	if (f_v) {
		cout << "schreier::transporter_from_point_"
				"to_orbit_rep" << endl;
		}
	pos = orbit_inv[pt];
	orbit_idx = orbit_number(pt); //orbit_no[pos];
	//cout << "lies in orbit " << orbit_idx << endl;
	coset_rep_with_verbosity(pos, verbose_level - 1);
	A->element_invert(cosetrep, Elt, 0);
	//A->element_move(cosetrep, Elt, 0);
	if (f_v) {
		cout << "schreier::transporter_from_point_to_orbit_rep "
				"done" << endl;
		}
}

void schreier::coset_rep(int j)
// j is a coset, not a point
// result is in cosetrep
// determines an element in the group
// that moves the orbit representative
// to the j-th point in the orbit.
{
	int *gen;
	
	if (prev[j] != -1) {
		coset_rep(orbit_inv[prev[j]]);
		gen = gens.ith(label[j]);
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		}
	else {
		A->element_one(cosetrep, 0);
		}
}

void schreier::coset_rep_with_verbosity(int j, int verbose_level)
// j is a coset, not a point
// result is in cosetrep
// determines an element in the group
// that moves the orbit representative
// to the j-th point in the orbit.
{
	int f_v = (verbose_level >= 1);
	int *gen;
	
	if (f_v) {
		cout << "schreier::coset_rep_with_verbosity j="
				<< j << " orbit[j]=" << orbit[j] << endl;
		}
	if (prev[j] != -1) {
		if (f_v) {
			cout << "schreier::coset_rep_with_verbosity j=" << j
					<< " label[j]=" << label[j]
					<< " orbit_inv[prev[j]]="
					<< orbit_inv[prev[j]] << endl;
			}
		coset_rep_with_verbosity(orbit_inv[prev[j]], verbose_level);
		gen = gens.ith(label[j]);
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		}
	else {
		A->element_one(cosetrep, 0);
		}
	if (f_v) {
		cout << "schreier::coset_rep_with_verbosity "
				"j=" << j << " done" << endl;
		}
}

void schreier::coset_rep_inv(int j)
// j is a coset, not a point
// result is in cosetrep
{
	int *gen;
	
	if (prev[j] != -1) {
		coset_rep_inv(orbit_inv[prev[j]]);
		gen = gens_inv.ith(label[j]);
		A->element_mult(gen, cosetrep, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		}
	else {
		A->element_one(cosetrep, 0);
		}
}

void schreier::compute_point_orbit(int pt, int verbose_level)
{
	int pt_loc, cur, cur_pt, total, i, next_pt;
	int next_pt_loc, total1, cur1;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 5);
	//int f_vvv = FALSE; //(verbose_level >= 3);

	if (f_v) {
		cout << "schreier::compute_point_orbit" << endl;
		cout << "computing orbit of point " << pt
				<< " in action " << A->label << endl;
		}
	pt_loc = orbit_inv[pt];
	cur = orbit_first[nb_orbits];
	if (pt_loc < cur) {
		cout << "schreier::compute_point_orbit "
				"i < orbit_first[nb_orbits]" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "schreier::compute_point_orbit "
				"computing orbit of pt " << pt << endl;
		}
	if (pt_loc > orbit_first[nb_orbits]) {
		swap_points(orbit_first[nb_orbits], pt_loc);
		}
	//orbit_no[orbit_first[nb_orbits]] = nb_orbits;
	total = cur + 1;
	while (cur < total) {
		cur_pt = orbit[cur];
		if (f_vv) {
			cout << "schreier::compute_point_orbit cur="
					<< cur << " total=" << total
					<< " applying generators to " << cur_pt << endl;
			}
		for (i = 0; i < gens.len; i++) {
			if (f_vv) {
				cout << "schreier::compute_point_orbit "
						"applying generator "
						<< i << " to point " << cur_pt << endl;
				}
			next_pt = get_image(cur_pt, i,
				0 /*verbose_level - 5*/); // !!
				// A->element_image_of(cur_pt, gens.ith(i), FALSE);
			next_pt_loc = orbit_inv[next_pt];
			if (f_vv) {
				cout << "schreier::compute_point_orbit generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
				}
			if (next_pt_loc < total)
				continue;
			if (f_vv) {
				cout << "schreier::compute_point_orbit n e w pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
				}
			swap_points(total, next_pt_loc);
			prev[total] = cur_pt;
			label[total] = i;
			//orbit_no[total] = nb_orbits;
			total++;
			total1 = total - orbit_first[nb_orbits];
			cur1 = cur - orbit_first[nb_orbits];
			if ((total1 % 10000) == 0 ||
					(cur1 > 0 && (cur1 % 10000) == 0)) {
				cout << "schreier::compute_point_orbit degree = "
						<< A->degree << " length = " << total1
					<< " processed = " << cur1 << " nb_orbits="
					<< nb_orbits << " cur_pt=" << cur_pt << " next_pt="
					<< next_pt << " orbit_first[nb_orbits]="
					<< orbit_first[nb_orbits] << endl;
				}
			if (FALSE) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	orbit_first[nb_orbits + 1] = total;
	orbit_len[nb_orbits] = total - orbit_first[nb_orbits];
	//orbit_first[nb_orbits + 2] = A->degree;
	//orbit_len[nb_orbits + 1] = A->degree - total;
	if (f_v) {
		cout << "found orbit of length " << orbit_len[nb_orbits]
				<< " total length " << total
				<< " degree=" << A->degree << endl;
		}
	if (FALSE) {
		cout << "{ ";
		for (i = orbit_first[nb_orbits];
				i < orbit_first[nb_orbits + 1]; i++) {
			cout << orbit[i];
			if (i < orbit_first[nb_orbits + 1] - 1)
				cout << ", ";
			}
		cout << " }" << endl;
		}
	if (FALSE) {
		cout << "coset reps:" << endl;
		for (i = orbit_first[nb_orbits];
				i < orbit_first[nb_orbits + 1]; i++) {
			cout << i << " : " << endl;
			coset_rep(i);
			A->element_print(cosetrep, cout);
			cout << "image = " << orbit[i] << " = "
					<< A->element_image_of(pt, cosetrep, 0) << endl;
			cout << endl;

			}
		}
	nb_orbits++;
}

void schreier::compute_point_orbit_with_limited_depth(
		int pt, int max_depth, int verbose_level)
{
	int pt_loc, cur, cur_pt, total, i, next_pt;
	int next_pt_loc, total1, cur1;
	int *depth;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 5);
	//int f_vvv = FALSE; //(verbose_level >= 3);

	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth" << endl;
		cout << "computing orbit of point " << pt
				<< " in action " << A->label << endl;
		}
	depth = NEW_int(A->degree);
	int_vec_zero(depth, A->degree);
	pt_loc = orbit_inv[pt];
	cur = orbit_first[nb_orbits];
	if (pt_loc < cur) {
		cout << "schreier::compute_point_orbit_with_limited_depth "
				"i < orbit_first[nb_orbits]" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth "
				"computing orbit of pt " << pt << endl;
		}
	if (pt_loc > orbit_first[nb_orbits]) {
		swap_points(orbit_first[nb_orbits], pt_loc);
		}
	depth[cur] = 0;
	//orbit_no[orbit_first[nb_orbits]] = nb_orbits;
	total = cur + 1;
	while (cur < total) {
		cur_pt = orbit[cur];
		if (depth[cur] > max_depth) {
			break;
		}
		if (f_vv) {
			cout << "schreier::compute_point_orbit_with_limited_depth cur="
					<< cur << " total=" << total
					<< " applying generators to " << cur_pt << endl;
			}
		for (i = 0; i < gens.len; i++) {
			if (f_vv) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"applying generator "
						<< i << " to point " << cur_pt << endl;
				}
			next_pt = get_image(cur_pt, i,
				0 /*verbose_level - 5*/); // !!
				// A->element_image_of(cur_pt, gens.ith(i), FALSE);
			next_pt_loc = orbit_inv[next_pt];
			if (f_vv) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
				}
			if (next_pt_loc < total)
				continue;
			if (f_vv) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"n e w pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
				}
			swap_points(total, next_pt_loc);
			depth[total] = depth[cur] + 1;
			prev[total] = cur_pt;
			label[total] = i;
			//orbit_no[total] = nb_orbits;
			total++;
			total1 = total - orbit_first[nb_orbits];
			cur1 = cur - orbit_first[nb_orbits];
			if ((total1 % 10000) == 0 ||
					(cur1 > 0 && (cur1 % 10000) == 0)) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"degree = "
						<< A->degree << " length = " << total1
					<< " processed = " << cur1 << " nb_orbits="
					<< nb_orbits << " cur_pt=" << cur_pt << " next_pt="
					<< next_pt << " orbit_first[nb_orbits]="
					<< orbit_first[nb_orbits] << endl;
				}
			if (FALSE) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	orbit_first[nb_orbits + 1] = total;
	orbit_len[nb_orbits] = total - orbit_first[nb_orbits];
	//orbit_first[nb_orbits + 2] = A->degree;
	//orbit_len[nb_orbits + 1] = A->degree - total;
	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth "
				"found an incomplete orbit of length "
				<< orbit_len[nb_orbits]
				<< " total length " << total
				<< " degree=" << A->degree << endl;
		}
	FREE_int(depth);
	nb_orbits++;
}






void schreier::extend_orbit(int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int cur, total0, total, cur_pt;
	int gen_first, i, next_pt, next_pt_loc;
	
	if (f_v) {
		cout << "extend_orbit() extending orbit "
				<< nb_orbits - 1 << " of length "
			<< orbit_len[nb_orbits - 1] << endl;
		}

	gens.append(elt);
	A->element_invert(elt, A->Elt1, FALSE);
	gens_inv.append(A->Elt1);
	images_append();
	
	cur = orbit_first[nb_orbits - 1];
	total = total0 = orbit_first[nb_orbits];
	while (cur < total) {
		cur_pt = orbit[cur];
		if (FALSE) {
			cout << "schreier::extend_orbit "
					"applying generator to " << cur_pt << endl;
			}
#if 0
		if (cur < total0)
			gen_first = gens.len - 1;
		else 
			gen_first = 0;
#endif
		gen_first = 0;
		for (i = gen_first; i < gens.len; i++) {
			next_pt = get_image(cur_pt, i, 0/*verbose_level - 3*/);
				// A->element_image_of(cur_pt, gens.ith(i), FALSE);
			next_pt_loc = orbit_inv[next_pt];
			if (FALSE) {
				cout << "schreier::extend_orbit generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
				}
			if (next_pt_loc < total)
				continue;
			if (FALSE) {
				cout << "schreier::extend_orbit n e w pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
				}
			swap_points(total, next_pt_loc);
			prev[total] = cur_pt;
			label[total] = i;
			//orbit_no[total] = nb_orbits - 1;
			total++;
			if (FALSE) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	orbit_first[nb_orbits] = total;
	orbit_len[nb_orbits - 1] = total - orbit_first[nb_orbits - 1];
	//orbit_first[nb_orbits + 1] = A->degree;
	//orbit_len[nb_orbits] = A->degree - total;
	if (f_v) {
		cout << "schreier::extend_orbit orbit extended to length "
				<< orbit_len[nb_orbits - 1] << endl;
		}
	if (FALSE) {
		cout << "{ ";
		for (i = orbit_first[nb_orbits - 1];
				i < orbit_first[nb_orbits]; i++) {
			cout << orbit[i];
			if (i < orbit_first[nb_orbits] - 1)
				cout << ", ";
			}
		cout << " }" << endl;
		}
}

void schreier::compute_all_point_orbits(int verbose_level)
{
	int pt, pt_loc, cur, pt0;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits "
				"verbose_level=" << verbose_level << endl;
		}
	if (A->degree > ONE_MILLION) {
		f_vv = FALSE;
		}
	initialize_tables();
	for (pt0 = 0, pt = 0; pt < A->degree; pt++) {
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		if (f_vv) {
			cout << "schreier::compute_all_point_orbits pt = "
					<< pt << " / " << A->degree
					<< " nb_orbits=" << nb_orbits
					<< " cur=" << cur
					<< ", computing orbit" << endl;
			}
		if (A->degree > ONE_MILLION && (pt - pt0) > 50000) {
			cout << "schreier::compute_all_point_orbits pt = "
					<< pt << " / " << A->degree
					<< " nb_orbits=" << nb_orbits
					<< " cur=" << cur
					<< ", computing orbit" << endl;
			pt0 = pt;
			}
		compute_point_orbit(pt, verbose_level - 2);
		}
	if (f_v) {
		cout << "schreier::compute_all_point_orbits found "
				<< nb_orbits << " orbits" << endl;
		classify Cl;

		Cl.init(orbit_len, nb_orbits, FALSE, 0);
		cout << "The distribution of orbit lengths is: ";
		Cl.print(FALSE);
		}
}

void schreier::compute_all_point_orbits_with_prefered_reps(
	int *prefered_reps, int nb_prefered_reps, 
	int verbose_level)
{
	int i, pt, pt_loc, cur;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_"
				"with_prefered_reps" << endl;
		}
	initialize_tables();
	for (i = 0; i < nb_prefered_reps; i++) {
		pt = prefered_reps[i];
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		compute_point_orbit(pt, verbose_level - 1);
		}
	for (pt = 0; pt < A->degree; pt++) {
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		compute_point_orbit(pt, verbose_level - 1);
		}
	if (f_v) {
		cout << "found " << nb_orbits << " orbit";
		if (nb_orbits != 1)
			cout << "s";
		cout << " on points" << endl;
		}
}


void schreier::compute_all_point_orbits_with_preferred_labels(
	int *preferred_labels, int verbose_level)
{
	int pt, pt_loc, cur, a, i;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int *labels, *perm, *perm_inv;
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_"
				"preferred_labels" << endl;
		//cout << "preferred_labels :";
		//int_vec_print(cout, preferred_labels, A->degree);
		//cout << endl;
		cout << "A->degree = " << A->degree << endl;
		}
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_"
				"preferred_labels allocating tables" << endl;
		}
	initialize_tables();
	labels = NEW_int(A->degree);
	perm = NEW_int(A->degree);
	perm_inv = NEW_int(A->degree);
	for (i = 0; i < A->degree; i++) {
		labels[i] = preferred_labels[i];
		}
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_"
				"with_preferred_labels allocating tables done, "
				"sorting" << endl;
		}
	int_vec_sorting_permutation(labels, A->degree,
			perm, perm_inv, TRUE /* f_increasingly */);

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_"
				"with_preferred_labels sorting done" << endl;
		}
	
	for (a = 0; a < A->degree; a++) {
		pt = perm_inv[a];
		pt_loc = orbit_inv[pt];
		cur = orbit_first[nb_orbits];
		if (pt_loc < cur) {
			continue;
			}
		// now we need to make sure that the point pt
		// is moved to position cur:
		// actually this is not needed as the
		// function compute_point_orbit does this, too.
		swap_points(cur, pt_loc);
		
		if (f_v) {
			cout << "schreier::compute_all_point_orbits_with_"
					"preferred_labels computing orbit of point "
					<< pt << " = " << a << " / " << A->degree << endl;
			}
		compute_point_orbit(pt, verbose_level - 2);
		if (f_v) {
			cout << "schreier::compute_all_point_orbits_with_"
					"preferred_labels computing orbit of point "
					<< pt << " done, found an orbit of length "
					<< orbit_len[nb_orbits - 1]
					<< " nb_orbits = " << nb_orbits << endl;
			}
		}
	if (f_v) {
		cout << "found " << nb_orbits << " orbit";
		if (nb_orbits != 1)
			cout << "s";
		cout << " on points" << endl;
		}
	FREE_int(labels);
	FREE_int(perm);
	FREE_int(perm_inv);
}

void schreier::compute_all_orbits_on_invariant_subset(
	int len, int *subset, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f;
	
	if (f_v) {
		cout << "schreier::compute_all_orbits_on_invariant_subset" << endl;
		cout << "computing orbits on a set of size " << len << endl;
		}
	initialize_tables();
	for (i = 0; i < len; i++) {
		move_point_here(i, subset[i]);
		}
	while (TRUE) {
		f = orbit_first[nb_orbits];
		if (f >= len)
			break;
		compute_point_orbit(orbit[f], 0 /* verbose_level */);
		}
	if (f > len) {
		cout << "schreier::compute_all_orbits_on_invariant_subset "
				"the set is not G-invariant" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "found " << nb_orbits << " orbits" << endl;
		print_orbit_length_distribution(cout);
		}
}

int schreier::sum_up_orbit_lengths()
{
	int i, l, N;
	
		N = 0;
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		N += l;
		}
	return N;
}

void schreier::non_trivial_random_schreier_generator(
	action *A_original, int verbose_level)
// computes non trivial random Schreier generator into schreier_gen
// non-trivial is with respect to A_original
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	int f_v4 = FALSE; //(verbose_level >= 4);
	int cnt = 0;
	
	if (f_v) {
		cout << "schreier::non_trivial_random_schreier_generator "
				"verbose_level=" << verbose_level << endl;
		}
	while (TRUE) {
		if (f_v) {
			cout << "schreier::non_trivial_random_schreier_generator "
					"calling random_schreier_generator "
					"(cnt=" << cnt << ")" << endl;
			}
		random_schreier_generator(verbose_level - 1);
		cnt++;
		if (!A_original->element_is_one(schreier_gen, verbose_level - 5)) {
			if (f_vv) {
				cout << "schreier::non_trivial_random_schreier_generator "
						"found a non-trivial random Schreier generator in "
						<< cnt << " trials" << endl;
				}
			if (f_vvv) {
				A->element_print(schreier_gen, cout);
				cout << endl;
				}
			return;
			}
		else {
			if (f_v4) {
				A->element_print(schreier_gen, cout);
				cout << endl;
				}
			if (f_vv) {
				cout << "schreier::non_trivial_random_schreier_generator "
						"the element is the identity in action "
						<< A_original->label << ", trying again" << endl;
				}
			}
		}
}

void schreier::random_schreier_generator_ith_orbit(
	int orbit_no, int verbose_level)
{
	int first, len, r1, r2, pt, pt2, pt2_coset;
	int *gen;
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; //(verbose_level >= 2);
	int f_vvv = FALSE; //(verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::random_schreier_generator_ith_orbit, "
				"orbit " << orbit_no << endl;
		}
	if (f_vvv) {
		cout << "generators are:" << endl;
		gens.print(cout);
		}
	first = orbit_first[orbit_no];
	len = orbit_len[orbit_no];
	pt = orbit[first];
	if (f_vv) {
		cout << "pt=" << pt << endl;
		cout << "orbit_first[orbit_no]=" << orbit_first[orbit_no] << endl;
		cout << "orbit_len[orbit_no]=" << orbit_len[orbit_no] << endl;
		cout << "gens.len=" << gens.len << endl;
		}
	
	// get a random coset:
	r1 = random_integer(orbit_len[orbit_no]);
	if (f_vv) {
		cout << "r1=" << r1 << endl;
		}
	//pt1 = orbit[r1];
	coset_rep(orbit_first[orbit_no] + r1);
	// coset rep now in cosetrep
	if (f_vvv) {
		cout << "cosetrep " << orbit_first[orbit_no] + r1 << endl;
		A->element_print_quick(cosetrep, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(cosetrep, cout);
			cout << endl;
			}
		}
		
	// get a random generator:
	r2 = random_integer(gens.len);
	if (f_vv) {
		cout << "r2=" << r2 << endl;
		}
	gen = gens.ith(r2);
	if (f_vvv) {
		cout << "generator " << r2 << endl;
		A->element_print(gen, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(gen, cout);
			cout << endl;
			}
		}
	if (f_vv) {
		cout << "random coset " << r1
				<< ", random generator " << r2 << endl;
		}
	
	A->element_mult(cosetrep, gen, schreier_gen1, 0);
	if (f_vvv) {
		cout << "cosetrep * generator " << endl;
		A->element_print_quick(schreier_gen1, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(schreier_gen1, cout);
			cout << endl;
			}
		}
	pt2 = A->element_image_of(pt, schreier_gen1, 0);
	if (f_vv) {
		//cout << "pt2=" << pt2 << endl;
		cout << "maps " << pt << " to " << pt2 << endl;
		}
	pt2_coset = orbit_inv[pt2];
	if (f_vv) {
		cout << "pt2_coset=" << pt2_coset << endl;
		}
	if (pt2_coset < first) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"pt2_coset < first" << endl;
		exit(1);
		}
	if (pt2_coset >= first + len) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"pt2_coset >= first + len" << endl;
		exit(1);
		}
	
	coset_rep_inv(pt2_coset);
	// coset rep now in cosetrep
	if (f_vvv) {
		cout << "cosetrep (inverse) " << pt2_coset << endl;
		A->element_print_quick(cosetrep, cout);
		if (A->degree < 100) {
			A->element_print_as_permutation(cosetrep, cout);
			cout << endl;
			}
		}
	
	A->element_mult(schreier_gen1, cosetrep, schreier_gen, 0);
	if (A->element_image_of(pt, schreier_gen, 0) != pt) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"fatal: schreier generator does not stabilize pt" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "schreier::random_schreier_generator_ith_orbit "
				"done" << endl;
		}
	if (f_vvv) {
		A->element_print_quick(schreier_gen, cout);
		cout << endl;
		if (A->degree < 100) {
			A->element_print_as_permutation(cosetrep, cout);
			cout << endl;
			}
		}
}

void schreier::random_schreier_generator(int verbose_level)
// computes random Schreier generator into schreier_gen
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int r1, r2, pt, pt2, pt2b, pt2_coset;
	int *gen;
	int pt1, pt1b;
	
	if (f_v) {
		cout << "schreier::random_schreier_generator orbit_len = " 
			<< orbit_len[0] << " nb generators = "
			<< gens.len << " in action " << A->label << endl;
		}
	pt = orbit[0];
	if (f_vv) {
		cout << "pt=" << pt << endl;
		}
	
	// get a random coset:
	r1 = random_integer(orbit_len[0]);
	pt1 = orbit[r1];
	
	coset_rep(r1);
	// coset rep now in cosetrep
	pt1b = A->element_image_of(pt, cosetrep, 0);
	if (f_vv) {
		cout << "random coset " << r1 << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "image of pt under cosetrep = " << pt1b << endl;
		}
	if (pt1b != pt1) {
		cout << "schreier::random_schreier_generator fatal: "
				"cosetrep does not work" << endl;
		cout << "pt=" << pt << endl;
		cout << "random coset " << r1 << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "image of pt under cosetrep = " << pt1b << endl;
		A->element_image_of(pt, cosetrep, 10);	
		exit(1);
		}
	
	// get a random generator:
	r2 = random_integer(gens.len);
	gen = gens.ith(r2);
	if (f_vv) {
		cout << "random coset " << r1 << ", "
				"random generator " << r2 << endl;
		cout << "generator:" << endl;
		A->element_print_quick(gen, cout);
		cout << "image of pt1 under generator = pt2 = "
				<< A->element_image_of(pt1, gen, 0) << endl;
		}
	pt2b = A->element_image_of(pt1, gen, 0);
	
	A->element_mult(cosetrep, gen, schreier_gen1, 0);
	if (f_vv) {
		cout << "cosetrep * gen=" << endl;
		A->element_print_quick(schreier_gen1, cout);
		}
	pt2 = A->element_image_of(pt, schreier_gen1, 0);
	if (f_vv) {
		cout << "image of pt under cosetrep*gen = " << pt2 << endl;
		}
	if (pt2 != pt2b) {
		cout << "schreier::random_schreier_generator "
				"something is wrong! " << endl;
		cout << "pt2=" << pt2 << " = image of pt "
				"under cosetrep * gen" << endl;
		cout << "pt2b=" << pt2b << " = image of pt1 "
				"under gen" << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "generator:" << endl;
		A->element_print_quick(gen, cout);
		cout << "cosetrep * gen=" << endl;
		A->element_print_quick(schreier_gen1, cout);
		cout << "pt=" << pt << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "pt1b=" << pt1b << endl;
		cout << "pt2=" << pt2 << endl;
		cout << "pt2b=" << pt2b << endl;

		cout << "repeat 1" << endl;
		cout << "repeating pt1b = A->element_image_of(pt, "
				"cosetrep, 0):" << endl;
		pt1b = A->element_image_of(pt, cosetrep, verbose_level + 3);
		cout << "pt1b = " << pt1b << endl;

		cout << "repeat 2" << endl;
		cout << "repeating pt2b = A->element_image_of(pt1, "
				"gen, 0):" << endl;
		pt2b = A->element_image_of(pt1, gen, verbose_level + 3);

		cout << "repeat 3" << endl;
		cout << "repeating pt2 = A->element_image_of(pt, "
				"schreier_gen1, 0):" << endl;
		pt2 = A->element_image_of(pt, schreier_gen1, verbose_level + 3);


		exit(1);
	}
	//cout << "maps " << pt << " to " << pt2 << endl;
	pt2_coset = orbit_inv[pt2];
	
	coset_rep_inv(pt2_coset);
	// coset rep now in cosetrep
	if (f_vv) {
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		cout << "image of pt2 under cosetrep = "
				<< A->element_image_of(pt2, cosetrep, 0) << endl;
		}
	
	A->element_mult(schreier_gen1, cosetrep, schreier_gen, 0);
	if (f_vv) {
		cout << "schreier_gen=cosetrep*gen*cosetrep:" << endl;
		A->element_print_quick(schreier_gen, cout);
		cout << "image of pt under schreier_gen = "
				<< A->element_image_of(pt, schreier_gen, 0) << endl;
		}
	if (A->element_image_of(pt, schreier_gen, 0) != pt) {
		cout << "schreier::random_schreier_generator() "
				"fatal: schreier generator does not stabilize pt" << endl;
		exit(1);
		}
	if (FALSE) {
		cout << "random Schreier generator:" << endl;
		A->element_print(schreier_gen, cout);
		cout << endl;
		}
}

void schreier::trace_back(int *path, int i, int &j)
{
	int ii = orbit_inv[i];
	
	if (prev[ii] == -1) {
		if (path) {
			path[0] = i;
			}
		j = 1;
		}
	else {
		trace_back(path, prev[ii], j);
		if (path) {
			path[j] = i;
			}
		j++;
		}
}

void schreier::intersection_vector(int *set,
		int len, int *intersection_cnt)
{
	int i, pt, /*pt_loc,*/ o;
	
	for (i = 0; i < nb_orbits; i++) {
		intersection_cnt[i] = 0;
		}
	for (i = 0; i < len; i++) {
		pt = set[i];
		//pt_loc = orbit_inv[pt];
		o = orbit_number(pt); // orbit_no[pt_loc];
		intersection_cnt[o]++;
		}
}

void schreier::orbits_on_invariant_subset_fast(
	int len, int *subset, int verbose_level)
{
	int i, p, j;
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast "
			"computing orbits on invariant subset "
			"of size " << len << " in action ";
		A->print_info();
		}
	
	for (i = 0; i < len; i++) {
		p = subset[i];
		j = orbit_inv[p];
		if (j >= orbit_first[nb_orbits]) {
			if (f_vvv) {
				cout << "computing orbit no " << nb_orbits << endl;
				}
			compute_point_orbit(p, 0);
			}
		}
#if 0
	if (orbit_first[nb_orbits] != len) {
		cout << "schreier::orbits_on_invariant_subset_"
				"fast orbit_first[nb_orbits] != len" << endl;
		cout << "orbit_first[nb_orbits] = "
				<< orbit_first[nb_orbits] << endl;
		cout << "len = " << len << endl;
		cout << "subset:" << endl;
		int_vec_print(cout, subset, len);
		cout << endl;
		print_tables(cout, FALSE);
		exit(1);
		}
#endif
	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast "
			"found " << nb_orbits
			<< " orbits on the invariant subset of size " << len << endl;
		}
}

void schreier::orbits_on_invariant_subset(int len, int *subset, 
	int &nb_orbits_on_subset, 
	int *&orbit_perm, int *&orbit_perm_inv)
{
	int i, j, a, pos;
	
	compute_all_point_orbits(0);
	nb_orbits_on_subset = 0;
	orbit_perm = NEW_int(nb_orbits);
	orbit_perm_inv = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		orbit_perm_inv[i] = -1;
		}
	for (i = 0; i < nb_orbits; i++) {
		j = orbit_first[i];
		a = orbit[j];
		for (pos = 0; pos < len; pos++) {
			if (subset[pos] == a) {
				orbit_perm[nb_orbits_on_subset] = i;
				orbit_perm_inv[i] = nb_orbits_on_subset;
				nb_orbits_on_subset++;
				break;
				}
			}
		}
	j = nb_orbits_on_subset;
	for (i = 0; i < nb_orbits; i++) {
		if (orbit_perm_inv[i] == -1) {
			orbit_perm[j] = i;
			orbit_perm_inv[i] = j;
			j++;
			}
		}	
}

void schreier::get_orbit_partition_of_points_and_lines(
	partitionstack &S, int verbose_level)
{
	int first_column_element, pos, first_column_orbit, i, j, f, l, a;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::get_orbit_partition_"
				"of_points_and_lines" << endl;
		}
	first_column_element = S.startCell[1];
	if (f_v) {
		cout << "first_column_element = "
				<< first_column_element << endl;
		}
	pos = orbit_inv[first_column_element];
	first_column_orbit = orbit_number(first_column_element);
	
	for (i = first_column_orbit - 1; i > 0; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
	for (i = nb_orbits - 1; i > first_column_orbit; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
}

void schreier::get_orbit_partition(partitionstack &S, 
	int verbose_level)
{
	int pos, i, j, f, l, a;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::get_orbit_partition" << endl;
		}
	for (i = nb_orbits - 1; i > 0; i--) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			pos = f + j;
			a = orbit[pos];
			S.subset[j] = a;
			}
		S.subset_size = l;
		S.split_cell(FALSE);
		}
}

strong_generators *schreier::stabilizer_any_point_plus_cosets(
	action *default_action, 
	longinteger_object &full_group_order, int pt, vector_ge *&cosets, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	strong_generators *gens0;
	strong_generators *gens;
	int orbit_index;
	int orbit_index1;
	int *transporter;
	int *transporter1;
	int i, fst, len;

	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets" << endl;
		}
	
	cosets = NEW_OBJECT(vector_ge);
	cosets->init(A);
	transporter = NEW_int(A->elt_size_in_int);
	transporter1 = NEW_int(A->elt_size_in_int);
	
	orbit_index = orbit_number(pt);

	gens0 = stabilizer_orbit_rep(default_action, 
		full_group_order, orbit_index, 0 /* verbose_level */);

	fst = orbit_first[orbit_index];
	len = orbit_len[orbit_index];
	cosets->allocate(len);

	transporter_from_point_to_orbit_rep(pt,
			orbit_index1, transporter, 0 /* verbose_level */);

	if (orbit_index1 != orbit_index) {
		cout << "schreier::stabilizer_any_point_plus_cosets "
				"orbit_index1 != orbit_index" << endl;
		exit(1);
		}
	
	gens = NEW_OBJECT(strong_generators);

	
	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets before "
				"gens->init_generators_for_the_conjugate_group_aGav" << endl;
		}
	gens->init_generators_for_the_conjugate_group_aGav(gens0, 
		transporter, verbose_level);
	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets after "
				"gens->init_generators_for_the_conjugate_group_aGav" << endl;
		}

	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets computing "
				"coset representatives" << endl;
		}
	for (i = 0; i < len; i++) {
		transporter_from_orbit_rep_to_point(orbit[fst + i],
				orbit_index1, transporter1, 0 /* verbose_level */);
		A->element_mult(transporter, transporter1, cosets->ith(i), 0);
		}
	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets computing "
				"coset representatives done" << endl;
		}

	FREE_int(transporter);
	FREE_int(transporter1);
	
	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets done" << endl;
		}
	FREE_OBJECT(gens0);
	return gens;
}

strong_generators *schreier::stabilizer_any_point(
	action *default_action, 
	longinteger_object &full_group_order, int pt, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	strong_generators *gens0;
	strong_generators *gens;
	int orbit_index;
	int orbit_index1;
	int *transporter;

	if (f_v) {
		cout << "schreier::stabilizer_any_point" << endl;
		}
	
	transporter = NEW_int(A->elt_size_in_int);
	
	orbit_index = orbit_number(pt);

	gens0 = stabilizer_orbit_rep(default_action, 
		full_group_order, orbit_index, 0 /* verbose_level */);

	transporter_from_point_to_orbit_rep(pt,
			orbit_index1, transporter, 0 /* verbose_level */);

	if (orbit_index1 != orbit_index) {
		cout << "schreier::stabilizer_any_point "
				"orbit_index1 != orbit_index" << endl;
		exit(1);
		}
	
	gens = NEW_OBJECT(strong_generators);

	
	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"before gens->init_generators_"
				"for_the_conjugate_group_aGav" << endl;
		}
	gens->init_generators_for_the_conjugate_group_aGav(gens0, 
		transporter, verbose_level);
	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"after gens->init_generators_"
				"for_the_conjugate_group_aGav" << endl;
		}

	FREE_int(transporter);
	
	if (f_v) {
		cout << "schreier::stabilizer_any_point done" << endl;
		}
	FREE_OBJECT(gens0);
	return gens;
}


strong_generators *schreier::stabilizer_orbit_rep(
	action *default_action, 
	longinteger_object &full_group_order, int orbit_idx, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	strong_generators *gens;
	sims *Stab;

	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep" << endl;
		}


	point_stabilizer(default_action, full_group_order, 
		Stab, orbit_idx, verbose_level);

	longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep "
				"found a stabilizer group "
				"of order " << stab_order << endl;
		}
	
	gens = NEW_OBJECT(strong_generators);
	gens->init(A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep done" << endl;
		}
	return gens;
}

void schreier::point_stabilizer(action *default_action, 
	longinteger_object &go, 
	sims *&Stab, int orbit_no, 
	int verbose_level)
// this function allocates a sims structure into Stab.
{
	Stab = NEW_OBJECT(sims);
	longinteger_object cur_go, target_go;
	longinteger_domain D;
	int len, r, cnt = 0, f_added, *p_gen, drop_out_level, image;
	int *residue;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);
	//int f_v5 = (verbose_level >= 5);
	
	
	if (f_v) {
		cout << "schreier::point_stabilizer "
				"computing stabilizer of representative of orbit "
			<< orbit_no << " inside a group of "
					"order " << go << " in action ";
		default_action->print_info();
		cout << endl;
		}
	residue = NEW_int(default_action->elt_size_in_int);
	len = orbit_len[orbit_no];
	D.integral_division_by_int(go, len, target_go, r);
	if (r) {	
		cout << "schreier::point_stabilizer "
				"orbit length does not divide group order" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "expecting group of order " << target_go << endl;
		}
	
	Stab->init(default_action);
	Stab->init_trivial_group(verbose_level - 1);
	while (TRUE) {
		Stab->group_order(cur_go);
		if (D.compare(cur_go, target_go) == 0) {
			break;
			}
		if (cnt % 2 || Stab->nb_gen[0] == 0) {
			random_schreier_generator_ith_orbit(orbit_no,
					0 /* verbose_level */);
			p_gen = schreier_gen;
			if (f_vvv) {
				cout << "random Schreier generator from the orbit:" << endl;
				default_action->element_print(p_gen, cout);
				}
			}
		else {
			Stab->random_schreier_generator(0 /* verbose_level */);
			p_gen = Stab->schreier_gen;
			if (f_v4) {
				cout << "random schreier generator from sims:" << endl;
				default_action->element_print(p_gen, cout);
				}
			}



		if (Stab->strip(p_gen, residue,
				drop_out_level, image, 0 /*verbose_level - 3*/)) {
			if (f_vvv) {
				cout << "element strips through" << endl;
				if (f_v4) {
					cout << "residue:" << endl;
					A->element_print(residue, cout);
					cout << endl;
					}
				}
			f_added = FALSE;
			}
		else {
			f_added = TRUE;
			if (f_vvv) {
				cout << "element needs to be inserted at level = " 
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					A->element_print(residue, cout);
					cout  << endl;
					}
				}
			Stab->add_generator_at_level(residue,
					drop_out_level, verbose_level - 4);
			}
		Stab->group_order(cur_go);
		if ((f_vv && f_added) || f_vvv) {
			cout << "iteration " << cnt
				<< " the n e w group order is " << cur_go
				<< " expecting a group of order "
				<< target_go << endl;
			}
		cnt++;
		}
	FREE_int(residue);
	if (f_v) {
		cout << "schreier::point_stabilizer finished" << endl;
		}
}

void schreier::get_orbit(int orbit_idx, int *set, int &len, 
	int verbose_level)
{
	int f, i;

	f = orbit_first[orbit_idx];
	len = orbit_len[orbit_idx];
	for (i = 0; i < len; i++) {
		set[i] = orbit[f + i];
		}
}

void schreier::compute_orbit_statistic(int *set, int set_size, 
	int *orbit_count, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, /*j,*/ o;

	if (f_v) {
		cout << "schreier::compute_orbit_statistic" << endl;
		}
	int_vec_zero(orbit_count, nb_orbits);
#if 0
	for (i = 0; i < nb_orbits; i++) {
		orbit_count[i] = 0;
		}
#endif
	for (i = 0; i < set_size; i++) {
		a = set[i];
		//j = orbit_inv[a];
		o = orbit_number(a); //orbit_no[j];
		orbit_count[o]++;
		}
	if (f_v) {
		cout << "schreier::compute_orbit_statistic done" << endl;
		}
}


void schreier::orbits_as_set_of_sets(
		set_of_sets *&S, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Sz;
	int i, j, a, f, l;
	
	if (f_v) {
		cout << "schreier::orbits_as_set_of_sets" << endl;
		}
	S = NEW_OBJECT(set_of_sets);
	Sz = NEW_int(nb_orbits);
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		Sz[i] = l;
		}
	
	S->init_basic(A->degree /* underlying_set_size */,
			nb_orbits, Sz, 0 /* verbose_level */);
	for (i = 0; i < nb_orbits; i++) {
		f = orbit_first[i];
		l = orbit_len[i];
		for (j = 0; j < l; j++) {
			a = orbit[f + j];
			S->Sets[i][j] = a;
			}
		}
	FREE_int(Sz);
	if (f_v) {
		cout << "schreier::orbits_as_set_of_sets done" << endl;
		}
}

void schreier::get_orbit_reps(int *&Reps,
		int &nb_reps, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, a, f;
	
	if (f_v) {
		cout << "schreier::get_orbit_reps" << endl;
		}
	nb_reps = nb_orbits;
	Reps = NEW_int(nb_reps);
	for (i = 0; i < nb_reps; i++) {
		f = orbit_first[i];
		a = orbit[f];
		Reps[i] = a;
		}
	if (f_v) {
		cout << "schreier::get_orbit_reps done" << endl;
		}
}

int schreier::find_shortest_orbit_if_unique(int &idx)
{
	int l_min = 0, l, i;
	int idx_min = -1;
	int f_is_unique = TRUE;
	
	for (i = 0; i < nb_orbits; i++) {
		l = orbit_len[i];
		if (idx_min == -1) {
			l_min = l;
			idx_min = i;
			f_is_unique = TRUE;
			}
		else if (l < l_min) {
			l_min = l;
			idx_min = i;
			f_is_unique = TRUE;
			}
		else if (l_min == l) {
			f_is_unique = FALSE;
			}
		}
	idx = idx_min;
	return f_is_unique;
}

void schreier::elements_in_orbit_of(int pt, 
	int *orb, int &nb, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int idx, f;

	if (f_v) {
		cout << "schreier::elements_in_orbit_of" << endl;
		}
	idx = orbit_number(pt);
	f = orbit_first[idx];
	nb = orbit_len[idx];
	int_vec_copy(orbit + f, orb, nb);
	if (f_v) {
		cout << "schreier::elements_in_orbit_of done" << endl;
		}
}

void schreier::get_orbit_lengths_once_each(
	int *&orbit_lengths, int &nb_orbit_lengths)
{
	int *val, *mult, len;	
	
	int_vec_distribution(orbit_len, nb_orbits, val, mult, len);
	//int_distribution_print(ost, val, mult, len);
	//ost << endl;
	
	nb_orbit_lengths = len;

	orbit_lengths = NEW_int(nb_orbit_lengths);

	int_vec_copy(val, orbit_lengths, nb_orbit_lengths);

	FREE_int(val);
	FREE_int(mult);
}


int schreier::orbit_number(int pt)
{
	int pos;
	int idx;

	pos = orbit_inv[pt];
	if (int_vec_search(orbit_first, nb_orbits, pos, idx)) {
		;
		}
	else {
		if (idx == 0) {
			cout << "schreier::orbit_number idx == 0" << endl;
			exit(1);
			}
		idx--;
		}
	if (orbit_first[idx] <= pos &&
			pos < orbit_first[idx] + orbit_len[idx]) {
		return idx;
		}
	else {
		cout << "schreier::orbit_number something is wrong, "
				"perhaps the orbit of the point has not yet "
				"been computed" << endl;
		exit(1);
		}
}

void schreier::get_orbit_decomposition_scheme_of_graph(
	int *Adj, int n, int *&Decomp_scheme, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J;
	int f1, l1;
	int f2, l2;
	int i, j, r, r0, a, b;

	if (f_v) {
		cout << "schreier::get_orbit_decomposition_"
				"scheme_of_graph" << endl;
		}
	Decomp_scheme = NEW_int(nb_orbits * nb_orbits);
	int_vec_zero(Decomp_scheme, nb_orbits * nb_orbits);
	for (I = 0; I < nb_orbits; I++) {
		f1 = orbit_first[I];
		l1 = orbit_len[I];
		if (FALSE) {
			cout << "I = " << I << " f1 = " << f1
					<< " l1 = " << l1 << endl;
			}
		for (J = 0; J < nb_orbits; J++) {
			r0 = 0;
			f2 = orbit_first[J];
			l2 = orbit_len[J];
			if (FALSE) {
				cout << "J = " << J << " f2 = " << f2
						<< " l2 = " << l2 << endl;
				}
			for (i = 0; i < l1; i++) {
				a = orbit[f1 + i];
				r = 0;
				for (j = 0; j < l2; j++) {
					b = orbit[f2 + j];
					if (Adj[a * n + b]) {
						r++;
						}
					}
				if (i == 0) {
					r0 = r;
					}
				else {
					if (r0 != r) {
						cout << "schreier::get_orbit_decomposition_"
								"scheme_of_graph not tactical" << endl;
						cout << "I=" << I << endl;
						cout << "J=" << J << endl;
						cout << "r0=" << r0 << endl;
						cout << "r=" << r << endl;
						exit(1); 
						}
					}
				}
			if (FALSE) {
				cout << "I = " << I << " J = " << J << " r = " << r0 << endl;
				}
			Decomp_scheme[I * nb_orbits + J] = r0;
			}
		}
	if (f_v) {
		cout << "Decomp_scheme = " << endl;
		int_matrix_print(Decomp_scheme, nb_orbits, nb_orbits);
		}
	if (f_v) {
		cout << "schreier::get_orbit_decomposition_"
				"scheme_of_graph done" << endl;
		}
}

void schreier::list_elements_as_permutations_vertically(ostream &ost)
{
	A->list_elements_as_permutations_vertically(&gens, ost);
}


void schreier::create_point_list_sorted(
		int *&point_list, int &point_list_length)
{
	int i, j, k, f, l, ff, p;

	point_list_length = 0;
	for (k = 0; k < nb_orbits; k++) {
		point_list_length += orbit_len[k];
		}
	point_list = NEW_int(point_list_length);

	ff = 0;
	for (k = 0; k < nb_orbits; k++) {
		f = orbit_first[k];
		l = orbit_len[k];
		for (j = 0; j < l; j++) {
			i = f + j;
			p = orbit[i];
			point_list[ff + j] = p;
			}
		ff += l;
		}
	if (ff != point_list_length) {
		cout << "schreier::get_schreier_vector_compact "
				"ff != point_list_length" << endl;
		exit(1);
		}
	int_vec_heapsort(point_list, point_list_length);
}

void schreier::shallow_tree_generators(int orbit_idx,
		schreier *&shallow_tree,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fst, len, root, cnt, l;
	int i, j, a, f, o;
	int *Elt1, *Elt2;
	int *candidates;
	int nb_candidates;

	if (f_v) {
		cout << "schreier::shallow_tree_generators " << endl;
		cout << "computing shallow tree for orbit " << orbit_idx
				<< " in action " << A->label << endl;
		}
	fst = orbit_first[orbit_idx];
	len = orbit_len[orbit_idx];
	root = orbit[fst];

	vector_ge *gens;

	candidates = NEW_int(len);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);

	gens = new vector_ge;

	gens->init(A);
	cnt = 0;
	while (TRUE) {
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"iteration " << cnt << ":" << endl;
		}
		schreier *S;

		S = new schreier;
		S->init(A);
		S->init_generators(*gens);
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"iteration " << cnt
					<< " before compute_point_orbit:" << endl;
		}
		S->compute_point_orbit_with_limited_depth(root,
				gens->len, 0 /*verbose_level*/);
		//S->compute_point_orbit(root, 0 /*verbose_level*/);
		l = S->orbit_len[0];
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"iteration " << cnt
					<< " after compute_point_orbit, "
					"found an orbit of length " << l << endl;
		}
		if (l == len) {
			shallow_tree = S;
			break;
		}

		// find an element that belongs to the original orbit
		// (i.e., the bad Schreier tree) but not
		// to the new orbit (the good Schreier tree).
		// When l < len, such an element must exist.
		nb_candidates = 0;
		f = S->orbit_first[0];
		for (i = 0; i < len; i++) {
			a = orbit[fst + i];
			j = S->orbit_inv[a];
			if (j >= f + l) {
				candidates[nb_candidates++] = a;
			}
		}

		if (nb_candidates == 0) {
			cout << "schreier::shallow_tree_generators "
					"did not find element in orbit" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"found " << nb_candidates
					<< " candidates of points outside the orbit" << endl;
		}
		//j = random_integer(nb_candidates);
		j = 0;
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"picking random candidate " << j << " / "
					<< nb_candidates << endl;
		}
		a = candidates[j];
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"found point " << a << " outside of orbit" << endl;
		}
		// our next generator will be the transporter from
		// the root node to the node we just found in the
		// old (bad) Schreier tree:
		transporter_from_orbit_rep_to_point(a,
				o, Elt1, 0 /*verbose_level*/);
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"new generator is:" << endl;
			A->element_print_quick(Elt1, cout);
			int o;

			o = A->element_order(Elt1);
			cout << "The order of the element is: " << o << endl;
		}
		A->element_invert(Elt1, Elt2, 0);
		// append the generator and its inverse to the generating set:
		gens->append(Elt1);
		gens->append(Elt2);
		delete S;
		cnt++;

	}
	if (f_v) {
		cout << "schreier::shallow_tree_generators cnt=" << cnt
				<< " number of generators=" << gens->len << endl;
		cout << "done" << endl;
		}

	FREE_int(candidates);
	delete gens;
	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "schreier::shallow_tree_generators " << endl;
		cout << "done" << endl;
		}
}


schreier_vector *schreier::get_schreier_vector(
	int gen_hdl_first, int nb_gen, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::get_schreier_vector" << endl;
	}
	//int *sv;
	schreier_vector * Schreier_vector;
	int f_trivial_group = FALSE;

	if (nb_gen == 0) {
		f_trivial_group = TRUE;
	}

	Schreier_vector = NEW_OBJECT(schreier_vector);
	//get_schreier_vector_compact(sv, f_trivial_group);
	Schreier_vector->init(gen_hdl_first, nb_gen, NULL, verbose_level - 1);

#if 0
	Schreier_vector->init_from_schreier(this,
			f_trivial_group, verbose_level);
#else
	Schreier_vector->init_shallow_schreier_forest(this,
			f_trivial_group, verbose_level);
#endif

	if (f_v) {
		cout << "schreier::get_schreier_vector done" << endl;
	}
	return Schreier_vector;
}

