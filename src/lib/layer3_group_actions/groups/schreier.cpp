// schreier.cpp
//
// Anton Betten
// December 9, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;
#include "shallow_schreier_ai.h"




namespace orbiter {
namespace layer3_group_actions {
namespace groups {


schreier::schreier()
{
	Record_birth();

	Generators_and_images = NULL;

	Forest = NULL;

	f_print_function = false;
	print_function = NULL;
	print_function_data = NULL;

	f_preferred_choice_function = false;
	preferred_choice_function = NULL;
	preferred_choice_function_data = NULL;
	preferred_choice_function_data2 = 0;
		// for compute_all_point_orbits

}

schreier::~schreier()
{
	Record_death();

	if (Generators_and_images) {
		FREE_OBJECT(Generators_and_images);
	}
	if (Forest) {
		FREE_OBJECT(Forest);
	}
}


void schreier::init_preferred_choice_function(
		void (*preferred_choice_function)(int pt,
				int &pt_pref, schreier *Sch, void *data, int data2,
				int verbose_level),
		void *preferred_choice_function_data,
		int preferred_choice_function_data2,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::init_preferred_choice_function" << endl;
	}
	f_preferred_choice_function = true;
	schreier::preferred_choice_function = preferred_choice_function;
	schreier::preferred_choice_function_data = preferred_choice_function_data;
	schreier::preferred_choice_function_data2 = preferred_choice_function_data2;
	if (f_v) {
		cout << "schreier::init_preferred_choice_function done" << endl;
	}
}



void schreier::init(
		actions::action *A, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::init" << endl;
	}



	if (A->degree > INT_MAX) {
		cout << "schreier::init A->degree > INT_MAX" << endl;
		exit(1);
	}

	Generators_and_images = NEW_OBJECT(generators_and_images);

	if (f_v) {
		cout << "schreier::init "
				"before Generators_and_images->init" << endl;
	}
	Generators_and_images->init(this, A, 0 /* verbose_level */);
	if (f_v) {
		cout << "schreier::init "
				"after Generators_and_images->init" << endl;
	}

	Forest = NEW_OBJECT(other::data_structures::forest);

	if (f_v) {
		cout << "schreier::init "
				"before Forest->init" << endl;
	}
	Forest->init(A->degree, verbose_level - 1);
	if (f_v) {
		cout << "schreier::init "
				"after Forest->init" << endl;
	}

	if (f_v) {
		cout << "schreier::init done" << endl;
	}
}


void schreier::init_images_only(
		actions::action *A,
		int nb_images, int *images,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_images_only" << endl;
	}
	Generators_and_images = NEW_OBJECT(generators_and_images);

	if (f_v) {
		cout << "generators_and_images::init_images_only "
				"before Generators_and_images->init_images_only" << endl;
	}
	Generators_and_images->init_images_only(
			this, A,
			nb_images, images,
			0 /* verbose_level */);
	if (f_v) {
		cout << "generators_and_images::init_images_only "
				"after Generators_and_images->init_images_only" << endl;
	}

	if (f_v) {
		cout << "generators_and_images::init_images_only "
				"before Forest->allocate_tables" << endl;
	}
	Forest->allocate_tables(verbose_level - 2);
	if (f_v) {
		cout << "generators_and_images::init_images_only "
				"after Forest->allocate_tables" << endl;
	}

	if (f_v) {
		cout << "generators_and_images::init_images_only done" << endl;
	}
}








void schreier::extend_orbit(
		int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::extend_orbit" << endl;
	}
	if (f_vv) {
		cout << "schreier::extend_orbit extending orbit "
				<< Forest->nb_orbits - 1 << " of length "
			<< Forest->orbit_len[Forest->nb_orbits - 1] << endl;
	}

	int cur, total0, total, cur_pt;
	int gen_first, i, next_pt, next_pt_loc;
	
	Generators_and_images->append_one(
			elt, verbose_level);

	cur = Forest->orbit_first[Forest->nb_orbits - 1];
	total = total0 = Forest->orbit_first[Forest->nb_orbits];

	while (cur < total) {

		cur_pt = Forest->orbit[cur];

		if (false) {
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

		for (i = gen_first; i < Generators_and_images->gens.len; i++) {

			next_pt = Generators_and_images->get_image(
					cur_pt, i, 0/*verbose_level - 3*/);
				// A->element_image_of(cur_pt, gens.ith(i), false);

			next_pt_loc = Forest->orbit_inv[next_pt];

			if (false) {
				cout << "schreier::extend_orbit generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
			}

			if (next_pt_loc < total) {
				continue;
			}

			if (false) {
				cout << "schreier::extend_orbit new pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
			}

			Forest->swap_points(
					total, next_pt_loc, 0 /*verbose_level*/);
			Forest->prev[total] = cur_pt;
			Forest->label[total] = i;
			total++;

			if (false) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				Forest->print_orbit(cur, total - 1);
			}
		}
		cur++;
	}

	Forest->orbit_first[Forest->nb_orbits] = total;
	Forest->orbit_len[Forest->nb_orbits - 1] =
			total - Forest->orbit_first[Forest->nb_orbits - 1];

	if (f_v) {
		cout << "schreier::extend_orbit orbit extended to length "
				<< Forest->orbit_len[Forest->nb_orbits - 1] << endl;
	}

	if (false) {
		cout << "{ ";
		for (i = Forest->orbit_first[Forest->nb_orbits - 1];
				i < Forest->orbit_first[Forest->nb_orbits]; i++) {
			cout << Forest->orbit[i];
			if (i < Forest->orbit_first[Forest->nb_orbits] - 1) {
				cout << ", ";
			}
		}
		cout << " }" << endl;
	}

	if (f_v) {
		cout << "schreier::extend_orbit done" << endl;
	}
}

void schreier::compute_all_point_orbits(
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits "
				"verbose_level=" << verbose_level << endl;
		cout << "schreier::compute_all_point_orbits "
				"print_interval=" << print_interval << endl;
		cout << "schreier::compute_all_point_orbits action=";
		Generators_and_images->A->print_info();
		//<< " degree=" << degree << endl;
	}

	int pt, pt_loc, cur, pt0;

#if 0
	if (degree > ONE_MILLION) {
		f_vv = false;
	}
#endif

	if (f_v) {
		cout << "schreier::compute_all_point_orbits "
				"before initialize_tables" << endl;
	}
	Forest->initialize_tables(verbose_level - 2);
	if (f_v) {
		cout << "schreier::compute_all_point_orbits "
				"after initialize_tables" << endl;
	}


	for (pt0 = 0, pt = 0; pt < Generators_and_images->degree; pt++) {

		pt_loc = Forest->orbit_inv[pt];

		cur = Forest->orbit_first[Forest->nb_orbits];

		if (pt_loc < cur) {
			continue;
		}

		int pt_pref;

		if (f_preferred_choice_function) {

			if (true) {
				cout << "schreier::compute_all_point_orbits "
						"before preferred_choice_function, pt=" << pt << endl;
			}
			(*preferred_choice_function)(
					pt, pt_pref,
					this,
					preferred_choice_function_data,
					preferred_choice_function_data2,
					verbose_level);
			if (true) {
				cout << "schreier::compute_all_point_orbits "
						"before preferred_choice_function, "
						"pt=" << pt << " pt_pref=" << pt_pref << endl;
			}

			if (Forest->orbit_inv[pt_pref] < cur) {
				cout << "schreier::compute_all_point_orbits "
						"preferred point is already in "
						"some other orbit" << endl;
				exit(1);
			}

		}
		else {
			pt_pref = pt;
		}

		//int f_preferred_choice_function;
		//void (*preferred_choice_function)(int pt, int &pt_pref, void *data);
		//void *preferred_choice_function_data;


		if (false) {
			cout << "schreier::compute_all_point_orbits pt = "
					<< pt << " / " << Generators_and_images->degree
					<< " nb_orbits=" << Forest->nb_orbits
					<< " cur=" << cur
					<< ", computing orbit of "
							"pt_pref=" << pt_pref << endl;
		}
		if (Generators_and_images->degree > ONE_MILLION && (pt - pt0) > 50000) {
			cout << "schreier::compute_all_point_orbits pt = "
					<< pt << " / " << Generators_and_images->degree
					<< " nb_orbits=" << Forest->nb_orbits
					<< " cur=" << cur
					<< ", computing orbit of "
							"pt_pref=" << pt_pref << endl;
			pt0 = pt;
		}
		compute_point_orbit(pt_pref, print_interval, 0 /*verbose_level - 2*/);
	}

	if (f_v) {
		cout << "schreier::compute_all_point_orbits found "
				<< Forest->nb_orbits << " orbits" << endl;
		other::data_structures::tally Cl;

		Cl.init(Forest->orbit_len, Forest->nb_orbits, false, 0);
		cout << "The distribution of orbit lengths is: ";
		Cl.print(false);
	}

}

void schreier::compute_all_point_orbits_with_preferred_labels(
	long int *preferred_labels, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_preferred_labels" << endl;
		//cout << "preferred_labels :";
		//int_vec_print(cout, preferred_labels, degree);
		//cout << endl;
		cout << "degree = " << Generators_and_images->degree << endl;
	}

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_preferred_labels "
				"allocating tables" << endl;
	}

	int pt, pt_loc, cur, a;
	int *labels, *perm, *perm_inv;
	other::data_structures::sorting Sorting;
	int print_interval = 10000;

	Forest->initialize_tables(verbose_level - 2);

	labels = NEW_int(Generators_and_images->degree);
	perm = NEW_int(Generators_and_images->degree);
	perm_inv = NEW_int(Generators_and_images->degree);

	Lint_vec_copy_to_int(preferred_labels, labels, Generators_and_images->degree);
#if 0
	for (i = 0; i < Generators_and_images->degree; i++) {
		labels[i] = preferred_labels[i];
	}
#endif

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_preferred_labels "
				"allocating tables done, "
				"sorting" << endl;
	}
	Sorting.int_vec_sorting_permutation(
			labels, Generators_and_images->degree,
			perm, perm_inv,
			true /* f_increasingly */);

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_preferred_labels "
				"sorting done" << endl;
	}
	
	for (a = 0; a < Generators_and_images->degree; a++) {

		pt = perm_inv[a];
		pt_loc = Forest->orbit_inv[pt];

		cur = Forest->orbit_first[Forest->nb_orbits];

		if (pt_loc < cur) {
			continue;
		}
		// now we need to make sure that the point pt
		// is moved to position cur:
		// actually this is not needed as the
		// function compute_point_orbit does this, too.
		Forest->swap_points(cur, pt_loc, 0 /*verbose_level*/);
		
		if (f_v) {
			cout << "schreier::compute_all_point_orbits_with_preferred_labels "
					"computing orbit of point "
					<< pt << " = " << a << " / " << Generators_and_images->degree << endl;
		}
		compute_point_orbit(
				pt,
				print_interval,
				0 /*verbose_level - 2*/);

		if (f_v) {
			cout << "schreier::compute_all_point_orbits_with_preferred_labels "
					"computing orbit of point "
					<< pt << " done, found an orbit of length "
					<< Forest->orbit_len[Forest->nb_orbits - 1]
					<< " nb_orbits = " << Forest->nb_orbits << endl;
		}
	}

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_preferred_labels "
				"Found " << Forest->nb_orbits << " orbits";
		cout << " on points" << endl;
	}

	FREE_int(labels);
	FREE_int(perm);
	FREE_int(perm_inv);

	if (f_v) {
		cout << "schreier::compute_all_point_orbits_with_preferred_labels "
				"done" << endl;
	}
}

void schreier::compute_all_orbits_on_invariant_subset(
	int len, long int *subset, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, f;
	
	if (f_v) {
		cout << "schreier::compute_all_orbits_on_invariant_subset" << endl;
		cout << "computing orbits on a set of size " << len << endl;
	}

	int print_interval = 10000;

	Forest->initialize_tables(verbose_level - 2);

	for (i = 0; i < len; i++) {
		Forest->move_point_here(i, subset[i]);
	}

	while (true) {

		f = Forest->orbit_first[Forest->nb_orbits];

		if (f >= len) {
			// finished
			break;
		}

		compute_point_orbit(
				Forest->orbit[f],
				print_interval,
				0 /* verbose_level */);

	}
	if (f > len) {
		cout << "schreier::compute_all_orbits_on_invariant_subset "
				"the set is not G-invariant" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "found " << Forest->nb_orbits << " orbits" << endl;
		Forest->print_orbit_length_distribution(cout);
	}
	if (f_v) {
		cout << "schreier::compute_all_orbits_on_invariant_subset done" << endl;
	}
}

void schreier::compute_all_orbits_on_invariant_subset_lint(
	int len, long int *subset, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f;

	if (f_v) {
		cout << "schreier::compute_all_orbits_on_invariant_subset" << endl;
		cout << "computing orbits on a set of size " << len << endl;
	}

	int print_interval = 100000;

	Forest->initialize_tables(verbose_level - 2);

	Forest->move_point_set_lint_to_here(subset, len);
#if 0
	for (i = 0; i < len; i++) {
		Forest->move_point_here(i, subset[i]);
	}
#endif

	while (true) {

		f = Forest->orbit_first[Forest->nb_orbits];

		if (f >= len) {
			break;
		}
		compute_point_orbit(
				Forest->orbit[f],
				print_interval,
				0 /* verbose_level */);
	}
	if (f > len) {
		cout << "schreier::compute_all_orbits_on_invariant_subset "
				"the set is not G-invariant" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "found " << Forest->nb_orbits << " orbits" << endl;
		Forest->print_orbit_length_distribution(cout);
	}
	if (f_v) {
		cout << "schreier::compute_all_orbits_on_invariant_subset done" << endl;
	}
}

void schreier::compute_point_orbit(
		int pt,
		int print_interval,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);


	if (f_v) {
		cout << "schreier::compute_point_orbit" << endl;
		cout << "computing orbit of point " << pt;
		if (Generators_and_images->f_images_only) {
			cout << " in no action, using table of images only" << endl;
		}
		else {
			cout << " in action " << Generators_and_images->A->label << endl;
		}
		cout << "schreier::compute_point_orbit "
				"verbose_level = " << verbose_level << endl;
	}

	int pt_loc, cur, cur_pt, total, i, next_pt;
	int next_pt_loc, total1, cur1;
	//int print_interval = 100000;

	//exit(1);
	pt_loc = Forest->orbit_inv[pt];
	cur = Forest->orbit_first[Forest->nb_orbits];

	if (pt_loc < cur) {
		cout << "schreier::compute_point_orbit "
				"i < orbit_first[nb_orbits]" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "schreier::compute_point_orbit "
				"computing orbit of pt " << pt << " cur=" << cur << endl;
		cout << "schreier::compute_point_orbit "
				"nb_orbits=" << Forest->nb_orbits << endl;
		cout << "schreier::compute_point_orbit "
				"pt_loc=" << pt_loc << endl;
		cout << "schreier::compute_point_orbit "
				"cur=" << cur << endl;
	}

	if (pt_loc > Forest->orbit_first[Forest->nb_orbits]) {

		if (f_v) {
			cout << "schreier::compute_point_orbit before swap_points" << endl;
		}
		Forest->swap_points(
				Forest->orbit_first[Forest->nb_orbits],
				pt_loc,
				0 /* verbose_level */);
		if (f_v) {
			cout << "schreier::compute_point_orbit after swap_points" << endl;
		}
	}

	//orbit_no[orbit_first[nb_orbits]] = nb_orbits;
	total = cur + 1;

	Forest->prev[cur] = -1; //pt;
	Forest->label[cur] = -1;

	while (cur < total) {
		if (f_vv) {
			cout << "schreier::compute_point_orbit "
					"cur=" << cur << " total=" << total << endl;
		}

		cur_pt = Forest->orbit[cur];
		if (f_vv) {
			cout << "schreier::compute_point_orbit "
					"expanding point " << cur_pt << endl;
		}

		for (i = 0; i < Generators_and_images->nb_images /* gens.len*/; i++) {
			if (f_vv) {
				cout << "schreier::compute_point_orbit "
						"expanding point " << cur_pt
						<< " using generator " << i
						<< " / " << Generators_and_images->nb_images << endl;
			}

			if (f_vvv) {
				cout << "schreier::compute_point_orbit "
						"before get_image" << endl;
			}
			next_pt = Generators_and_images->get_image(
					cur_pt, i, verbose_level - 1);
			if (f_vvv) {
				cout << "schreier::compute_point_orbit "
						"after get_image" << endl;
			}

			if (f_vvv) {
				cout << "schreier::compute_point_orbit "
						<< cur_pt
						<< " -> " << next_pt << endl;
			}

				// A->element_image_of(cur_pt, gens.ith(i), false);
			next_pt_loc = Forest->orbit_inv[next_pt];

			if (f_vv) {
				cout << "schreier::compute_point_orbit " << cur_pt
						<< " -> " << next_pt << endl;
			}

			if (next_pt_loc < total) {
				continue;
			}
			if (f_vv) {
				cout << "schreier::compute_point_orbit "
						"expanding: cur_pt = "
						<< cur_pt << " -> next_pt = " << next_pt
						<< " under generator " << i
						<< " / " << Generators_and_images->nb_images << endl;
			}
			Forest->swap_points(total, next_pt_loc, 0 /*verbose_level*/);
			Forest->prev[total] = cur_pt;
			Forest->label[total] = i;

			//orbit_no[total] = nb_orbits;

			total++;

			total1 = total - Forest->orbit_first[Forest->nb_orbits];

			cur1 = cur - Forest->orbit_first[Forest->nb_orbits];

			if ((total1 % print_interval) == 0 ||
					(cur1 > 0 && (cur1 % print_interval) == 0)) {

				cout << "schreier::compute_point_orbit"
						<< " degree = " << Generators_and_images->degree
						<< " length = " << total1
						<< " processed = " << cur1 << " nb_orbits="
						<< Forest->nb_orbits << " cur_pt=" << cur_pt << " next_pt="
						<< next_pt << " orbit_first[nb_orbits]="
						<< Forest->orbit_first[Forest->nb_orbits] << endl;
			}

			if (false) {
				cout << "schreier::compute_point_orbit "
						"cur = " << cur << " total = " << total << endl;
				//print_orbit(cur, total - 1);
			}
		}
		if (f_vv) {
			cout << "schreier::compute_point_orbit "
					"cur_pt " << cur_pt
					<< " has been expanded under all generators" << endl;
			cout << "cur=" << cur << " total = " << total << endl;
		}
		cur++;
	}
	if (f_v) {
		cout << "schreier::compute_point_orbit "
				"orbit is complete, nb_orbits = " << Forest->nb_orbits  + 1 << endl;
	}

	Forest->orbit_first[Forest->nb_orbits + 1] = total;

	Forest->orbit_len[Forest->nb_orbits] =
			total - Forest->orbit_first[Forest->nb_orbits];

	if (f_v) {
		cout << "schreier::compute_point_orbit "
				"found orbit of length " << Forest->orbit_len[Forest->nb_orbits]
				<< " total length " << total
				<< " degree=" << Generators_and_images->degree << endl;
	}

	if (f_vvv) {
		cout << "{ ";
		for (i = Forest->orbit_first[Forest->nb_orbits];
				i < Forest->orbit_first[Forest->nb_orbits + 1]; i++) {
			cout << Forest->orbit[i];
			if (i < Forest->orbit_first[Forest->nb_orbits + 1] - 1) {
				cout << ", ";
			}
		}
		cout << " }" << endl;
	}

	if (false) {
		cout << "coset reps:" << endl;
		for (i = Forest->orbit_first[Forest->nb_orbits];
				i < Forest->orbit_first[Forest->nb_orbits + 1]; i++) {
			cout << i << " : " << endl;
			Generators_and_images->coset_rep(i, verbose_level - 1);
			Generators_and_images->A->Group_element->element_print(
					Generators_and_images->cosetrep, cout);
			cout << "image = " << Forest->orbit[i] << " = "
					<< Generators_and_images->A->Group_element->element_image_of(
							pt, Generators_and_images->cosetrep, 0) << endl;
			cout << endl;

		}
	}

	Forest->nb_orbits++;

	if (f_v) {
		cout << "schreier::compute_point_orbit done" << endl;
	}
}

void schreier::compute_point_orbit_with_limited_depth(
		int pt, int max_depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 5);
	//int f_vvv = false; //(verbose_level >= 3);

	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth" << endl;
		cout << "computing orbit of point " << pt
				<< " in action " << Generators_and_images->A->label << endl;
	}

	int pt_loc, cur, cur_pt, total, i, next_pt;
	int next_pt_loc, total1, cur1;
	int *depth;

	depth = NEW_int(Generators_and_images->A->degree);
	Int_vec_zero(depth, Generators_and_images->A->degree);

	pt_loc = Forest->orbit_inv[pt];
	cur = Forest->orbit_first[Forest->nb_orbits];

	if (pt_loc < cur) {
		cout << "schreier::compute_point_orbit_with_limited_depth "
				"i < orbit_first[nb_orbits]" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth "
				"computing orbit of pt " << pt << endl;
	}

	if (pt_loc > Forest->orbit_first[Forest->nb_orbits]) {
		Forest->swap_points(
				Forest->orbit_first[Forest->nb_orbits], pt_loc,
				0 /*verbose_level*/);
	}

	depth[cur] = 0;
	total = cur + 1;
	while (cur < total) {

		cur_pt = Forest->orbit[cur];

		if (depth[cur] > max_depth) {
			break;
		}
		if (f_vv) {
			cout << "schreier::compute_point_orbit_with_limited_depth cur="
					<< cur << " total=" << total
					<< " applying generators to " << cur_pt << endl;
		}
		for (i = 0; i < Generators_and_images->nb_images /* gens.len */; i++) {

			if (f_vv) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"applying generator "
						<< i << " to point " << cur_pt << endl;
			}

			next_pt = Generators_and_images->get_image(
					cur_pt, i,
				0 /*verbose_level - 5*/); // !!
				// A->element_image_of(cur_pt, gens.ith(i), false);

			next_pt_loc = Forest->orbit_inv[next_pt];

			if (f_vv) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"generator "
						<< i << " maps " << cur_pt
						<< " to " << next_pt << endl;
			}
			if (next_pt_loc < total) {
				continue;
			}
			if (f_vv) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"new pt "
						<< next_pt << " reached from "
						<< cur_pt << " under generator " << i << endl;
			}

			Forest->swap_points(total, next_pt_loc, 0 /*verbose_level*/);

			depth[total] = depth[cur] + 1;

			Forest->prev[total] = cur_pt;
			Forest->label[total] = i;

			total++;

			total1 = total - Forest->orbit_first[Forest->nb_orbits];

			cur1 = cur - Forest->orbit_first[Forest->nb_orbits];

			if ((total1 % 10000) == 0 ||
					(cur1 > 0 && (cur1 % 10000) == 0)) {
				cout << "schreier::compute_point_orbit_with_limited_depth "
						"degree = "
						<< Generators_and_images->A->degree << " length = " << total1
					<< " processed = " << cur1 << " nb_orbits="
					<< Forest->nb_orbits << " cur_pt=" << cur_pt << " next_pt="
					<< next_pt << " orbit_first[nb_orbits]="
					<< Forest->orbit_first[Forest->nb_orbits] << endl;
			}

			if (false) {
				cout << "cur = " << cur << endl;
				cout << "total = " << total << endl;
				Forest->print_orbit(cur, total - 1);
			}
		}
		cur++;
	}

	Forest->orbit_first[Forest->nb_orbits + 1] = total;

	Forest->orbit_len[Forest->nb_orbits] =
			total - Forest->orbit_first[Forest->nb_orbits];

	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth "
				"found an incomplete orbit of length "
				<< Forest->orbit_len[Forest->nb_orbits]
				<< " total length " << total
				<< " degree=" << Generators_and_images->A->degree << endl;
	}
	FREE_int(depth);

	Forest->nb_orbits++;

	if (f_v) {
		cout << "schreier::compute_point_orbit_with_limited_depth done" << endl;
	}
}

void schreier::non_trivial_random_schreier_generator(
		actions::action *A_original,
		int *Elt, int verbose_level)
// computes non trivial random Schreier generator into schreier_gen
// non-trivial is with respect to A_original
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	int f_v4 = false; //(verbose_level >= 4);
	
	if (f_v) {
		cout << "schreier::non_trivial_random_schreier_generator "
				"verbose_level=" << verbose_level << endl;
	}

	int cnt = 0;

	if (Generators_and_images->f_images_only) {
		cout << "schreier::non_trivial_random_schreier_generator "
				"is not allowed if f_images_only is true" << endl;
		exit(1);
	}

	while (true) {
		if (f_v) {
			cout << "schreier::non_trivial_random_schreier_generator "
					"calling random_schreier_generator "
					"(cnt=" << cnt << ")" << endl;
		}
		Generators_and_images->random_schreier_generator(Elt, verbose_level - 1);
		cnt++;
		if (!A_original->Group_element->element_is_one(
				Generators_and_images->schreier_gen, verbose_level - 5)) {
			if (f_vv) {
				cout << "schreier::non_trivial_random_schreier_generator "
						"found a non-trivial random Schreier generator in "
						<< cnt << " trials" << endl;
			}
			if (f_vvv) {
				Generators_and_images->A->Group_element->element_print(Elt, cout);
				cout << endl;
			}
			return;
		}
		else {
			if (f_v4) {
				Generators_and_images->A->Group_element->element_print(Elt, cout);
				cout << endl;
			}
			if (f_vv) {
				cout << "schreier::non_trivial_random_schreier_generator "
						"the element is the identity in action "
						<< A_original->label << ", trying again" << endl;
			}
		}
	}
	if (f_v) {
		cout << "schreier::non_trivial_random_schreier_generator done" << endl;
	}
}


void schreier::orbits_on_invariant_subset_fast(
	int len, int *subset, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast "
			"computing orbits on invariant subset "
			"of size " << len;
		if (Generators_and_images->f_images_only) {
			cout << " using images only" << endl;
		}
		else {
			cout << " in action " << Generators_and_images->A->label << endl;
			//A->print_info();
		}
	}

	int i, p, j;
	int print_interval = 100000;

	for (i = 0; i < len; i++) {

		p = subset[i];
		j = Forest->orbit_inv[p];

		if (j >= Forest->orbit_first[Forest->nb_orbits]) {
			if (f_v) {
				cout << "schreier::orbits_on_invariant_subset_fast "
						"computing orbit no " << Forest->nb_orbits << endl;
			}
			compute_point_orbit(p, print_interval, 0);
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
		print_tables(cout, false);
		exit(1);
		}
#endif
	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast "
			"found " << Forest->nb_orbits
			<< " orbits on the invariant subset of size " << len << endl;
	}
}

void schreier::orbits_on_invariant_subset_fast_lint(
	int len, long int *subset, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast_lint "
			"computing orbits on invariant subset "
			"of size " << len;
		if (Generators_and_images->f_images_only) {
			cout << " using images only" << endl;
		}
		else {
			cout << " in action " << Generators_and_images->A->label << endl;
			//A->print_info();
		}
	}

	int i, p, j;
	int print_interval = 100000;

	for (i = 0; i < len; i++) {

		p = subset[i];

		j = Forest->orbit_inv[p];

		if (j >= Forest->orbit_first[Forest->nb_orbits]) {
			if (f_v) {
				cout << "schreier::orbits_on_invariant_subset_fast_lint "
						"computing orbit no " << Forest->nb_orbits << endl;
			}
			compute_point_orbit(p, print_interval, 0);
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
		print_tables(cout, false);
		exit(1);
	}
#endif

	if (f_v) {
		cout << "schreier::orbits_on_invariant_subset_fast_lint "
			"found " << Forest->nb_orbits
			<< " orbits on the invariant subset of size " << len << endl;
	}
}

void schreier::orbits_on_invariant_subset(
		int len, int *subset,
	int &nb_orbits_on_subset, 
	int *&orbit_perm, int *&orbit_perm_inv)
{
	int i, j, a, pos;
	int print_interval = 100000;
	
	compute_all_point_orbits(print_interval, 0);

	nb_orbits_on_subset = 0;

	orbit_perm = NEW_int(Forest->nb_orbits);
	orbit_perm_inv = NEW_int(Forest->nb_orbits);

	Int_vec_mone(orbit_perm_inv, Forest->nb_orbits);
#if 0
	for (i = 0; i < Forest->nb_orbits; i++) {
		orbit_perm_inv[i] = -1;
	}
#endif
	for (i = 0; i < Forest->nb_orbits; i++) {
		j = Forest->orbit_first[i];
		a = Forest->orbit[j];
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
	for (i = 0; i < Forest->nb_orbits; i++) {
		if (orbit_perm_inv[i] == -1) {
			orbit_perm[j] = i;
			orbit_perm_inv[i] = j;
			j++;
		}
	}
}

strong_generators *schreier::stabilizer_any_point_plus_cosets(
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
	int pt, data_structures_groups::vector_ge *&cosets,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets" << endl;
	}
	if (Generators_and_images->f_images_only) {
		cout << "schreier::stabilizer_any_point_plus_cosets is not "
				"allowed if Generators_and_images->f_images_only is true" << endl;
		exit(1);
	}

	strong_generators *gens0;
	strong_generators *gens;
	int orbit_index;
	int orbit_index1;
	int *transporter;
	int *transporter1;
	int i, fst, len;

	cosets = NEW_OBJECT(data_structures_groups::vector_ge);
	cosets->init(Generators_and_images->A, verbose_level - 2);
	transporter = NEW_int(Generators_and_images->A->elt_size_in_int);
	transporter1 = NEW_int(Generators_and_images->A->elt_size_in_int);
	
	orbit_index = Forest->orbit_number(pt);

	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets "
				"before stabilizer_orbit_rep" << endl;
	}
	gens0 = stabilizer_orbit_rep(
			default_action,
		full_group_order, orbit_index,
		0 /* verbose_level */);

	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets "
				"after stabilizer_orbit_rep" << endl;
	}

	fst = Forest->orbit_first[orbit_index];
	len = Forest->orbit_len[orbit_index];
	cosets->allocate(len, verbose_level - 2);

	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets "
				"before Generators_and_images->transporter_from_point_to_orbit_rep" << endl;
	}
	Generators_and_images->transporter_from_point_to_orbit_rep(
			pt,
			orbit_index1, transporter,
			0 /* verbose_level */);
	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets "
				"after Generators_and_images->transporter_from_point_to_orbit_rep" << endl;
	}

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

	gens->init_generators_for_the_conjugate_group_aGav(
			gens0,
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

		Generators_and_images->transporter_from_orbit_rep_to_point(
				Forest->orbit[fst + i],
				orbit_index1, transporter1,
				0 /* verbose_level */);

		Generators_and_images->A->Group_element->element_mult(
				transporter, transporter1, cosets->ith(i), 0);
	}
	if (f_v) {
		cout << "schreier::stabilizer_any_point_plus_cosets "
				"computing coset representatives done" << endl;
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
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
	int pt,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::stabilizer_any_point" << endl;
	}

	if (Generators_and_images->f_images_only) {
		cout << "schreier::stabilizer_any_point is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}

	strong_generators *gens0;
	strong_generators *gens;
	int orbit_index;
	int orbit_index1;
	int *transporter;

	transporter = NEW_int(Generators_and_images->A->elt_size_in_int);
	
	orbit_index = Forest->orbit_number(pt);

	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"before stabilizer_orbit_rep" << endl;
	}
	gens0 = stabilizer_orbit_rep(
			default_action,
		full_group_order, orbit_index,
		0 /* verbose_level */);

	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"after stabilizer_orbit_rep" << endl;
	}

	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"before Generators_and_images->transporter_from_point_to_orbit_rep" << endl;
	}

	Generators_and_images->transporter_from_point_to_orbit_rep(
			pt,
			orbit_index1, transporter,
			0 /* verbose_level */);
	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"after Generators_and_images->transporter_from_point_to_orbit_rep" << endl;
	}

	if (orbit_index1 != orbit_index) {
		cout << "schreier::stabilizer_any_point "
				"orbit_index1 != orbit_index" << endl;
		exit(1);
	}
	
	gens = NEW_OBJECT(strong_generators);

	
	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"before gens->init_generators_for_the_conjugate_group_aGav" << endl;
	}
	gens->init_generators_for_the_conjugate_group_aGav(
			gens0,
			transporter,
			verbose_level);
	if (f_v) {
		cout << "schreier::stabilizer_any_point "
				"after gens->init_generators_for_the_conjugate_group_aGav" << endl;
	}

	FREE_int(transporter);
	
	if (f_v) {
		cout << "schreier::stabilizer_any_point done" << endl;
	}
	FREE_OBJECT(gens0);
	return gens;
}


data_structures_groups::set_and_stabilizer
	*schreier::get_orbit_rep(
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
		int orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (Generators_and_images->f_images_only) {
		cout << "schreier::get_orbit_rep is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}

	data_structures_groups::set_and_stabilizer *SaS;
	strong_generators *SG;
	long int *Set;

	SaS = NEW_OBJECT(data_structures_groups::set_and_stabilizer);

	if (f_v) {
		cout << "schreier::get_orbit_rep "
				"before stabilizer_orbit_rep" << endl;
	}

	SG = stabilizer_orbit_rep(
			default_action,
			full_group_order, orbit_idx,
			verbose_level);

	if (f_v) {
		cout << "schreier::get_orbit_rep "
				"after stabilizer_orbit_rep" << endl;
	}

	Set = NEW_lint(1);

	Set[0] = Forest->orbit[Forest->orbit_first[orbit_idx]];

	if (f_v) {
		cout << "schreier::get_orbit_rep "
				"before SaS->init_everything" << endl;
	}

	SaS->init_everything(
			default_action,
			Generators_and_images->A,
			Set, 1 /* set_sz */,
			SG, verbose_level);

	if (f_v) {
		cout << "schreier::get_orbit_rep "
				"after SaS->init_everything" << endl;
	}
	return SaS;
}

void schreier::get_orbit_rep_to(
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
		int orbit_idx,
		data_structures_groups::set_and_stabilizer *Rep,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (f_v) {
		cout << "schreier::get_orbit_rep_to" << endl;
	}

	//set_and_stabilizer *SaS;
	strong_generators *SG;
	long int *Set;

	if (Generators_and_images->f_images_only) {
		cout << "schreier::get_orbit_rep_to is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	//SaS = NEW_OBJECT(set_and_stabilizer);
	if (f_v) {
		cout << "schreier::get_orbit_rep_to "
				"before stabilizer_orbit_rep" << endl;
	}

	SG = stabilizer_orbit_rep(
			default_action,
			full_group_order, orbit_idx,
			verbose_level - 2);

	if (f_v) {
		cout << "schreier::get_orbit_rep_to "
				"after stabilizer_orbit_rep" << endl;
	}
	Set = NEW_lint(1);
	Set[0] = Forest->orbit[Forest->orbit_first[orbit_idx]];
	if (f_v) {
		cout << "schreier::get_orbit_rep_to "
				"before Rep->init_everything" << endl;
	}

	Rep->init_everything(
			default_action,
			Generators_and_images->A,
			Set, 1 /* set_sz */,
			SG, verbose_level - 2);

	if (f_v) {
		cout << "schreier::get_orbit_rep_to "
				"after Rep->init_everything" << endl;
	}
	//return SaS;
	if (f_v) {
		cout << "schreier::get_orbit_rep_to done" << endl;
	}
}


strong_generators *schreier::stabilizer_orbit_rep(
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &full_group_order,
		int orbit_idx,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep" << endl;
		cout << "default_action=" << default_action->label << endl;
		cout << "orbit_idx=" << orbit_idx << endl;
	}

	strong_generators *gens;
	sims *Stab;

	if (Generators_and_images->f_images_only) {
		cout << "schreier::stabilizer_orbit_rep is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep "
				"before point_stabilizer" << endl;
	}

	point_stabilizer(
			default_action, full_group_order,
		Stab, orbit_idx,
		verbose_level - 2);

	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep "
				"after point_stabilizer" << endl;
	}

	algebra::ring_theory::longinteger_object stab_order;

	Stab->group_order(stab_order);
	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep "
				"found a stabilizer group "
				"of order " << stab_order << endl;
	}
	
	gens = NEW_OBJECT(strong_generators);
	gens->init(Generators_and_images->A);
	gens->init_from_sims(Stab, verbose_level);

	FREE_OBJECT(Stab);
	if (f_v) {
		cout << "schreier::stabilizer_orbit_rep done" << endl;
	}
	return gens;
}

void schreier::point_stabilizer(
		actions::action *default_action,
		algebra::ring_theory::longinteger_object &go,
		groups::sims *&Stab,
		int orbit_no,
	int verbose_level)
// this function allocates a sims structure into Stab.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int f_v4 = (verbose_level >= 4);
	
	
	if (f_v) {
		cout << "schreier::point_stabilizer" << endl;
		cout << "default_action=" << default_action->label << endl;
		cout << "default_action->elt_size_in_int="
				<< default_action->elt_size_in_int << endl;
	}
	if (Generators_and_images->f_images_only) {
		cout << "schreier::point_stabilizer is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}

	algebra::ring_theory::longinteger_object cur_go, target_go;
	algebra::ring_theory::longinteger_domain D;
	int len, r, cnt = 0, f_added, drop_out_level, image;
	int *residue;
	int *Elt;
	int *Elt1;

	Elt = NEW_int(default_action->elt_size_in_int);
	Elt1 = NEW_int(Generators_and_images->A->elt_size_in_int);

		// the schreier generator must be computed in the action
		// attached to the this instance of the schreier class
		// the assumption is that this could be a longer representation,
		// and that the representation of the element in default_action
		// is at the beginning of the longer representation.
		// This is true for permutation_representation.


	residue = NEW_int(default_action->elt_size_in_int);

	Stab = NEW_OBJECT(sims);

	if (f_v) {
		cout << "schreier::point_stabilizer "
				"computing stabilizer of representative of orbit "
			<< orbit_no << " inside a group of "
					"order " << go << " in action ";
		default_action->print_info();
		cout << endl;
	}

	len = Forest->orbit_len[orbit_no];

	D.integral_division_by_int(go, len, target_go, r);

	if (r) {	
		cout << "schreier::point_stabilizer "
				"orbit length does not divide group order" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "schreier::point_stabilizer "
				"expecting group of order " << target_go << endl;
	}
	
	if (f_vv) {
		cout << "schreier::point_stabilizer "
				"before Stab->init" << endl;
	}
	Stab->init(default_action, verbose_level);
	if (f_vv) {
		cout << "schreier::point_stabilizer "
				"after Stab->init" << endl;
	}

	if (f_vv) {
		cout << "schreier::point_stabilizer "
				"before Stab->init_trivial_group" << endl;
	}
	Stab->init_trivial_group(verbose_level - 1);
	if (f_vv) {
		cout << "schreier::point_stabilizer "
				"after Stab->init_trivial_group" << endl;
	}


	if (f_vv) {
		cout << "schreier::point_stabilizer "
				"Stab->my_base_len=" << Stab->my_base_len << endl;
	}

	while (true) {

		if (f_vv) {
			cout << "schreier::point_stabilizer cnt=" << cnt << endl;
		}
		if (f_vv) {
			cout << "schreier::point_stabilizer "
					"Stab->my_base_len=" << Stab->my_base_len << endl;
		}

		Stab->group_order(cur_go);
		if (D.compare(cur_go, target_go) == 0) {
			break;
		}
		if (cnt % 2 || Stab->nb_gen[0] == 0) {
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"before random_schreier_generator_ith_orbit" << endl;
				cout << "schreier::point_stabilizer "
						"Stab->my_base_len=" << Stab->my_base_len << endl;
			}
			Generators_and_images->random_schreier_generator_ith_orbit(
					Elt1, orbit_no,
					verbose_level - 5);

			// this creates a schreier generator in action A,
			// not in default_action


			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"after random_schreier_generator_ith_orbit" << endl;
				cout << "schreier::point_stabilizer "
						"Stab->my_base_len=" << Stab->my_base_len << endl;
			}

			// and now we copy over the part of Elt1 that belongs to default_action:

			default_action->Group_element->element_move(Elt1, Elt, 0);


			if (f_vvv) {
				cout << "schreier::point_stabilizer "
						"random Schreier generator from the orbit:" << endl;
				default_action->Group_element->element_print_quick(Elt, cout);
			}
		}
		else {
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"before Stab->random_schreier_generator" << endl;
				cout << "schreier::point_stabilizer "
						"Stab->my_base_len=" << Stab->my_base_len << endl;
			}
			Stab->random_schreier_generator(Elt, verbose_level - 5);
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"after Stab->random_schreier_generator" << endl;
				cout << "schreier::point_stabilizer "
						"Stab->my_base_len=" << Stab->my_base_len << endl;
			}
			if (f_v4) {
				cout << "schreier::point_stabilizer "
						"random schreier generator from sims:" << endl;
				default_action->Group_element->element_print_quick(Elt, cout);
			}
		}



		if (f_vv) {
			cout << "schreier::point_stabilizer "
					"before Stab->strip" << endl;
		}
		if (f_vv) {
			cout << "schreier::point_stabilizer "
					"Stab->my_base_len=" << Stab->my_base_len << endl;
		}
		if (Stab->strip(
				Elt, residue,
				drop_out_level, image,
				verbose_level - 5)) {
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"element strips through" << endl;
				if (f_v4) {
					cout << "schreier::point_stabilizer residue:" << endl;
					Generators_and_images->A->Group_element->element_print_quick(residue, cout);
					cout << endl;
				}
			}
			f_added = false;
		}
		else {
			f_added = true;
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"element needs to be inserted at level = "
					<< drop_out_level << " with image " << image << endl;
				if (false) {
					Generators_and_images->A->Group_element->element_print_quick(residue, cout);
					cout  << endl;
				}
			}
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"before Stab->add_generator_at_level, "
						"drop_out_level=" << drop_out_level << endl;
			}
			Stab->add_generator_at_level(
					residue,
					drop_out_level,
					verbose_level - 1);
			if (f_vv) {
				cout << "schreier::point_stabilizer "
						"after Stab->add_generator_at_level, "
						"drop_out_level=" << drop_out_level << endl;
			}
		}
		if (f_vv) {
			cout << "schreier::point_stabilizer "
					"before Stab->group_order" << endl;
		}
		Stab->group_order_verbose(cur_go, verbose_level);

		if ((f_vv && f_added) || f_vvv) {
			cout << "schreier::point_stabilizer "
					"iteration " << cnt
				<< " the new group order is " << cur_go
				<< " expecting a group of order "
				<< target_go << endl;
		}
		cnt++;
	}
	FREE_int(Elt);
	FREE_int(Elt1);
	FREE_int(residue);

	if (f_v) {
		cout << "schreier::point_stabilizer finished" << endl;
	}
}


void schreier::shallow_tree_generators(
		int orbit_idx,
		int f_randomized,
		schreier *&shallow_tree,
		int verbose_level)
// Seress algorithm: double the cube
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::shallow_tree_generators " << endl;
		cout << "computing shallow tree for orbit " << orbit_idx
				<< " in action " << Generators_and_images->A->label << endl;
	}

	int fst, len, root, cnt, l;
	int i, j, a, f, o;
	int *Elt1, *Elt2;
	int *candidates;
	int nb_candidates;
	other::orbiter_kernel_system::os_interface Os;

	fst = Forest->orbit_first[orbit_idx];
	len = Forest->orbit_len[orbit_idx];
	root = Forest->orbit[fst];

	data_structures_groups::vector_ge *gens;

	candidates = NEW_int(len);

	Elt1 = NEW_int(Generators_and_images->A->elt_size_in_int);
	Elt2 = NEW_int(Generators_and_images->A->elt_size_in_int);

	gens = NEW_OBJECT(data_structures_groups::vector_ge);

	gens->init(Generators_and_images->A, verbose_level - 2);

	cnt = 0;

	while (true) {
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"iteration " << cnt << ":" << endl;
		}
		schreier *S;

		S = NEW_OBJECT(schreier);
		S->init(Generators_and_images->A, verbose_level - 2);
		S->Generators_and_images->init_generators(*gens, verbose_level - 2);
		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"iteration " << cnt
					<< " before compute_point_orbit:" << endl;
		}
		S->compute_point_orbit_with_limited_depth(
				root,
				gens->len,
				0 /*verbose_level*/);
		//S->compute_point_orbit(root, 0 /*verbose_level*/);

		l = S->Forest->orbit_len[0];

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

		f = S->Forest->orbit_first[0];
		for (i = 0; i < len; i++) {
			a = Forest->orbit[fst + i];
			j = S->Forest->orbit_inv[a];
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
		if (f_randomized) {
			j = Os.random_integer(nb_candidates);
		}
		else {
			j = 0;
		}

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

		Generators_and_images->transporter_from_orbit_rep_to_point(
				a,
				o, Elt1,
				0 /*verbose_level*/);

		if (f_v) {
			cout << "schreier::shallow_tree_generators "
					"new generator is:" << endl;
			Generators_and_images->A->Group_element->element_print_quick(Elt1, cout);
			int o;

			o = Generators_and_images->A->Group_element->element_order(Elt1);
			cout << "The order of the element is: " << o << endl;
		}
		Generators_and_images->A->Group_element->element_invert(Elt1, Elt2, 0);
		// append the generator and its inverse to the generating set:
		gens->append(Elt1, verbose_level - 2);
		gens->append(Elt2, verbose_level - 2);
		FREE_OBJECT(S);
		cnt++;

	}
	if (f_v) {
		cout << "schreier::shallow_tree_generators cnt=" << cnt
				<< " number of generators=" << gens->len << endl;
		cout << "done" << endl;
	}

	FREE_int(candidates);
	FREE_OBJECT(gens);
	FREE_int(Elt1);
	FREE_int(Elt2);

	if (f_v) {
		cout << "schreier::shallow_tree_generators " << endl;
		cout << "done" << endl;
	}
}

#if 0
#if 0
schreier_vector *schreier::get_schreier_vector(
	int gen_hdl_first, int nb_gen, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::get_schreier_vector" << endl;
	}
	//int *sv;
	schreier_vector * Schreier_vector;
	int f_trivial_group = false;

	if (nb_gen == 0) {
		f_trivial_group = true;
	}

	Schreier_vector = NEW_OBJECT(schreier_vector);
	//get_schreier_vector_compact(sv, f_trivial_group);
	Schreier_vector->init(gen_hdl_first, nb_gen, NULL, verbose_level - 1);

#if 1
	Schreier_vector->init_from_schreier(this,
			f_trivial_group, verbose_level);
#else
	Schreier_vector->init_shallow_schreier_forest(this,
			f_trivial_group, verbose_level);
#endif

	if (nb_gen) {
		Schreier_vector->init_local_generators(
				&gens,
				0 /*verbose_level */);
	}

	if (f_v) {
		cout << "schreier::get_schreier_vector done" << endl;
	}
	return Schreier_vector;
}
#else
data_structures_groups::schreier_vector *schreier::get_schreier_vector(
		int gen_hdl_first, int nb_gen,
		enum shallow_schreier_tree_strategy
			Shallow_schreier_tree_strategy,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::get_schreier_vector" << endl;
	}
//int *sv;
	data_structures_groups::schreier_vector * Schreier_vector;
	int f_trivial_group = false;

	if (nb_gen == 0) {
		f_trivial_group = true;
	}

	Schreier_vector = NEW_OBJECT(data_structures_groups::schreier_vector);
	//get_schreier_vector_compact(sv, f_trivial_group);
	Schreier_vector->init(gen_hdl_first, nb_gen, NULL, verbose_level - 1);



	switch (Shallow_schreier_tree_strategy) {

	case shallow_schreier_tree_standard :

		if (f_v) {
			cout << "schreier::get_schreier_vector "
					"shallow_schreier_tree_standard" << endl;
		}

		Schreier_vector->init_from_schreier(this, f_trivial_group, verbose_level);

		if (nb_gen) {
			Schreier_vector->init_local_generators(&Generators_and_images->gens, 0 /*verbose_level */);
		}

		break;


	case shallow_schreier_tree_Seress_deterministic:

		if (f_v) {
			cout << "schreier::get_schreier_vector "
					"shallow_schreier_tree_Seress" << endl;
		}

		Schreier_vector->init_shallow_schreier_forest(
				this,
			f_trivial_group,
			false /* f_randomized*/,
			verbose_level);
		if (f_v) {
			cout << "schreier::get_schreier_vector after "
					"Schreier_vector->init_shallow_schreier_forest, nb "
					"local gens in Schreier_vector = "
					<< Schreier_vector->local_gens->len << endl;
			cout << "f_has_local_generators=" <<
					Schreier_vector->f_has_local_generators << endl;
		}
		//Schreier_vector->init_from_schreier(this, f_trivial_group, verbose_level);

		break;


	case shallow_schreier_tree_Seress_randomized:

		if (f_v) {
			cout << "schreier::get_schreier_vector "
					"shallow_schreier_tree_Seress" << endl;
		}

		Schreier_vector->init_shallow_schreier_forest(this,
			f_trivial_group,
			true /* f_randomized*/,
			verbose_level);
		if (f_v) {
			cout << "schreier::get_schreier_vector after Schreier_vector->"
					"init_shallow_schreier_forest, nb local gens in "
					"Schreier_vector = " << Schreier_vector->local_gens->len
					<< endl;
			cout << "f_has_local_generators="
					<< Schreier_vector->f_has_local_generators << endl;
		}
		//Schreier_vector->init_from_schreier(this, f_trivial_group, verbose_level);

		break;


	case shallow_schreier_tree_Sajeeb:

		if (f_v) {
			cout << "schreier::get_schreier_vector "
					"shallow_schreier_tree_Sajeeb" << endl;
		}

		shallow_schreier_ai shallow_tree;
		shallow_tree.generate_shallow_tree(*this, verbose_level);
		shallow_tree.get_degree_sequence(*this, verbose_level);
		shallow_tree.print_degree_sequence();



		Schreier_vector->init_from_schreier(
				this, f_trivial_group, verbose_level);

		if (nb_gen) {
			Schreier_vector->init_local_generators(
					&Generators_and_images->gens, 0 /*verbose_level */);
		}

		break;


	}



	if (f_v) {
		cout << "nb_times_image_of_called="
				<< Generators_and_images->A->ptr->nb_times_image_of_called << endl;
	}

	if (f_v) {
		cout << "schreier::get_schreier_vector done" << endl;
	}
	return Schreier_vector;
}
#endif
#endif


data_structures_groups::schreier_vector *schreier::get_schreier_vector(
		int gen_hdl_first, int nb_gen,
		//enum shallow_schreier_tree_strategy
		//	Shallow_schreier_tree_strategy,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier::get_schreier_vector" << endl;
	}

	//int *sv;

	data_structures_groups::schreier_vector * Schreier_vector;
	int f_trivial_group = false;

	if (nb_gen == 0) {
		f_trivial_group = true;
	}

	Schreier_vector = NEW_OBJECT(data_structures_groups::schreier_vector);

	//get_schreier_vector_compact(sv, f_trivial_group);

	Schreier_vector->init(gen_hdl_first, nb_gen, NULL, verbose_level - 1);



	if (f_v) {
		cout << "schreier::get_schreier_vector "
				"before Schreier_vector->init_from_schreier" << endl;
	}

	Schreier_vector->init_from_schreier(this, f_trivial_group, verbose_level);

	if (f_v) {
		cout << "schreier::get_schreier_vector "
				"after Schreier_vector->init_from_schreier" << endl;
	}


	if (nb_gen) {
		Schreier_vector->init_local_generators(&Generators_and_images->gens, 0 /*verbose_level */);
	}

	return Schreier_vector;

}


void schreier::compute_orbit_invariant(
		int *&orbit_invariant,
		int (*compute_orbit_invariant_callback)(schreier *Sch,
				int orbit_idx, void *data, int verbose_level),
		void *compute_orbit_invariant_data,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int orbit_idx;

	if (f_v) {
		cout << "schreier::compute_orbit_invariant" << endl;
	}

	orbit_invariant = NEW_int(Forest->nb_orbits);

	for (orbit_idx = 0; orbit_idx < Forest->nb_orbits; orbit_idx++) {

		orbit_invariant[orbit_idx] = (*compute_orbit_invariant_callback)
				(this, orbit_idx, compute_orbit_invariant_data, verbose_level - 2);

	}
	if (f_v) {
		cout << "schreier::compute_orbit_invariant done" << endl;
	}
}



}}}

