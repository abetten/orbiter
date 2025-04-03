/*
 * generators_and_images.cpp
 *
 *  Created on: Mar 27, 2025
 *      Author: betten
 */








#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace groups {


generators_and_images::generators_and_images()
{
	Record_birth();

	Schreier = NULL;
	A = NULL;
	f_images_only = false;
	degree = 0;
	nb_images = 0;
	images = NULL;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	schreier_gen = NULL;
	schreier_gen1 = NULL;
	cosetrep = NULL;
	cosetrep_tmp = NULL;

}


generators_and_images::~generators_and_images()
{
	Record_death();

	if (A) {
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


void generators_and_images::init(
		schreier *Schreier,
		actions::action *A,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init" << endl;
	}

	generators_and_images::Schreier = Schreier;
	generators_and_images::A = A;
	degree = A->degree;
	//allocate_tables();
	gens.init(A, verbose_level - 2);
	gens_inv.init(A, verbose_level - 2);
	//initialize_tables();
	init2();

}

void generators_and_images::init2()
{
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	schreier_gen = NEW_int(A->elt_size_in_int);
	schreier_gen1 = NEW_int(A->elt_size_in_int);
	cosetrep = NEW_int(A->elt_size_in_int);
	cosetrep_tmp = NEW_int(A->elt_size_in_int);
}

void generators_and_images::delete_images()
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

void generators_and_images::init_images(
		int nb_images, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i;

	if (f_v) {
		cout << "generators_and_images::init_images" << endl;
	}
#if 0
	if (A == NULL) {
		cout << "generators_and_images::init_images action is NULL" << endl;
		exit(1);
		}
#endif
	delete_images();
	generators_and_images::nb_images = nb_images;
	images = NEW_pint(nb_images);
	for (i = 0; i < nb_images; i++) {
		if (f_v) {
			cout << "generators_and_images::init_images "
					"allocating images[i], i=" << i << endl;
		}
		images[i] = NEW_int(2 * degree);
		Int_vec_mone(images[i], 2 * degree);
#if 0
		for (j = 0; j < 2 * degree; j++) {
			images[i][j] = -1;
		}
#endif
	}
	if (f_v) {
		cout << "generators_and_images::init_images done" << endl;
	}
}

void generators_and_images::init_images_only(
		schreier *Schreier,
		actions::action *A,
		int nb_images,
		int *images, int verbose_level)
// images[nb_images * A->degree]
{
	int f_v = (verbose_level >= 1);
	int i;
	combinatorics::other_combinatorics::combinatorics_domain Combi;

	if (f_v) {
		cout << "generators_and_images::init_images_only" << endl;
	}
	delete_images();
	f_images_only = true;
	generators_and_images::Schreier = Schreier;
	generators_and_images::A = A;
	generators_and_images::degree = A->degree;
	generators_and_images::nb_images = nb_images;
	generators_and_images::images = NEW_pint(nb_images);
	for (i = 0; i < nb_images; i++) {
		if (f_v) {
			cout << "generators_and_images::init_images_only "
					"allocating images[i], i=" << i << endl;
		}
		generators_and_images::images[i] = NEW_int(2 * degree);
		Int_vec_copy(
				images + i * degree,
				generators_and_images::images[i], degree);
		Combi.Permutations->perm_inverse(
				generators_and_images::images[i],
				generators_and_images::images[i] + degree,
				degree);
	}
	if (f_v) {
		cout << "generators_and_images::init_images_only done" << endl;
	}
}


#if 0
void generators_and_images::init_images_recycle(
		int nb_images,
		int **old_images, int idx_deleted_generator,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i, j;

	if (f_v) {
		cout << "generators_and_images::init_images_recycle" << endl;
	}
#if 0
	if (A == NULL) {
		cout << "generators_and_images::init_images_recycle action is NULL" << endl;
		exit(1);
	}
#endif
	delete_images();
	generators_and_images::nb_images = nb_images;
	images = NEW_pint(nb_images);
	for (i = 0; i < nb_images; i++) {
		if (f_v) {
			cout << "generators_and_images::init_images_recycle "
					"allocating images[i], i=" << i << endl;
		}
		images[i] = NEW_int(2 * degree);
		if (i == idx_deleted_generator) {
			for (j = 0; j < 2 * degree; j++) {
				images[i][j] = -1;
			}
		}
		else {
			if (old_images[i]) {
				Int_vec_copy(old_images[i], images[i], 2 * degree);
			}
			else {
				for (j = 0; j < 2 * degree; j++) {
					images[i][j] = -1;
				}
			}
		}
	}

	if (f_v) {
		cout << "generators_and_images::init_images_recycle done" << endl;
	}
}
#endif



void generators_and_images::init_images_recycle(
		int nb_images,
		int **old_images, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int i;

	if (f_v) {
		cout << "generators_and_images::init_images_recycle" << endl;
	}
#if 0
	if (A == NULL) {
		cout << "generators_and_images::init_images_recycle action is NULL" << endl;
		exit(1);
		}
#endif
	delete_images();
	generators_and_images::nb_images = nb_images;
	images = NEW_pint(nb_images);
	for (i = 0; i < nb_images; i++) {
		if (f_v) {
			cout << "generators_and_images::init_images_recycle allocating "
					"images[i], i=" << i << endl;
		}
		images[i] = NEW_int(2 * degree);
		if (old_images[i]) {
			Int_vec_copy(old_images[i], images[i], 2 * degree);
		}
		else {
			Int_vec_mone(images[i], 2 * degree);
#if 0
			for (j = 0; j < 2 * degree; j++) {
				images[i][j] = -1;
			}
#endif
		}
	}

	if (f_v) {
		cout << "generators_and_images::init_images_recycle done" << endl;
	}
}



void generators_and_images::images_append(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::images_append" << endl;
	}

	int **new_images = NEW_pint(nb_images + 1);
	int i;

	new_images[nb_images] = NEW_int(2 * degree);

	Int_vec_mone(new_images[nb_images], 2 * degree);
#if 0
	for (j = 0; j < 2 * degree; j++) {
		new_images[nb_images][j] = -1;
	}
#endif

	for (i = 0; i < nb_images; i++) {
		new_images[i] = images[i];
	}
	FREE_pint(images);
	images = new_images;
	nb_images++;

	if (f_v) {
		cout << "generators_and_images::images_append done" << endl;
	}
}

void generators_and_images::init_single_generator(
		int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_single_generator" << endl;
	}
	init_generators(1, elt, verbose_level);
	if (f_v) {
		cout << "generators_and_images::init_single_generator done" << endl;
	}
}

void generators_and_images::init_generators(
		data_structures_groups::vector_ge &generators,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_generators" << endl;
	}
	if (generators.len) {
		init_generators(generators.len,
				generators.ith(0), verbose_level);
	}
	else {
		init_generators(generators.len, NULL, verbose_level);
	}
	if (f_v) {
		cout << "generators_and_images::init_generators done" << endl;
	}
}

void generators_and_images::init_generators(
		int nb, int *elt,
		int verbose_level)
// elt must point to nb * A->elt_size_in_int
// int's that are
// group elements in int format
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_generators nb=" << nb << endl;
	}

	gens.allocate(nb, verbose_level - 2);
	gens_inv.allocate(nb, verbose_level - 2);
	for (i = 0; i < nb; i++) {
		if (f_v) {
			cout << "generators_and_images::init_generators i = " << i << " / " << nb << endl;
		}

		gens.copy_in(i, elt + i * A->elt_size_in_int);

		A->Group_element->element_invert(
				elt + i * A->elt_size_in_int,
				gens_inv.ith(i), 0);
	}
	if (f_v) {
		cout << "generators_and_images::init_generators before init_images" << endl;
	}
	init_images(nb, 0 /* verbose_level */);
	if (f_v) {
		cout << "generators_and_images::init_generators after init_images" << endl;
	}
	if (f_v) {
		cout << "generators_and_images::init_generators done" << endl;
	}
}




#if 0
void generators_and_images::init_generators_recycle_images(
		data_structures_groups::vector_ge &generators,
		int **old_images,
		int idx_generator_to_delete, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images" << endl;
	}
	if (generators.len) {
		init_generators_recycle_images(
				generators.len,
				generators.ith(0),
				old_images,
				idx_generator_to_delete);
	}
	else {
		init_generators_recycle_images(generators.len,
				NULL, old_images, idx_generator_to_delete);
	}
	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images done" << endl;
	}
}

void generators_and_images::init_generators_recycle_images(
		data_structures_groups::vector_ge &generators,
		int **old_images, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images" << endl;
	}
	if (generators.len) {
		init_generators_recycle_images(
				generators.len,
				generators.ith(0),
				old_images,
				verbose_level);
	}
	else {
		init_generators_recycle_images(
				generators.len,
				NULL,
				old_images,
				verbose_level);
	}
	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images done" << endl;
	}
}
#endif

#if 0
void generators_and_images::init_generators_recycle_images(
		int nb, int *elt,
		int **old_images, int idx_generator_to_delete,
		int verbose_level)
// elt must point to nb * A->elt_size_in_int
// int's that are
// group elements in int format
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images" << endl;
	}

	gens.allocate(nb, verbose_level - 2);
	gens_inv.allocate(nb, verbose_level - 2);
	for (i = 0; i < nb; i++) {
		//cout << "schreier::init_generators i = " << i << endl;
		gens.copy_in(i, elt + i * A->elt_size_in_int);
		A->Group_element->element_invert(
				elt + i * A->elt_size_in_int,
				gens_inv.ith(i), 0);
	}
	init_images_recycle(nb, old_images,
			idx_generator_to_delete,
			0 /* verbose_level */);
	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images done" << endl;
	}
}



void generators_and_images::init_generators_recycle_images(
		int nb,
		int *elt, int **old_images, int verbose_level)
// elt must point to nb * A->elt_size_in_int
// int's that are
// group elements in int format
{
	int i;

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images" << endl;
	}
	gens.allocate(nb, verbose_level - 2);
	gens_inv.allocate(nb, verbose_level - 2);
	for (i = 0; i < nb; i++) {
		//cout << "schreier::init_generators i = " << i << endl;
		gens.copy_in(i, elt + i * A->elt_size_in_int);
		A->Group_element->element_invert(
				elt + i * A->elt_size_in_int,
				gens_inv.ith(i), 0);
	}
	init_images_recycle(nb, old_images, verbose_level - 2);
	if (f_v) {
		cout << "generators_and_images::init_generators_recycle_images done" << endl;
	}
}
#endif



void generators_and_images::init_generators_by_hdl(
		int nb_gen,
	int *gen_hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;

	if (f_v) {
		cout << "generators_and_images::init_generators_by_hdl" << endl;
		cout << "nb_gen = " << nb_gen << endl;
		cout << "degree = " << degree << endl;
	}

	gens.allocate(nb_gen, verbose_level - 2);
	gens_inv.allocate(nb_gen, verbose_level - 2);
	for (i = 0; i < nb_gen; i++) {
		//cout << "schreier::init_generators_by_hdl "
		// "i = " << i << endl;
		A->Group_element->element_retrieve(
				gen_hdl[i], gens.ith(i), 0);

		//cout << "schreier::init_generators_by_hdl "
		// "generator i = " << i << ":" << endl;
		//A->element_print_quick(gens.ith(i), cout);

		A->Group_element->element_invert(
				gens.ith(i), gens_inv.ith(i), 0);
	}
	if (f_vv) {
		cout << "generators_and_images::init_generators_by_hdl "
				"generators:" << endl;
		gens.print(cout);
	}
	if (f_v) {
		cout << "generators_and_images::init_generators_by_hdl "
				"before init_images()" << endl;
	}
	init_images(nb_gen, verbose_level);
	if (f_v) {
		cout << "generators_and_images::init_generators_by_hdl "
				"done" << endl;
	}
}

void generators_and_images::init_generators_by_handle(
		std::vector<int> &gen_hdl,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i;
	int nb_gen;

	if (f_v) {
		cout << "generators_and_images::init_generators_by_handle" << endl;
		cout << "degree = " << degree << endl;
	}

	nb_gen = gen_hdl.size();

	gens.allocate(nb_gen, verbose_level - 2);
	gens_inv.allocate(nb_gen, verbose_level - 2);

	for (i = 0; i < nb_gen; i++) {

		//cout << "schreier::init_generators_by_hdl "
		// "i = " << i << endl;
		A->Group_element->element_retrieve(
				gen_hdl[i], gens.ith(i), 0);

		//cout << "schreier::init_generators_by_hdl "
		// "generator i = " << i << ":" << endl;
		//A->element_print_quick(gens.ith(i), cout);

		A->Group_element->element_invert(
				gens.ith(i), gens_inv.ith(i), 0);
	}
	if (f_vv) {
		cout << "generators_and_images::init_generators_by_handle "
				"generators:" << endl;
		gens.print(cout);
	}
	if (f_v) {
		cout << "generators_and_images::init_generators_by_handle "
				"before init_images()" << endl;
	}
	init_images(nb_gen, verbose_level);
	if (f_v) {
		cout << "generators_and_images::init_generators_by_handle "
				"done" << endl;
	}
}

void generators_and_images::append_one(
		int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::append_one" << endl;
	}

	gens.append(Elt, verbose_level - 2);
	A->Group_element->element_invert(Elt, A->Group_element->Elt1, false);
	gens_inv.append(A->Group_element->Elt1, verbose_level - 2);
	images_append(verbose_level - 2);

	if (f_v) {
		cout << "generators_and_images::append_one done" << endl;
	}
}

long int generators_and_images::get_image(
		long int i, int gen_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	long int a;

	if (f_v) {
		cout << "generators_and_images::get_image "
				"computing image of point "
				<< i << " under generator " << gen_idx
				<< " verbose_level = " << verbose_level << endl;
	}
	if (images == NULL) {
		if (f_v) {
			cout << "generators_and_images::get_image "
					"not using image table" << endl;
		}
		if (f_images_only) {
			cout << "generators_and_images::get_image images == NULL "
					"and f_images_only" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "generators_and_images::get_image "
					"before A->element_image_of" << endl;
		}
		a = A->Group_element->element_image_of(
				i,
				gens.ith(gen_idx),
				verbose_level - 2);
		if (f_v) {
			cout << "generators_and_images::get_image "
					"after A->element_image_of" << endl;
		}
		//cout << "generators_and_images::get_image"
		// "images == NULL" << endl;
		//exit(1);
	}
	else {
		if (f_v) {
			cout << "generators_and_images::get_image using image table" << endl;
		}
		a = images[gen_idx][i];
		if (a == -1) {
			if (f_images_only) {
				cout << "generators_and_images::get_image a == -1 "
						"is not allowed if f_images_only is true" << endl;
				exit(1);
			}
			if (f_v) {
				cout << "generators_and_images::get_image "
						"before A->element_image_of" << endl;
			}
			a = A->Group_element->element_image_of(
					i, gens.ith(gen_idx),
					verbose_level - 2);
			if (f_v) {
				cout << "generators_and_images::get_image image of "
						"i=" << i << " is " << a << endl;
			}
			images[gen_idx][i] = a;
			images[gen_idx][A->degree + a] = i;
		}
	}
	if (f_v) {
		cout << "generators_and_images::get_image image of point "
				<< i << " under generator " << gen_idx
				<< " is " << a << endl;
	}
	return a;
}


void generators_and_images::transporter_from_orbit_rep_to_point(
		int pt,
		int &orbit_idx, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pos;

	if (f_v) {
		cout << "generators_and_images::transporter_from_orbit_rep_to_point" << endl;
	}
	if (f_images_only) {
		cout << "generators_and_images::transporter_from_orbit_rep_to_point "
				"is not allowed if f_images_only is true" << endl;
		exit(1);
	}
	pos = Schreier->Forest->orbit_inv[pt];
	orbit_idx = Schreier->Forest->orbit_number(pt); //orbit_no[pos];
	//cout << "lies in orbit " << orbit_idx << endl;
	coset_rep(pos, verbose_level - 1);
	A->Group_element->element_move(cosetrep, Elt, 0);
	if (f_v) {
		cout << "generators_and_images::transporter_from_orbit_rep_to_point "
				"done" << endl;
	}
}

void generators_and_images::transporter_from_point_to_orbit_rep(
		int pt,
	int &orbit_idx, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int pos;

	if (f_v) {
		cout << "generators_and_images::transporter_from_point_to_orbit_rep" << endl;
	}
	if (f_images_only) {
		cout << "generators_and_images::transporter_from_point_to_orbit_rep "
				"is not allowed if f_images_only is true" << endl;
		exit(1);
	}
	pos = Schreier->Forest->orbit_inv[pt];
	orbit_idx = Schreier->Forest->orbit_number(pt); //orbit_no[pos];
	//cout << "lies in orbit " << orbit_idx << endl;

	coset_rep(pos, verbose_level - 1);

	A->Group_element->element_invert(cosetrep, Elt, 0);
	//A->element_move(cosetrep, Elt, 0);
	if (f_v) {
		cout << "generators_and_images::transporter_from_point_to_orbit_rep "
				"done" << endl;
	}
}


void generators_and_images::coset_rep(
		int j, int verbose_level)
// j is a coset, not a point
// result is in cosetrep
// determines an element in the group
// that moves the orbit representative
// to the j-th point in the orbit.
{
	int f_v = (verbose_level >= 1);
	int *gen;

	if (f_v) {
		cout << "generators_and_images::coset_rep coset "
				"j=" << j << " pt=" << Schreier->Forest->orbit[j] << endl;
	}
	if (f_images_only) {
		cout << "generators_and_images::coset_rep is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	if (Schreier->Forest->prev[j] != -1) {

		if (f_v) {
			cout << "generators_and_images::coset_rep "
					"j=" << j << " pt=" << Schreier->Forest->orbit[j];
			cout << " prev[j]=" << Schreier->Forest->prev[j];
			cout << " orbit_inv[prev[j]]="
					<< Schreier->Forest->orbit_inv[Schreier->Forest->prev[j]];
			cout << " label[j]=" << Schreier->Forest->label[j] << endl;
		}
		coset_rep(
				Schreier->Forest->orbit_inv[Schreier->Forest->prev[j]],
				verbose_level);

		gen = gens.ith(Schreier->Forest->label[j]);

		A->Group_element->element_mult(
				cosetrep, gen, cosetrep_tmp, 0);

		A->Group_element->element_move(
				cosetrep_tmp, cosetrep, 0);
	}
	else {
		A->Group_element->element_one(cosetrep, 0);
	}
	if (f_v) {
		cout << "generators_and_images::coset_rep "
				"j=" << j << " pt=" << Schreier->Forest->orbit[j]<< " done" << endl;
	}
}


void generators_and_images::coset_rep_inv(
		int j, int verbose_level)
// j is a coset, not a point
// result is in cosetrep
{
	int f_v = (verbose_level >= 1);
	int *gen;

	if (f_v) {
		cout << "generators_and_images::coset_rep_inv j=" << j << endl;
	}
	if (f_images_only) {
		cout << "generators_and_images::coset_rep_inv is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	if (Schreier->Forest->prev[j] != -1) {
		if (f_v) {
			cout << "generators_and_images::coset_rep_inv j=" << j
					<< " orbit_inv[prev[j]]="
					<< Schreier->Forest->orbit_inv[Schreier->Forest->prev[j]]
					<< " label[j]=" << Schreier->Forest->label[j] << endl;
		}

		coset_rep_inv(
				Schreier->Forest->orbit_inv[Schreier->Forest->prev[j]],
				verbose_level);

		gen = gens_inv.ith(Schreier->Forest->label[j]);
		A->Group_element->element_mult(gen, cosetrep, cosetrep_tmp, 0);
		A->Group_element->element_move(cosetrep_tmp, cosetrep, 0);
	}
	else {
		A->Group_element->element_one(cosetrep, 0);
	}
	if (f_v) {
		cout << "generators_and_images::coset_rep_inv j=" << j << " done" << endl;
	}
}

void generators_and_images::random_schreier_generator(
		int *Elt, int verbose_level)
// computes random Schreier generator
// for the first orbit into Elt
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "generators_and_images::random_schreier_generator "
				"orbit_len = "
			<< Schreier->Forest->orbit_len[0] << " nb generators = "
			<< gens.len << " in action " << A->label << endl;
	}

	if (f_v) {
		cout << "generators_and_images::random_schreier_generator "
				"before random_schreier_generator_ith_orbit" << endl;
	}
	random_schreier_generator_ith_orbit(
			Elt,
			0 /* orbit_no */, verbose_level - 1);
	if (f_v) {
		cout << "generators_and_images::random_schreier_generator "
				"after random_schreier_generator_ith_orbit" << endl;
	}

#if 0

	int f_vv = (verbose_level >= 2);
	int r1, r2, pt, pt2, pt2b, pt2_coset;
	int *gen;
	int pt1, pt1b;
	other::orbiter_kernel_system::os_interface Os;

	if (f_images_only) {
		cout << "generators_and_images::random_schreier_generator is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	pt = Schreier->Forest->orbit[0];
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator pt=" << pt << endl;
	}

	// get a random coset:
	r1 = Os.random_integer(Schreier->Forest->orbit_len[0]);
	pt1 = Schreier->Forest->orbit[r1];

	coset_rep(r1, verbose_level - 1);

	// coset rep now in cosetrep

	pt1b = A->Group_element->element_image_of(pt, cosetrep, 0);

	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator "
				"random coset " << r1 << endl;
		cout << "generators_and_images::random_schreier_generator "
				"pt1=" << pt1 << endl;
		cout << "generators_and_images::random_schreier_generator "
				"cosetrep:" << endl;
		A->Group_element->element_print_quick(
				cosetrep, cout);
		cout << "generators_and_images::random_schreier_generator "
				"image of pt under cosetrep = " << pt1b << endl;
	}
	if (pt1b != pt1) {
		cout << "generators_and_images::random_schreier_generator fatal: "
				"cosetrep does not work" << endl;
		cout << "pt=" << pt << endl;
		cout << "random coset " << r1 << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "cosetrep:" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
		cout << "image of pt under cosetrep = " << pt1b << endl;
		A->Group_element->element_image_of(pt, cosetrep, 10);
		exit(1);
	}

	// get a random generator:
	r2 = Os.random_integer(gens.len);
	gen = gens.ith(r2);

	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator "
				"random coset " << r1 << ", "
				"generators_and_images::random_schreier_generator "
				"random generator " << r2 << endl;
		cout << "generators_and_images::random_schreier_generator "
				"generator:" << endl;
		A->Group_element->element_print_quick(gen, cout);
		cout << "generators_and_images::random_schreier_generator "
				"image of pt1 under generator = pt2 = "
				<< A->Group_element->element_image_of(pt1, gen, 0) << endl;
	}
	pt2b = A->Group_element->element_image_of(pt1, gen, 0);

	A->Group_element->element_mult(
			cosetrep, gen, schreier_gen1, 0);

	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator "
				"cosetrep * gen=" << endl;
		A->Group_element->element_print_quick(schreier_gen1, cout);
	}
	pt2 = A->Group_element->element_image_of(
			pt, schreier_gen1, 0);
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator "
				"image of pt under cosetrep*gen = " << pt2 << endl;
	}
	if (pt2 != pt2b) {
		cout << "generators_and_images::random_schreier_generator "
				"something is wrong! " << endl;
		cout << "pt2=" << pt2 << " = image of pt "
				"under cosetrep * gen" << endl;
		cout << "pt2b=" << pt2b << " = image of pt1 "
				"under gen" << endl;
		cout << "cosetrep:" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
		cout << "generator:" << endl;
		A->Group_element->element_print_quick(gen, cout);
		cout << "cosetrep * gen=" << endl;
		A->Group_element->element_print_quick(schreier_gen1, cout);
		cout << "pt=" << pt << endl;
		cout << "pt1=" << pt1 << endl;
		cout << "pt1b=" << pt1b << endl;
		cout << "pt2=" << pt2 << endl;
		cout << "pt2b=" << pt2b << endl;

		cout << "repeat 1" << endl;
		cout << "repeating pt1b = A->element_image_of(pt, "
				"cosetrep, 0):" << endl;
		pt1b = A->Group_element->element_image_of(
				pt, cosetrep, verbose_level + 3);
		cout << "pt1b = " << pt1b << endl;

		cout << "repeat 2" << endl;
		cout << "repeating pt2b = A->element_image_of(pt1, "
				"gen, 0):" << endl;
		pt2b = A->Group_element->element_image_of(
				pt1, gen, verbose_level + 3);

		cout << "repeat 3" << endl;
		cout << "repeating pt2 = A->element_image_of(pt, "
				"schreier_gen1, 0):" << endl;
		pt2 = A->Group_element->element_image_of(
				pt, schreier_gen1, verbose_level + 3);


		exit(1);
	}
	//cout << "maps " << pt << " to " << pt2 << endl;

	pt2_coset = Schreier->Forest->orbit_inv[pt2];

	coset_rep_inv(pt2_coset, verbose_level - 1);

	// coset rep now in cosetrep
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator cosetrep:" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
		cout << "generators_and_images::random_schreier_generator "
				"image of pt2 under cosetrep = "
				<< A->Group_element->element_image_of(pt2, cosetrep, 0) << endl;
	}

	A->Group_element->element_mult(schreier_gen1, cosetrep, Elt, 0);
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator "
				"Elt=cosetrep*gen*cosetrep:" << endl;
		A->Group_element->element_print_quick(Elt, cout);
		cout << "generators_and_images::random_schreier_generator image of pt under Elt = "
				<< A->Group_element->element_image_of(pt, Elt, 0) << endl;
	}

	int pt3;

	pt3 = A->Group_element->element_image_of(pt, Elt, 0);

	if (pt3 != pt) {
		cout << "generators_and_images::random_schreier_generator "
				"fatal: schreier generator does not stabilize pt" << endl;
		cout << "pt=" << pt << endl;
		cout << "pt image=" << pt3 << endl;
		cout << "r1=" << r1 << endl;
		cout << "pt1=" << pt1 << endl;

		cout << "r2=" << r2 << endl;
		cout << "generators_and_images::random_schreier_generator "
				"generator r2:" << endl;
		A->Group_element->element_print_quick(gen, cout);

		cout << "generators_and_images::random_schreier_generator "
				"cosetrep * gen=" << endl;
		A->Group_element->element_print_quick(schreier_gen1, cout);

		cout << "pt2=" << pt2 << endl;
		cout << "pt2_coset=" << pt2_coset << endl;

		cout << "generators_and_images::random_schreier_generator "
				"coset_rep_inv=" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);

		cout << "generators_and_images::random_schreier_generator "
				"cosetrep * gen * coset_rep_inv=" << endl;
		A->Group_element->element_print_quick(Elt, cout);


		cout << "generators_and_images::random_schreier_generator "
				"recomputing original cosetrep" << endl;
		coset_rep(pt2_coset, verbose_level + 5);
		cout << "generators_and_images::random_schreier_generator "
				"original cosetrep=" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);


		cout << "generators_and_images::random_schreier_generator "
				"recomputing original cosetrep inverse" << endl;
		coset_rep_inv(pt2_coset, verbose_level + 5);
		cout << "generators_and_images::random_schreier_generator "
				"original cosetrep_inv=" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);

		cout << "redoing the multiplication, "
				"schreier_gen1 * cosetrep=" << endl;
		cout << "in action " << A->label << endl;
		A->Group_element->element_mult(
				schreier_gen1, cosetrep, Elt, 10);
		A->Group_element->element_print_quick(Elt, cout);


		exit(1);
	}
	if (false) {
		cout << "generators_and_images::random_schreier_generator "
				"random Schreier generator:" << endl;
		A->Group_element->element_print(Elt, cout);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "generators_and_images::random_schreier_generator done" << endl;
	}
}


void generators_and_images::random_schreier_generator_ith_orbit(
		int *Elt,
		int orbit_no, int verbose_level)
// computes random Schreier generator
// for the orbit orbit_no into Elt
{
	int first, len, r1, r2, pt, pt2, pt2_coset;
	int *gen;
	int f_v = (verbose_level >= 1);
	int f_vv = false; //(verbose_level >= 2);
	int f_vvv = false; //(verbose_level >= 3);
	other::orbiter_kernel_system::os_interface Os;

	if (f_v) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit, "
				"orbit " << orbit_no << " action=" << A->label << endl;
	}
	if (f_images_only) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	if (f_vvv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"generators are:" << endl;
		gens.print(cout);
	}
	first = Schreier->Forest->orbit_first[orbit_no];
	len = Schreier->Forest->orbit_len[orbit_no];
	pt = Schreier->Forest->orbit[first];
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"pt=" << pt << endl;
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"orbit_first[orbit_no]=" << Schreier->Forest->orbit_first[orbit_no] << endl;
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"orbit_len[orbit_no]=" << Schreier->Forest->orbit_len[orbit_no] << endl;
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"gens.len=" << gens.len << endl;
	}

	// get a random coset:
	r1 = Os.random_integer(Schreier->Forest->orbit_len[orbit_no]);
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"r1=" << r1 << endl;
	}
	//pt1 = orbit[r1];
	coset_rep(
			Schreier->Forest->orbit_first[orbit_no] + r1,
			verbose_level - 1);
	// coset rep now in cosetrep
	if (f_vvv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"cosetrep " << Schreier->Forest->orbit_first[orbit_no] + r1 << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
		if (A->degree < 100) {
			A->Group_element->element_print_as_permutation(cosetrep, cout);
			cout << endl;
		}
	}

	// get a random generator:
	r2 = Os.random_integer(gens.len);
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"r2=" << r2 << endl;
	}
	gen = gens.ith(r2);
	if (f_vvv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"generator " << r2 << endl;
		A->Group_element->element_print(gen, cout);
		if (A->degree < 100) {
			A->Group_element->element_print_as_permutation(gen, cout);
			cout << endl;
		}
	}
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"random coset " << r1
				<< ", random generator " << r2 << endl;
	}

	A->Group_element->element_mult(cosetrep, gen, schreier_gen1, 0);
	if (f_vvv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"cosetrep * generator " << endl;
		A->Group_element->element_print_quick(schreier_gen1, cout);
		if (A->degree < 100) {
			A->Group_element->element_print_as_permutation(
					schreier_gen1, cout);
			cout << endl;
		}
	}
	pt2 = A->Group_element->element_image_of(pt, schreier_gen1, 0);
	if (f_vv) {
		//cout << "pt2=" << pt2 << endl;
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"maps " << pt << " to " << pt2 << endl;
	}
	pt2_coset = Schreier->Forest->orbit_inv[pt2];
	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"pt2_coset=" << pt2_coset << endl;
	}
	if (pt2_coset < first) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"pt2_coset < first" << endl;
		exit(1);
	}
	if (pt2_coset >= first + len) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"pt2_coset >= first + len" << endl;
		exit(1);
	}

	coset_rep_inv(pt2_coset, verbose_level - 1);
	// coset rep now in cosetrep
	if (f_vvv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"cosetrep (inverse) " << pt2_coset << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
		if (A->degree < 100) {
			A->Group_element->element_print_as_permutation(cosetrep, cout);
			cout << endl;
		}
	}

	A->Group_element->element_mult(
			schreier_gen1, cosetrep, Elt, 0);

	if (A->Group_element->element_image_of(pt, Elt, 0) != pt) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"fatal: schreier generator does not stabilize pt" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit "
				"done" << endl;
	}

	if (f_vvv) {
		A->Group_element->element_print_quick(Elt, cout);
		cout << endl;
		if (A->degree < 100) {
			A->Group_element->element_print_as_permutation(Elt, cout);
			cout << endl;
		}
	}
	if (f_v) {
		cout << "generators_and_images::random_schreier_generator_ith_orbit, "
				"orbit " << orbit_no << " done" << endl;
	}
}


void generators_and_images::print_generators()
{
	int j;

	cout << gens.len << " generators in action "
			<< A->label << " of degree " << A->degree << ":" << endl;
	for (j = 0; j < gens.len; j++) {
		cout << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		A->Group_element->element_print_quick(
				gens.ith(j), cout);
		//A->element_print_as_permutation(gens.ith(j), cout);
		if (j < gens.len - 1) {
			cout << ", " << endl;
		}
	}
}

void generators_and_images::print_generators_latex(
		std::ostream &ost)
{
	int j;

	ost << gens.len << " generators in action $"
			<< A->label_tex << "$ of degree "
			<< A->degree << ":\\\\" << endl;
	for (j = 0; j < gens.len; j++) {
		ost << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		ost << "$$" << endl;
		A->Group_element->element_print_latex(gens.ith(j), ost);
		//A->element_print_as_permutation(gens.ith(j), cout);
		if (j < gens.len - 1) {
			ost << ", " << endl;
		}
		ost << "$$" << endl;
	}
}

void generators_and_images::print_generators_with_permutations()
{
	int j;

	cout << gens.len << " generators in action "
			<< A->label << " of degree "
			<< A->degree << ":" << endl;
	for (j = 0; j < gens.len; j++) {
		cout << "generator " << j << ":" << endl;
		//A->element_print(gens.ith(j), cout);
		A->Group_element->element_print_quick(gens.ith(j), cout);
		A->Group_element->element_print_as_permutation(gens.ith(j), cout);
		cout << endl;
		if (j < gens.len - 1) {
			cout << ", " << endl;
		}
	}
}


void generators_and_images::list_elements_as_permutations_vertically(
		std::ostream &ost)
{
	if (f_images_only) {
		cout << "generators_and_images::list_elements_as_permutations_vertically is not "
				"allowed if f_images_only is true" << endl;
		exit(1);
	}
	A->list_elements_as_permutations_vertically(&gens, ost);
}



}}}


