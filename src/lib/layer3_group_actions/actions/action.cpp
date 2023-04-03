// action.cpp
//
// Anton Betten
// July 8, 2003

#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace actions {



action::action()
{
	int verbose_level = 0;

	orbiter_kernel_system::Orbiter->nb_times_action_created++;

	null();

	Known_groups = NEW_OBJECT(known_groups);
	Known_groups->init(this, verbose_level);

	Induced_action = NEW_OBJECT(induced_action);
	Induced_action->init(this, verbose_level);

	Group_element = NEW_OBJECT(group_element);
	Group_element->init(this, verbose_level);

}

action::~action()
{
	freeself();
}

void action::null()
{
	//label[0] = 0;
	//label_tex[0] = 0;
	
	//user_data_type = 0;
	type_G = unknown_symmetry_group_t;
	
	subaction = NULL;
	f_has_strong_generators = false;
	Strong_gens = NULL;
	//strong_generators = NULL;


	//transversal_reps = NULL;

	null_element_data();

	degree = 0;
	f_is_linear = false;
	dimension = 0;

	f_has_stabilizer_chain = false;

	Stabilizer_chain = NULL;

	Known_groups = NULL;

	Induced_action = NULL;

	elt_size_in_int = 0;
	coded_elt_size_in_char = 0;
	//group_prefix[0] = 0;
	//f_has_transversal_reps = false;
	f_group_order_is_small = false;
	make_element_size = 0;
	low_level_point_size = 0;

	ptr = NULL;

	f_allocated = false;
	f_has_subaction = false;
	f_subaction_is_allocated = false;
	f_has_sims = false;
	f_has_kernel = false;
}

void action::freeself()
{
	//int i;
	int f_v = false;
	int f_vv = false;

	if (f_v) {
		cout << "action::freeself deleting action " << label << endl;
		print_info();
	}
	if (f_allocated) {
		if (f_vv) {
			cout << "action::freeself freeing G of type ";

			action_global AG;
			AG.action_print_symmetry_group_type(cout, type_G);
			cout << endl;
		}
		if (type_G == matrix_group_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.matrix_grp" << endl;
				cout << "G.matrix_grp=" << G.matrix_grp << endl;
			}
			FREE_OBJECT(G.matrix_grp);
			if (f_vv) {
				cout << "action::~action freeing G.matrix_grp finished"
						<< endl;
			}
			G.matrix_grp = NULL;
		}
		else if (type_G == wreath_product_t) {
			if (f_vv) {
				cout << "action::freeself freeing "
						"G.wreath_product_group" << endl;
				cout << "G.wreath_product_group="
						<< G.wreath_product_group << endl;
			}
			FREE_OBJECT(G.wreath_product_group);
			if (f_vv) {
				cout << "action::freeself freeing "
						"G.wreath_product_group finished" << endl;
			}
			G.wreath_product_group = NULL;
		}
		else if (type_G == perm_group_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.perm_group_t" << endl;
			}
			FREE_OBJECT(G.perm_grp);
			if (f_vv) {
				cout << "action::freeself freeing G.perm_group_t finished"
						<< endl;
			}
			G.perm_grp = NULL;
		}
		else if (type_G == action_on_sets_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.on_sets" << endl;
				cout << "G.on_sets=" << G.on_sets << endl;
			}
			FREE_OBJECT(G.on_sets);
			if (f_vv) {
				cout << "action::freeself freeing G.on_sets finished" << endl;
			}
			G.on_sets = NULL;
		}
		else if (type_G == action_on_set_partitions_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnSetPartitions" << endl;
				cout << "G.OnSetPartitions=" << G.OnSetPartitions << endl;
			}
			FREE_OBJECT(G.OnSetPartitions);
			if (f_vv) {
				cout << "action::freeself freeing G.OnSetPartitions finished" << endl;
			}
			G.OnSetPartitions = NULL;
		}
		else if (type_G == action_on_k_subsets_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.on_sets" << endl;
				cout << "G.on_k_subsets=" << G.on_k_subsets << endl;
			}
			FREE_OBJECT(G.on_k_subsets);
			if (f_vv) {
				cout << "action::freeself freeing G.on_k_subsets finished"
						<< endl;
			}
			G.on_k_subsets = NULL;
		}
		else if (type_G == action_on_orbits_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnOrbits" << endl;
				cout << "G.OnOrbits=" << G.OnOrbits << endl;
			}
			FREE_OBJECT(G.OnOrbits);
			if (f_vv) {
				cout << "action::freeself freeing G.OnOrbits finished" << endl;
			}
			G.OnOrbits = NULL;
		}
		else if (type_G == action_on_bricks_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnBricks" << endl;
				cout << "G.OnBricks=" << G.OnBricks << endl;
			}
			FREE_OBJECT(G.OnBricks);
			if (f_vv) {
				cout << "action::freeself freeing G.OnBricks finished" << endl;
			}
			G.OnBricks = NULL;
		}
		else if (type_G == action_on_andre_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnAndre" << endl;
				cout << "G.OnAndre=" << G.OnAndre << endl;
			}
			FREE_OBJECT(G.OnAndre);
			if (f_vv) {
				cout << "action::freeself freeing G.OnAndre finished" << endl;
			}
			G.OnAndre = NULL;
		}
		else if (type_G == action_by_right_multiplication_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.ABRM" << endl;
			}
			FREE_OBJECT(G.ABRM);
			G.ABRM = NULL;
		}
		else if (type_G == action_by_restriction_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.ABR" << endl;
			}
			FREE_OBJECT(G.ABR);
			G.ABR = NULL;
		}
		else if (type_G == action_by_conjugation_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.ABC" << endl;
			}
			FREE_OBJECT(G.ABC);
			G.ABC = NULL;
		}
		else if (type_G == action_by_representation_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.Rep" << endl;
			}
			FREE_OBJECT(G.Rep);
			G.Rep = NULL;
		}
		else if (type_G == action_on_determinant_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.AD" << endl;
			}
			FREE_OBJECT(G.AD);
			G.AD = NULL;
		}
		else if (type_G == action_on_galois_group_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.on_Galois_group" << endl;
			}
			FREE_OBJECT(G.on_Galois_group);
			G.on_Galois_group = NULL;
		}
		else if (type_G == action_on_sign_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnSign" << endl;
			}
			FREE_OBJECT(G.OnSign);
			G.OnSign = NULL;
		}
		else if (type_G == action_on_grassmannian_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.AG" << endl;
			}
			FREE_OBJECT(G.AG);
			G.AG = NULL;
		}
		else if (type_G == action_on_factor_space_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.AF" << endl;
			}
			FREE_OBJECT(G.AF);
			G.AF = NULL;
		}
		else if (type_G == action_on_wedge_product_t) {
			//FREE_OBJECT(G.AW);
			G.AW = NULL;
		}
		else if (type_G == action_on_homogeneous_polynomials_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.OnHP" << endl;
			}
			FREE_OBJECT(G.OnHP);
			G.OnHP = NULL;
		}
		else if (type_G == action_by_subfield_structure_t) {
			if (f_vv) {
				cout << "action::freeself freeing G.SubfieldStructure" << endl;
			}
			FREE_OBJECT(G.SubfieldStructure);
			G.SubfieldStructure = NULL;
		}
		else {
			cout << "action::freeself don't know "
					"how to free the object; action type is ";
			print_symmetry_group_type(cout);
			cout << endl;
			exit(1);
		}
		f_allocated = false;
		type_G = unknown_symmetry_group_t;
	}
	if (f_v) {
		cout << "action::freeself after freeing G " << endl;
	}

	if (Stabilizer_chain) {
		FREE_OBJECT(Stabilizer_chain);
	}

	if (f_v) {
		cout << "action::freeself after free_base_data" << endl;
	}
	

	if (Known_groups) {
		FREE_OBJECT(Known_groups);
	}


	if (f_v) {
		cout << "action::freeself after freeing transversal reps" << endl;
	}
		
	free_element_data();

	if (f_v) {
		cout << "action::freeself after free_element_data" << endl;
	}
	
	if (f_has_strong_generators) {
		if (f_v) {
			cout << "we are freeing strong generators" << endl;
		}
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
		//FREE_OBJECT(strong_generators); //delete strong_generators;
		//FREE_int(tl);
		//strong_generators = NULL;
		//tl = NULL;
		f_has_strong_generators = false;
	}

	if (f_v) {
		cout << "action::freeself after freeing strong generators" << endl;
	}

	if (f_has_subaction && f_subaction_is_allocated) {
		if (f_v) {
			cout << "subaction is allocated, so we free it" << endl;
			subaction->print_info();
		}
		FREE_OBJECT(subaction);
		subaction = NULL;
		f_subaction_is_allocated = false;
		f_has_subaction = false;
	}

	if (f_v) {
		cout << "action::freeself after freeing subaction" << endl;
	}
	
	if (f_has_sims) {
		if (f_v) {
			cout << "action::freeself freeing Sims" << endl;
		}
		FREE_OBJECT(Sims);
		Sims = NULL;
		f_has_sims = false;
		if (f_v) {
			cout << "action::freeself freeing Sims finished" << endl;
		}
	}

	if (f_v) {
		cout << "action::freeself after freeing sims" << endl;
	}

	if (f_has_kernel) {
		if (f_v) {
			cout << "action::freeself freeing Kernel" << endl;
		}
		FREE_OBJECT(Kernel);
		Kernel = NULL;
		f_has_kernel = false;
		if (f_v) {
			cout << "action::freeself freeing Kernel finished" << endl;
		}
	}

	if (ptr) {
		FREE_OBJECT(ptr);
	}

	if (f_v) {
		cout << "action::freeself "
				"deleting action " << label << " done" << endl;
	}
}

int action::f_has_base()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_f_has_base();
	}
	else {
		return false;
	}
}


int action::base_len()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_base_len();
	}
	else {
		cout << "action::base_len Stabilizer_chain == NULL" << endl;
		exit(1);
		//return 0;
	}
}

void action::set_base_len(int base_len)
{
	if (Stabilizer_chain) {
		Stabilizer_chain->get_base_len() = base_len;
	}
	else {
		cout << "action::set_base_len no stabilizer chain" << endl;
	}

}

long int &action::base_i(int i)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->base_i(i);
	}
	else {
		cout << "action::base_i no Stabilizer_chain" << endl;
		exit(1);
	}
}

long int *&action::get_base()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_base();
	}
	else {
		cout << "action::get_base no Stabilizer_chain" << endl;
		exit(1);
	}
}

int &action::transversal_length_i(int i)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->transversal_length_i(i);
	}
	else {
		cout << "action::transversal_length_i no Stabilizer_chain" << endl;
		exit(1);
	}
}

int *&action::get_transversal_length()
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->get_transversal_length();
	}
	else {
		cout << "action::transversal_length no Stabilizer_chain" << endl;
		exit(1);
	}
}

long int &action::orbit_ij(int i, int j)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->orbit_ij(i, j);
	}
	else {
		cout << "action::orbit_ij no Stabilizer_chain" << endl;
		exit(1);
	}
}

long int &action::orbit_inv_ij(int i, int j)
{
	if (Stabilizer_chain) {
		return Stabilizer_chain->orbit_inv_ij(i, j);
	}
	else {
		cout << "action::orbit_inv_ij no Stabilizer_chain" << endl;
		exit(1);
	}
}



void action::null_element_data()
{
	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	element_rw_memory_object = NULL;
}

void action::allocate_element_data()
{
	Elt1 = Elt2 = Elt3 = Elt4 = Elt5 = NULL;
	eltrk1 = eltrk2 = eltrk3 = NULL;
	elt_mult_apply = NULL;
	elt1 = NULL;
	Elt1 = NEW_int(elt_size_in_int);
	Elt2 = NEW_int(elt_size_in_int);
	Elt3 = NEW_int(elt_size_in_int);
	Elt4 = NEW_int(elt_size_in_int);
	Elt5 = NEW_int(elt_size_in_int);
	eltrk1 = NEW_int(elt_size_in_int);
	eltrk2 = NEW_int(elt_size_in_int);
	eltrk3 = NEW_int(elt_size_in_int);
	elt_mult_apply = NEW_int(elt_size_in_int);
	elt1 = NEW_uchar(coded_elt_size_in_char);
	element_rw_memory_object = NEW_char(coded_elt_size_in_char);
}

void action::free_element_data()
{
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
	if (Elt4) {
		FREE_int(Elt4);
	}
	if (Elt5) {
		FREE_int(Elt5);
	}
	if (eltrk1) {
		FREE_int(eltrk1);
	}
	if (eltrk2) {
		FREE_int(eltrk2);
	}
	if (eltrk3) {
		FREE_int(eltrk3);
	}
	if (elt_mult_apply) {
		FREE_int(elt_mult_apply);
	}
	if (elt1) {
		FREE_uchar(elt1);
	}
	if (element_rw_memory_object) {
		FREE_char(element_rw_memory_object);
	}
	null_element_data();
}



// #############################################################################



void action::map_a_set_based_on_hdl(
		long int *set,
		long int *image_set,
		int n, action *A_base, int hdl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *Elt;

	if (f_v) {
		cout << "action::map_a_set_based_on_hdl" << endl;
	}
	Elt = NEW_int(elt_size_in_int);

	A_base->Group_element->element_retrieve(hdl, Elt, false);

	Group_element->map_a_set(set,
		image_set, n, Elt, verbose_level);

	FREE_int(Elt);
	if (f_v) {
		cout << "action::map_a_set_based_on_hdl done" << endl;
	}
}




void action::init_sims_only(groups::sims *G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int i, k;

	if (f_v) {
		cout << "action::init_sims_only action " << label
				<< " base_len = " << base_len() << endl;
		cout << "action::init_sims_only "
				"Stabilizer_chain->A = " << Stabilizer_chain->get_A()->label << endl;
	}
	if (f_has_sims) {
		FREE_OBJECT(Sims);
		Sims = NULL;
		f_has_sims = false;
	}
	if (G->A != this) {
		cout << "action::init_sims_only action " << label
				<< " sims object has different action "
				<< G->A->label << endl;
		exit(1);
	}
	Sims = G;
	f_has_sims = true;
	if (f_v) {
		cout << "action::init_sims_only "
				"before Stabilizer_chain->init_base_from_sims" << endl;
	}
	Stabilizer_chain->init_base_from_sims(G, verbose_level);
	if (f_v) {
		cout << "action::init_sims_only "
				"after Stabilizer_chain->init_base_from_sims" << endl;
	}
#if 0
	if (f_v) {
		cout << "action::init_sims_only action " << label
				<< " before compute_strong_generators_from_sims" << endl;
	}
	compute_strong_generators_from_sims(0/*verbose_level - 2*/);
	if (f_v) {
		cout << "action::init_sims_only action " << label
				<< " after compute_strong_generators_from_sims" << endl;
	}
#endif
	
	if (f_v) {
		cout << "action::init_sims_only done" << endl;
	}
}

void action::compute_strong_generators_from_sims(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_strong_generators_from_sims" << endl;
	}
	if (!f_has_sims) {
		cout << "action::compute_strong_generators_from_sims need sims" << endl;
		exit(1);
	}
	if (f_has_strong_generators) {
		FREE_OBJECT(Strong_gens);
		Strong_gens = NULL;
		f_has_strong_generators = false;
	}
	Strong_gens = NEW_OBJECT(groups::strong_generators);
	Strong_gens->init_from_sims(Sims, verbose_level - 2);
	f_has_strong_generators = true;
	if (f_v) {
		cout << "action::compute_strong_generators_from_sims done" << endl;
	}
}

void action::compute_all_point_orbits(groups::schreier &S,
		data_structures_groups::vector_ge &gens, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_all_point_orbits" << endl;
	}
	S.init(this, verbose_level - 2);
	S.init_generators(gens, verbose_level - 2);
	S.compute_all_point_orbits(verbose_level - 1);
	if (f_v) {
		cout << "action::compute_all_point_orbits done" << endl;
	}
}

int action::depth_in_stab_chain(int *Elt)
// the index of the first moved base point
{
	int i, j, b;
	
	for (i = 0; i < base_len(); i++) {
		b = base_i(i);
		j = Group_element->element_image_of(b, Elt, 0);
		if (j != b) {
			return i;
		}
	}
	return base_len();
}

void action::strong_generators_at_depth(
		int depth,
		data_structures_groups::vector_ge &gen,
		int verbose_level)
// all strong generators that leave base points 0,..., depth - 1 fix
{
	int i, j, l, n;
	
	l = Strong_gens->gens->len;
	gen.init(this, verbose_level - 2);
	gen.allocate(l, verbose_level - 2);
	n = 0;
	for (i = 0; i < l; i++) {
		j = depth_in_stab_chain(Strong_gens->gens->ith(i));
		if (j >= depth) {
			gen.copy_in(n, Strong_gens->gens->ith(i));
			n++;
		}
	}
	gen.len = n;
}

void action::compute_point_stabilizer_chain(
		data_structures_groups::vector_ge &gen,
		groups::sims *S, int *sequence, int len,
		int verbose_level)
// S points to len + 1 many sims objects
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;
	
	if (f_v) {
		cout << "action::compute_point_stabilizer_chain for sequence ";
		Int_vec_print(cout, sequence, len);
		cout << endl;
	}
	for (i = 0; i <= len; i++) {
		S[i].init(this, verbose_level - 2);
	}
	S[0].init_generators(gen, 0);
	S[0].compute_base_orbits(0 /*verbose_level - 1*/);
	if (f_vv) {
		cout << "automorphism group has order ";
		S[0].print_group_order(cout);
		cout << endl;
		if (f_vvv) {
			cout << "generators:" << endl;
			S[0].print_generators();
		}
	}
	
	for (i = 0; i < len; i++) {
		if (f_vv) {
			cout << "computing stabilizer of " << i 
				<< "-th point in the sequence" << endl;
		}
		S[i].point_stabilizer_stabchain_with_action(this, 
			S[i + 1], sequence[i], 0 /*verbose_level - 2*/);
		if (f_vv) {
			cout << "stabilizer of " << i << "-th point "
					<< sequence[i] << " has order ";
			S[i + 1].print_group_order(cout);
			cout << endl;
			if (f_vvv) {
				cout << "generators:" << endl;
				S[i + 1].print_generators();
			}
		}
	}
	if (f_v) {
		cout << "action::compute_point_stabilizer_chain for sequence ";
		Int_vec_print(cout, sequence, len);
		cout << " finished" << endl;
		cout << "i : order of i-th stabilizer" << endl;
		for (i = 0; i <= len; i++) {
			cout << i << " : ";
			S[i].print_group_order(cout);
			cout << endl;
		}
		if (f_vv) {
			for (i = 0; i <= len; i++) {
				cout << i << " : ";
				cout << "generators:" << endl;
				S[i].print_generators();
			}
		}
	}
}

void action::compute_stabilizer_orbits(
		data_structures::partitionstack *&Staborbits,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int i;
	data_structures_groups::vector_ge gen;

	if (f_v) {
		cout << "action::compute_stabilizer_orbits" << endl;
		cout << "base_len = " << base_len() << endl;
		for (i = 0; i < base_len(); i++) {
			cout << i << " : " << base_i(i) << " : " << transversal_length_i(i);
			//int_vec_print(cout, Stabilizer_chain->orbit[i], Stabilizer_chain->transversal_length[i]);
			cout << endl;
		}
		cout << "degree = " << degree << endl;
	}
	Staborbits = NEW_OBJECTS(data_structures::partitionstack, base_len());
		// where is this freed??? in backtrack.cpp

	for (i = 0; i < base_len(); i++) {
		strong_generators_at_depth(i, gen, verbose_level - 2);
		if (false) {
			cout << "level " << i << " found "
					<< gen.len << " strong generators" << endl;
		}
		if (false) {
			gen.print(cout);
		}

		data_structures::partitionstack *S;
		groups::schreier Schreier;


		S = &Staborbits[i];
		S->allocate(degree, false);

		if (false) {
			cout << "computing point orbits" << endl;
		}

		compute_all_point_orbits(Schreier, gen, 0 /*verbose_level - 2*/);

		if (false) {
			Schreier.print(cout);
		}

		Schreier.get_orbit_partition(*S, 0 /*verbose_level - 2*/);
		if (false) {
			cout << "found " << S->ht << " orbits" << endl;
		}
		if (f_vv) {
			cout << "level " << i << " with "
					<< gen.len << " strong generators : ";
			//cout << "orbit partition at level " << i << ":" << endl;
			S->print(cout);
		}

	}
	if (f_v) {
		cout << "action::compute_stabilizer_orbits finished" << endl;
	}
}


void action::find_strong_generators_at_level(
	int base_len, long int *the_base, int level,
	data_structures_groups::vector_ge &gens,
	data_structures_groups::vector_ge &subset_of_gens,
	int verbose_level)
{
	int nb_generators_found;
	int *gen_idx;
	int nb_gens, i, j, bj, bj_image;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "action::find_strong_generators_at_level "
				"level=" << level << " base: ";
		Lint_vec_print(cout, the_base, base_len);
		cout << endl;
	}
	nb_gens = gens.len;
	gen_idx = NEW_int(gens.len);
	
	nb_generators_found = 0;
	for (i = 0; i < nb_gens; i++) {
		for (j = 0; j < level; j++) {
			bj = the_base[j];
			bj_image = Group_element->element_image_of(bj, gens.ith(i), 0);
			if (bj_image != bj) {
				break;
			}
		}
		if (j == level) {
			gen_idx[nb_generators_found++] = i;
		}
	}
	subset_of_gens.init(this, verbose_level - 2);
	subset_of_gens.allocate(nb_generators_found, verbose_level - 2);
	for (i = 0; i < nb_generators_found; i++) {
		j = gen_idx[i];
		Group_element->element_move(gens.ith(j), subset_of_gens.ith(i), 0);
	}
	FREE_int(gen_idx);
	if (f_v) {
		cout << "action::find_strong_generators_at_level found " 
			<< nb_generators_found << " strong generators" << endl;
		if (f_vv) {
			subset_of_gens.print(cout);
			cout << endl;
		}
	}
}



void action::group_order(ring_theory::longinteger_object &go)
{
	//longinteger_domain D;
	
	if (Stabilizer_chain == NULL) {
		cout << "action::group_order Stabilizer_chain == NULL" << endl;
		go.create(0, __FILE__, __LINE__);
	}
	else {
		Stabilizer_chain->group_order(go);
		//D.multiply_up(go, Stabilizer_chain->transversal_length, base_len());
	}
}

long int action::group_order_lint()
{
	ring_theory::longinteger_object go;

	group_order(go);
	return go.as_lint();
}






void action::get_generators_from_ascii_coding(
		std::string &ascii_coding,
		data_structures_groups::vector_ge *&gens, int *&tl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	ring_theory::longinteger_object go;
	data_structures_groups::group_container *G;

	if (f_v) {
		cout << "action::get_generators_from_ascii_coding" << endl;
	}
	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(this, verbose_level - 2);
	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"before G->init_ascii_coding_to_sims" << endl;
	}
	G->init_ascii_coding_to_sims(ascii_coding, verbose_level - 2);
	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"after G->init_ascii_coding_to_sims" << endl;
	}
		

	G->S->group_order(go);

	gens = NEW_OBJECT(data_structures_groups::vector_ge);
	tl = NEW_int(base_len());
	G->S->extract_strong_generators_in_order(*gens, tl,
			0 /* verbose_level */);

	if (f_vv) {
		cout << "action::get_generators_from_ascii_coding "
				"Group order = " << go << endl;
	}

	FREE_OBJECT(G);
	if (f_v) {
		cout << "action::get_generators_from_ascii_coding done" << endl;
	}
}


void action::lexorder_test(
		long int *set, int set_sz,
	int &set_sz_after_test,
	data_structures_groups::vector_ge *gens, int max_starter,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_v5 = false; //(verbose_level  >= 1);
	groups::schreier *Sch;
	int i, orb, first, a, a0;

	if (f_v) {
		cout << "action::lexorder_test" << endl;
	}

	Sch = NEW_OBJECT(groups::schreier);

	if (f_v) {
		cout << "action::lexorder_test computing orbits in action "
				"of degree " << degree << ", max_starter="
				<< max_starter << endl;
	}
	Sch->init(this, verbose_level - 2);
	Sch->init_generators(*gens, verbose_level - 2);

	//Sch->compute_all_point_orbits(0);
	if (f_v) {
		cout << "action::lexorder_test "
				"before compute_all_orbits_on_invariant_subset" << endl;
	}
	Sch->compute_all_orbits_on_invariant_subset(set_sz, 
		set, 0 /* verbose_level */);
	if (f_v) {
		cout << "action::lexorder_test "
				"after compute_all_orbits_on_invariant_subset" << endl;
	}

	if (f_v) {
		cout << "action::lexorder_test: there are "
				<< Sch->nb_orbits << " orbits on set" << endl;
		Sch->print_orbit_length_distribution(cout);
	}
	if (f_v5) {
		Sch->print_and_list_orbits(cout);
	}

	if (f_v) {
		cout << "action::lexorder_test "
				"max_starter=" << max_starter << endl;
	}
	set_sz_after_test = 0;
	for (i = 0; i < set_sz; i++) {
		a = set[i];
		if (false) {
			cout << "action::lexorder_test "
					"Looking at point " << a << endl;
		}
		orb = Sch->orbit_number(a);
		first = Sch->orbit_first[orb];
		a0 = Sch->orbit[first];
		if (a0 < max_starter) {
			if (f_v) {
				cout << "action::lexorder_test  Point " << a
						<< " maps to " << a0 << " which is less than "
						"max_starter = " << max_starter
						<< " so we eliminate" << endl;
			}
		}
		else {
			set[set_sz_after_test++] = a;
		}
	}
	if (f_v) {
		cout << "action::lexorder_test Of the " << set_sz
				<< " points, we accept " << set_sz_after_test
				<< " and we reject " << set_sz - set_sz_after_test << endl;
	}
	FREE_OBJECT(Sch);
	if (f_v) {
		cout << "action::lexorder_test done" << endl;
	}
	
}

void action::compute_orbits_on_points(
		groups::schreier *&Sch,
		data_structures_groups::vector_ge *gens,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::compute_orbits_on_points" << endl;
	}
	Sch = NEW_OBJECT(groups::schreier);
	if (f_v) {
		cout << "action::compute_orbits_on_points in action ";
		print_info();
	}
	if (f_v) {
		cout << "action::compute_orbits_on_points "
				"before Sch->init" << endl;
	}
	Sch->init(this, verbose_level - 2);
	if (f_v) {
		cout << "action::compute_orbits_on_points "
				"before Sch->init_generators" << endl;
	}
	Sch->init_generators(*gens, verbose_level - 2);
	if (f_v) {
		cout << "action::compute_orbits_on_points "
				"before Sch->compute_all_point_orbits, "
				"degree = " << degree << endl;
	}
	Sch->compute_all_point_orbits(verbose_level - 3);
	if (f_v) {
		cout << "action::compute_orbits_on_points "
				"after Sch->compute_all_point_orbits" << endl;
		cout << "action::compute_orbits_on_points "
				"Sch->nb_orbits=" << Sch->nb_orbits << endl;
	}
	//Sch.print_and_list_orbits(cout);
	if (f_v) {
		cout << "action::compute_orbits_on_points done, we found "
				<< Sch->nb_orbits << " orbits" << endl;
	}
}

void action::point_stabilizer_any_point(
		int &pt,
		groups::schreier *&Sch, groups::sims *&Stab,
		groups::strong_generators *&stab_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::point_stabilizer_any_point" << endl;
	}
	
	int f; //, len;
	ring_theory::longinteger_object go;
	
	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"computing all point orbits:" << endl;
	}
	Sch = Strong_gens->orbits_on_points_schreier(
			this, 0 /* verbose_level */);
	//compute_all_point_orbits(Sch,
	//*Strong_gens->gens, 0 /* verbose_level */);
	if (f_v) {
		cout << "computing all point orbits done, found "
				<< Sch->nb_orbits << " orbits" << endl;
	}


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"orbit rep = "
				<< pt << endl;
	}

	group_order(go);
	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"Computing point stabilizer:" << endl;
	}
	Sch->point_stabilizer(this, go, 
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);

	Stab->group_order(go);

	if (f_v) {
		cout << "action::point_stabilizer_any_point "
				"Computing point stabilizer done:" << endl;
		cout << "action::point_stabilizer_any_point "
				"point stabilizer is a group of order " << go << endl;
	}

	if (f_v) {
		cout << "action::point_stabilizer_any_point computing "
				"strong generators for the point stabilizer:" << endl;
	}
	stab_gens = NEW_OBJECT(groups::strong_generators);
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "action::point_stabilizer_any_point strong generators "
				"for the point stabilizer have been computed" << endl;
	}

	if (f_v) {
		cout << "action::point_stabilizer_any_point done" << endl;
	}
}

void action::point_stabilizer_any_point_with_given_group(
		groups::strong_generators *input_gens,
	int &pt, 
	groups::schreier *&Sch, groups::sims *&Stab,
	groups::strong_generators *&stab_gens,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group" << endl;
	}
	
	int f; //, len;
	ring_theory::longinteger_object go;
	
	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"computing all point orbits:" << endl;
	}
	Sch = input_gens->orbits_on_points_schreier(this, 0 /* verbose_level */);
	//compute_all_point_orbits(Sch, *Strong_gens->gens, 0 /* verbose_level */);
	cout << "computing all point orbits done, found "
			<< Sch->nb_orbits << " orbits" << endl;


	f = Sch->orbit_first[0];
	//len = Sch->orbit_len[0];
	pt = Sch->orbit[f];

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"orbit rep = " << pt << endl;
	}

	input_gens->group_order(go);
	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"Computing point stabilizer:" << endl;
	}
	Sch->point_stabilizer(this, go, 
		Stab, 0 /* orbit_no */, 0 /* verbose_level */);

	Stab->group_order(go);

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"Computing point stabilizer done:" << endl;
		cout << "action::point_stabilizer_any_point_with_given_group "
				"point stabilizer is a group of order " << go << endl;
	}

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"computing strong generators for the point stabilizer:"
				<< endl;
	}
	stab_gens = NEW_OBJECT(groups::strong_generators);
	stab_gens->init_from_sims(Stab, 0 /* verbose_level */);
	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group "
				"strong generators for the point stabilizer "
				"have been computed" << endl;
	}

	if (f_v) {
		cout << "action::point_stabilizer_any_point_with_given_group done"
				<< endl;
	}
}




int action::matrix_group_dimension()
{
#if 0
	if (type_G == matrix_group_t) {
		matrix_group *M;

		M = get_matrix_group();
		return M->n;
	}
	else if (type_G == wreath_product_t) {
		wreath_product *W;

		W = G.wreath_product_group;
		int vector_space_dimension;

		vector_space_dimension = W->dimension_of_tensor_action;
		return vector_space_dimension;
	}
	else {
		cout << "action::matrix_group_dimension not a matrix group" << endl;
		cout << "type_G=";
		action_print_symmetry_group_type(cout, type_G);
		cout << endl;
		exit(1);
	}
#else
	return dimension;
#endif
}

field_theory::finite_field *action::matrix_group_finite_field()
{
	if (!is_matrix_group()) {
			cout << "action::matrix_group_finite_field is not a matrix group" << endl;
			exit(1);
	}
	else {
		algebra::matrix_group *M;

		M = get_matrix_group();
		return M->GFq;
	}
}

int action::is_semilinear_matrix_group()
{
	if (!is_matrix_group()) {
			cout << "action::is_semilinear_matrix_group is not a matrix group" << endl;
			exit(1);
	}
	else {
		algebra::matrix_group *M;

		M = get_matrix_group();
		if (M->f_semilinear) {
			return true;
		}
		else {
			return false;
		}
	}
}

int action::is_projective()
{
	if (!is_matrix_group()) {
			cout << "action::is_projective is not a matrix group" << endl;
			exit(1);
	}
	else {
		algebra::matrix_group *M;

		M = get_matrix_group();
		if (M->f_projective) {
			return true;
		}
		else {
			return false;
		}
	}
}

int action::is_affine()
{
	if (!is_matrix_group()) {
			cout << "action::is_affine is not a matrix group" << endl;
			exit(1);
	}
	else {
		algebra::matrix_group *M;

		M = get_matrix_group();
		if (M->f_affine) {
			return true;
		}
		else {
			return false;
		}
	}
}

int action::is_general_linear()
{
	if (!is_matrix_group()) {
			cout << "action::is_general_linear is not a matrix group" << endl;
			exit(1);
	}
	else {
		algebra::matrix_group *M;

		M = get_matrix_group();
		if (M->f_general_linear) {
			return true;
		}
		else {
			return false;
		}
	}
}

int action::is_matrix_group()
{
	if (type_G == matrix_group_t) {
			return true;
	}
	else if (type_G == action_on_orthogonal_t) {
			return true;
	}
	else {
		if (f_has_subaction) {
			return subaction->is_matrix_group();
		}
		return false;
	}
}

algebra::matrix_group *action::get_matrix_group()
{
	if (type_G == unknown_symmetry_group_t) {
		cout << "action::get_matrix_group type_G == unknown_symmetry_group_t" << endl;
		exit(1);
	}
	else if (type_G == matrix_group_t) {
		return G.matrix_grp;
	}
	else if (type_G == perm_group_t) {
		cout << "action::get_matrix_group type_G == perm_group_t" << endl;
		exit(1);
	}
	else if (type_G == wreath_product_t) {
		cout << "action::get_matrix_group type_G == wreath_product_t" << endl;
		exit(1);
	}
	else if (type_G == direct_product_t) {
		cout << "action::get_matrix_group type_G == direct_product_t" << endl;
		exit(1);
	}
	else if (type_G == action_on_sets_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_subgroups_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_k_subsets_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_pairs_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_ordered_pairs_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == base_change_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == product_action_t) {
		cout << "action::get_matrix_group type_G == product_action_t" << endl;
		exit(1);
	}
	else if (type_G == action_by_right_multiplication_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_restriction_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_conjugation_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_determinant_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_sign_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_grassmannian_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_spread_set_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_orthogonal_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_cosets_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_factor_space_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_wedge_product_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_representation_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_by_subfield_structure_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_bricks_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_andre_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_orbits_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_flags_t) {
		return subaction->get_matrix_group();
	}
	else if (type_G == action_on_homogeneous_polynomials_t) {
		return subaction->get_matrix_group();
	}
	else {
		cout << "action::get_matrix_group unknown type" << endl;
		exit(1);
	}
}



}}}


