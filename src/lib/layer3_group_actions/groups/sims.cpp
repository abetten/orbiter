// sims.cpp
//
// Anton Betten
// December 21, 2003

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace groups {

sims::sims()
{
	Record_birth();
	A = NULL;
	my_base_len = 0;

	gens.null();
	gens_inv.null();

	gen_depth = NULL;
	gen_perm = NULL;
	nb_gen = NULL;

	transversal_length = 0;
	path = NULL;

#if 0
	nb_images = 0;
	images = NULL;
#endif

	orbit_len = NULL;
	orbit = NULL;
	orbit_inv = NULL;
	prev = NULL;
	label = NULL;

	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	strip1 = NULL;
	strip2 = NULL;
	eltrk1 = NULL;
	eltrk2 = NULL;
	eltrk3 = NULL;
	cosetrep_tmp = NULL;
	schreier_gen = NULL;
	schreier_gen1 = NULL;
	cosetrep = NULL;
	//null();

}






sims::~sims()
{
	Record_death();
	int i;
	int f_v = false;
	
	if (f_v) {
		cout << "sims::~sims freeing gen_depth" << endl;
	}
	if (gen_depth) {
		FREE_int(gen_depth);
	}
	if (gen_perm) {
		FREE_int(gen_perm);
	}
	if (nb_gen) {
		FREE_int(nb_gen);
	}
	if (path) {
		FREE_int(path);
	}
		
	if (f_v) {
		cout << "sims::~sims freeing orbit, "
				"my_base_len=" << my_base_len << endl;
	}
	if (orbit) {
		for (i = 0; i < my_base_len; i++) {
			if (f_v) {
				cout << "sims::~sims freeing orbit i=" << i << endl;
			}
			if (f_v) {
				cout << "sims::~sims freeing orbit[i]" << endl;
			}
			FREE_int(orbit[i]);
			if (f_v) {
				cout << "sims::~sims freeing orbit_inv[i]" << endl;
			}
			FREE_int(orbit_inv[i]);
			if (f_v) {
				cout << "sims::~sims freeing prev[i]" << endl;
			}
			FREE_int(prev[i]);
			if (f_v) {
				cout << "sims::~sims freeing label[i]" << endl;
			}
			FREE_int(label[i]);
		}
		if (f_v) {
			cout << "sims::~sims freeing orbit"<< endl;
		}
		FREE_pint(orbit);
		FREE_pint(orbit_inv);
		FREE_pint(prev);
		FREE_pint(label);
	}
	if (f_v) {
		cout << "sims::~sims freeing orbit_len" << endl;
	}
	if (orbit_len) {
		FREE_int(orbit_len);
	}
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
	if (strip1) {
		FREE_int(strip1);
	}
	if (strip2) {
		FREE_int(strip2);
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
	if (cosetrep) {
		FREE_int(cosetrep);
	}
	if (cosetrep_tmp) {
		FREE_int(cosetrep_tmp);
	}
	if (schreier_gen) {
		FREE_int(schreier_gen);
	}
	if (schreier_gen1) {
		FREE_int(schreier_gen1);
	}
	A = NULL;

	if (f_v) {
		cout << "sims::~sims done" << endl;
	}
}


void sims::init(
		actions::action *A, int verbose_level)
// initializes the trivial group with the base as given in A
{
	int i;
	int f_v = (verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 2);

	if (f_v) {
		cout << "sims::init action=" << A->label << endl;
	}
	
	if (f_v) {
		cout << "sims::init before init_without_base" << endl;
	}
	init_without_base(A, 0 /* verbose_level */);
	if (f_v) {
		cout << "sims::init after init_without_base" << endl;
	}
	
#if 0
	if (A->Stabilizer_chain) {
		my_base_len = A->base_len();
	}
	else {
		cout << "sims::init A->Stabilizer_chain == NULL, "
				"setting my_base_len to degree" << endl;
		my_base_len = A->degree;
	}
#endif
	
	if (f_vv) {
		cout << "sims::init my_base_len=" << my_base_len << endl;
	}
	if (f_vv) {
		cout << "sims::init allocating orbit" << endl;
	}
	orbit = NEW_pint(my_base_len);
	if (f_vv) {
		cout << "sims::init allocating orbit_inv" << endl;
	}
	orbit_inv = NEW_pint(my_base_len);
	if (f_vv) {
		cout << "sims::init allocating prev" << endl;
	}
	prev = NEW_pint(my_base_len);
	if (f_vv) {
		cout << "sims::init allocating label" << endl;
	}
	label = NEW_pint(my_base_len);

	transversal_length = A->degree;
	if (f_vv) {
		cout << "sims::init "
				"transversal_length=" << transversal_length << endl;
	}
	for (i = 0; i < my_base_len; i++) {
		if (f_vv) {
			cout << "sims::init allocating "
					"orbit " << i << " / " << my_base_len << endl;
		}
		orbit[i] = NEW_int(transversal_length);
		orbit_inv[i] = NEW_int(transversal_length);
		prev[i] = NEW_int(transversal_length);
		label[i] = NEW_int(transversal_length);
	}
	FREE_int(nb_gen);
	
	
	nb_gen = NEW_int(my_base_len + 1);
	Int_vec_zero(nb_gen, my_base_len + 1);
	
	path = NEW_int(my_base_len);

	orbit_len = NEW_int(my_base_len);
	
	for (i = 0; i < my_base_len; i++) {
		if (f_vv) {
			cout << "sims::init before "
					"initialize_table " << i << " / " << my_base_len << endl;
		}
		initialize_table(i, 0 /*verbose_level*/);
		orbit_len[i] = 1;
	}
	if (f_v) {
		cout << "sims::init done" << endl;
	}
	
}


void sims::init_without_base(
		actions::action *A, int verbose_level)
{	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::init_without_base "
				"action=" << A->label << endl;
		cout << "sims::init_without_base "
				"A->degree=" << A->degree << endl;
	}
	sims::A = A;
	//nb_images = 0;
	//images = NULL;
	
	transversal_length = A->degree;
	my_base_len = A->base_len();
	if (f_v) {
		cout << "sims::init_without_base "
				"my_base_len=" << my_base_len << endl;
	}
	
	gens.init(A, verbose_level - 2);
	gens_inv.init(A, verbose_level - 2);
	
	nb_gen = NEW_int(my_base_len + 1);

	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	Elt4 = NEW_int(A->elt_size_in_int);
	strip1 = NEW_int(A->elt_size_in_int);
	strip2 = NEW_int(A->elt_size_in_int);
	eltrk1 = NEW_int(A->elt_size_in_int);
	eltrk2 = NEW_int(A->elt_size_in_int);
	eltrk3 = NEW_int(A->elt_size_in_int);
	cosetrep = NEW_int(A->elt_size_in_int);
	cosetrep_tmp = NEW_int(A->elt_size_in_int);
	schreier_gen = NEW_int(A->elt_size_in_int);
	schreier_gen1 = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "sims::init_without_base done" << endl;
	}
}

void sims::reallocate_base(
		int old_base_len, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	int *old_nb_gen = nb_gen;
	int *old_path = path;
	int *old_orbit_len = orbit_len;
	int **old_orbit = orbit;
	int **old_orbit_inv = orbit_inv;
	int **old_prev = prev;
	int **old_label = label;
	
	if (f_v) {
		cout << "sims::reallocate_base" << endl;
	}
	if (f_v) {
		cout << "sims::reallocate_base A = " << A->label << endl;
	}
	if (f_v) {
		cout << "sims::reallocate_base from " 
			<< old_base_len << " to " << A->base_len() << endl;
	}

	if (A->base_len() < old_base_len) {
		cout << "sims::reallocate_base "
				"error: the new base is shorter than the old" << endl;
		exit(1);
	}

	my_base_len = A->base_len();
	if (f_v) {
		cout << "sims::reallocate_base "
				"my_base_len=" << my_base_len << endl;
	}
	
	nb_gen = NEW_int(my_base_len + 1);
	path = NEW_int(my_base_len);
	orbit_len = NEW_int(my_base_len);
	orbit = NEW_pint(my_base_len);
	orbit_inv = NEW_pint(my_base_len);
	prev = NEW_pint(my_base_len);
	label = NEW_pint(my_base_len);

	for (i = 0; i < old_base_len; i++) {
		if (f_v) {
			cout << "sims::reallocate_base "
					"i=" << i << " / " << old_base_len << endl;
		}
		nb_gen[i] = old_nb_gen[i];
		path[i] = old_path[i];
		orbit_len[i] = old_orbit_len[i];
		orbit[i] = old_orbit[i];
		orbit_inv[i] = old_orbit_inv[i];
		prev[i] = old_prev[i];
		label[i] = old_label[i];
	}
	for (i = old_base_len; i < my_base_len; i++) {
		if (f_v) {
			cout << "sims::reallocate_base "
					"i=" << i << " / "
					<< old_base_len << endl;
		}
		nb_gen[i] = 0;
		path[i] = 0;
		orbit[i] = NEW_int(A->degree);
		orbit_inv[i] = NEW_int(A->degree);
		prev[i] = NEW_int(A->degree);
		label[i] = NEW_int(A->degree);
		if (f_v) {
			cout << "sims::reallocate_base "
					"before initialize_table" << endl;
		}
		initialize_table(i, 0 /* verbose_level */);
		if (f_v) {
			cout << "sims::reallocate_base "
					"after initialize_table" << endl;
		}
		if (f_v) {
			cout << "sims::reallocate_base "
					"before init_trivial_orbit" << endl;
		}
		init_trivial_orbit(i, verbose_level);
		if (f_v) {
			cout << "sims::reallocate_base "
					"after init_trivial_orbit" << endl;
		}
	}
	nb_gen[my_base_len] = 0;
	if (old_nb_gen) {
		FREE_int(old_nb_gen);
	}
	if (old_path) {
		FREE_int(old_path);
	}
	if (old_orbit_len) {
		FREE_int(old_orbit_len);
	}
	if (old_orbit) {
		FREE_pint(old_orbit);
	}
	if (old_orbit_inv) {
		FREE_pint(old_orbit_inv);
	}
	if (old_prev) {
		FREE_pint(old_prev);
	}
	if (old_label) {
		FREE_pint(old_label);
	}
	if (f_v) {
		cout << "sims::reallocate_base done" << endl;
	}
}

void sims::initialize_table(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	
	if (f_v) {
		cout << "sims::initialize_table" << endl;
	}
	Combi.Permutations->perm_identity(orbit[i], transversal_length);
	Combi.Permutations->perm_identity(orbit_inv[i], transversal_length);

	Int_vec_mone(prev[i], transversal_length);
	Int_vec_mone(label[i], transversal_length);

	orbit_len[i] = 0;
	if (f_v) {
		cout << "sims::initialize_table done" << endl;
	}
}

void sims::init_trivial_group(
		int verbose_level)
// clears the generators array, 
// and sets the i-th transversal to contain
// only the i-th base point (for all i).
{
	int f_v = (verbose_level >= 2);
	int f_vv = (verbose_level >= 3);
	int i;
	
	if (f_v) {
		cout << "sims::init_trivial_group" << endl;
	}
	if (f_v) {
		cout << "sims::init_trivial_group "
				"A->label = " << A->label << endl;
	}

	if (A->Stabilizer_chain == NULL) {
		cout << "sims::init_trivial_group "
				"A->Stabilizer_chain == NULL" << endl;
		return;
	}

	if (my_base_len != A->base_len()) {
		cout << "sims::init_trivial_group: "
				"my_base_len != A->base_len" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "sims::init_trivial_group "
				"before init_generators" << endl;
	}
	init_generators(0, NULL, 0 /*verbose_level - 3*/);
	if (f_v) {
		cout << "sims::init_trivial_group "
				"after init_generators" << endl;
	}

	if (f_v) {
		cout << "sims::init_trivial_group "
				"setting initial orbits, my_base_len = " << my_base_len << endl;
	}
	for (i = 0; i < my_base_len; i++) {
		if (f_vv) {
			cout << "sims::init_trivial_group "
					"before init_trivial_orbit i=" << i << endl;
		}
		init_trivial_orbit(i, 0 /*verbose_level*/);
	}
	if (f_v) {
		cout << "sims::init_trivial_group "
				"after initial orbits" << endl;
	}
	if (f_v) {
		cout << "sims::init_trivial_group done" << endl;
	}
}

void sims::init_trivial_orbit(
		int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int coset_of_base_point;
	int bi;
	
	if (f_v) {
		cout << "sims::init_trivial_orbit "
				"i=" << i << endl;
	}
	if (my_base_len != A->base_len()) {
		cout << "sims::init_trivial_orbit: "
				"my_base_len != A->base_len" << endl;
		exit(1);
	}
	bi = A->base_i(i);
	if (f_v) {
		cout << "sims::init_trivial_orbit "
				"bi=" << bi << endl;
	}

	if (f_v) {
		cout << "sims::init_trivial_orbit "
				"before get_orbit_inv" << endl;
	}
	coset_of_base_point = get_orbit_inv(i, bi);
	if (f_v) {
		cout << "sims::init_trivial_orbit "
				"after get_orbit_inv" << endl;
		cout << "sims::init_trivial_orbit "
				"coset_of_base_point = "
				<< coset_of_base_point << endl;
	}

	if (coset_of_base_point) {
		swap_points(i, coset_of_base_point, 0);
	}
	//cout << "sims::init_trivial_orbit " << i
	// << " : " << A->base[i] << endl;
	//cout << "orbit[i][0] = " << orbit[i][0] << endl;
	orbit_len[i] = 1;
	if (f_v) {
		cout << "sims::init_trivial_orbit "
				"i=" << i << " done" << endl;
	}
}

void sims::init_generators(
		data_structures_groups::vector_ge &generators,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "sims::init_generators" << endl;
		cout << "generators.len=" << generators.len << endl;
	}
	if (generators.len) {
		init_generators(generators.len,
				generators.ith(0), verbose_level);
	}
	else {
		init_generators(generators.len, NULL, verbose_level);
	}
	if (f_v) {
		cout << "sims::init_generators done" << endl;
	}
}

void sims::init_generators(
		int nb, int *elt,
		int verbose_level)
// copies the given elements into the generator array, 
// then computes depth and perm
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;
	
	if (f_v) {
		cout << "sims::init_generators" << endl;
		cout << "sims::init_generators A->label = " << A->label << endl;
		cout << "sims::init_generators A->degree = " << A->degree << endl;
		cout << "sims::init_generators number of generators = " << nb << endl;
	}
	gens.allocate(nb, verbose_level - 2);
	gens_inv.allocate(nb, verbose_level - 2);
	for (i = 0; i < nb; i++) {
		if (f_vv) {
			cout << "sims::init_generators "
					"i = " << i << " / " << nb << ":" << endl;
		}
		if (f_vv) {
			cout << "sims::init_generators "
					"i = " << i << " / " << nb
					<< " : before gens.copy_in" << endl;
		}
		gens.copy_in(i, elt + i * A->elt_size_in_int);
		if (f_vv) {
			cout << "sims::init_generators "
					"i = " << i << " / " << nb
					<< " : after gens.copy_in" << endl;
		}
		if (f_vvv) {
			A->Group_element->element_print_quick(
					elt + i * A->elt_size_in_int, cout);
		}
		if (f_vv) {
			cout << "sims::init_generators "
					"i = " << i << " / " << nb
					<< " : before A->Group_element->element_invert" << endl;
		}
		A->Group_element->element_invert(
				elt + i * A->elt_size_in_int,
				gens_inv.ith(i), verbose_level - 1);
		if (f_vv) {
			cout << "sims::init_generators "
					"i = " << i << " / " << nb
					<< " : after A->Group_element->element_invert" << endl;
		}
	}

	if (f_v) {
		cout << "sims::init_generators "
				"before init_generator_depth_and_perm" << endl;
	}
	init_generator_depth_and_perm(verbose_level);
	if (f_v) {
		cout << "sims::init_generators "
				"after init_generator_depth_and_perm" << endl;
	}
	if (f_v) {
		cout << "sims::init_generators done" << endl;
	}
}

void sims::init_generators_by_hdl(
		int nb_gen, int *gen_hdl, int verbose_level)
{
	int i;
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "sims::init_generators_by_hdl" << endl;
	}
	gens.allocate(nb_gen, verbose_level - 2);
	gens_inv.allocate(nb_gen, verbose_level - 2);

	for (i = 0; i < nb_gen; i++) {
		//cout << "sims::init_generators i = " << i << endl;
		A->Group_element->element_retrieve(gen_hdl[i], gens.ith(i), 0);
		A->Group_element->element_invert(gens.ith(i), gens_inv.ith(i), 0);
	}
	//init_images(nb_gen);
	init_generator_depth_and_perm(0);
	if (f_v) {
		cout << "sims::init_generators_by_hdl done" << endl;
	}

}

void sims::init_generator_depth_and_perm(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, d;
	
	if (f_v) {
		cout << "sims::init_generator_depth_and_perm" << endl;
		cout << "gens.len=" << gens.len << endl;
		cout << "action=" << A->label << " : ";
		A->print_info();
		cout << endl;
	}
	if (my_base_len != A->base_len()) {
		cout << "sims::init_generator_depth_and_perm "
				"my_base_len != A->base_len" << endl;
		exit(1);
	}

	Int_vec_zero(nb_gen, A->base_len() + 1);

	gen_depth = NEW_int(gens.len);
	gen_perm = NEW_int(gens.len);
	for (i = 0; i < gens.len; i++) {
		gen_perm[i] = i;
		gen_depth[i] = generator_depth_in_stabilizer_chain(i, verbose_level);
		if (f_vv) {
			cout << "sims::init_generator_depth_and_perm "
					"generator " << i
					<< " has depth " << gen_depth[i] << endl;
		}
		if (i) {
			if (gen_depth[i] > gen_depth[i - 1]) {
				cout << "sims::init_generator_depth_and_perm "
						"generators must be of decreasing depth" << endl;
				cout << "i=" << i << endl;
				cout << "gens.len=" << gens.len << endl;
				cout << "gen_depth[i]=" << gen_depth[i] << endl;
				cout << "gen_depth[i - 1]=" << gen_depth[i - 1] << endl;
				cout << "A: ";
				A->print_info();
				cout << "gens=" << endl;
				gens.print(cout);
				cout << "generator depth = ";
				Int_vec_print(cout, gen_depth, i + 1);
				cout << endl;



				for (j = 0; j <= i; j++) {

					std::string s;

					s = stringify_base_images(
						j, 0 /* verbose_level*/);

					cout << "base images of generator "
							<< j << " are " << s << endl;
				}
				exit(1);
			}
		}
	}
	nb_gen[A->base_len()] = 0;
	for (i = 0; i < gens.len; i++) {
		d = gen_depth[i];
		for (j = d; j >= 0; j--) {
			nb_gen[j]++;
		}
	}
	if (f_v) {
		print_generator_depth_and_perm();
	}
	if (f_v) {
		cout << "sims::init_generator_depth_and_perm done" << endl;
	}
}

void sims::add_generator(
		int *elt, int verbose_level)
// adds elt to list of generators, 
// computes the depth of the element, 
// updates the arrays gen_depth, gen_perm and nb_gen accordingly
// does not change the transversals
{
	int f_v = (verbose_level >= 1);
	int old_nb_gen, idx, depth, i;
	int *new_gen_depth;
	int *new_gen_perm;
	
	if (f_v) {
		cout << "sims::add_generator "
				"verbose_level=" << verbose_level << endl;
		cout << "sims::add_generator "
				"generator no " << gens.len << " is:" << endl;
		A->Group_element->element_print_quick(elt, cout);
		cout << endl;
		cout << "sims::add_generator "
				"my_base_len=" << my_base_len << endl;
		cout << "sims::add_generator "
				"A->base_len()=" << A->base_len() << endl;
	}
	if (f_v) {
		cout << "sims::add_generator "
				"before adding the generator:" << endl;
		print_generator_depth_and_perm();
	}
	if (my_base_len != A->base_len()) {
		cout << "sims::add_generator: "
				"my_base_len != A->base_len" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "sims::add_generator "
				"allocating new_gen_depth and new_gen_perm" << endl;
	}

	new_gen_depth = NEW_int(gens.len + 1);
	new_gen_perm = NEW_int(gens.len + 1);


	if (f_v) {
		cout << "sims::add_generator "
				"copying gen_depth and gen_perm over" << endl;
	}
	old_nb_gen = idx = gens.len;
	for (i = 0; i < old_nb_gen; i++) {
		new_gen_depth[i] = gen_depth[i];
		new_gen_perm[i] = gen_perm[i];
	}
	if (gen_depth) {
		if (f_v) {
			cout << "sims::add_generator "
					"freeing gen_depth/gen_perm" << endl;
		}
		FREE_int(gen_depth);
		FREE_int(gen_perm);
	}
	gen_depth = new_gen_depth;
	gen_perm = new_gen_perm;
	
	if (f_v) {
		cout << "sims::add_generator "
				"before gens.append" << endl;
	}
	gens.append(elt, verbose_level - 2);
	if (f_v) {
		cout << "sims::add_generator "
				"before gens_inv.append" << endl;
	}
	gens_inv.append(elt, verbose_level - 2);
	if (f_v) {
		cout << "sims::add_generator "
				"before A->element_invert" << endl;
	}
	A->Group_element->element_invert(
			elt, gens_inv.ith(idx), false);
	
	if (f_v) {
		cout << "sims::add_generator "
				"before generator_depth_in_stabilizer_chain" << endl;
	}
	depth = generator_depth_in_stabilizer_chain(idx, verbose_level);
	if (f_v) {
		cout << "sims::add_generator "
				"depth = " << depth << endl;
	}
	new_gen_depth[idx] = depth;
	for (i = old_nb_gen - 1; i >= nb_gen[depth]; i--) {
		gen_perm[i + 1] = gen_perm[i];
	}
	gen_perm[nb_gen[depth]] = idx;
	for (i = depth; i >= 0; i--) {
		nb_gen[i]++;
	}

	if (f_v) {
		cout << "sims::add_generator "
				"after adding the generator:" << endl;
		print_generator_depth_and_perm();
	}

	if (f_v) {
		cout << "sims::add_generator done" << endl;
	}
}




int sims::generator_depth_in_stabilizer_chain(
		int gen_idx, int verbose_level)
// returns the index of the first base point 
// which is moved by a given generator. 
// previously called generator_depth
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::generator_depth_in_stabilizer_chain" << endl;
		cout << "sims::generator_depth_in_stabilizer_chain A = " << A->label << endl;
	}
	int i, bi, j;
	
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		j = get_image(bi, gen_idx);
		if (f_v) {
			cout << "sims::generator_depth_in_stabilizer_chain "
					"i = " << i << " : bi = " << bi << " : j = " << j << endl;
		}
		if (j != bi) {
			if (f_v) {
				cout << "sims::generator_depth_in_stabilizer_chain "
						"depth is equal to " << i << endl;
			}
			return i;
		}
	}
	if (f_v) {
		cout << "sims::generator_depth_in_stabilizer_chain "
				"depth is equal to " << A->base_len() << endl;
	}
	return A->base_len();
}

std::string sims::stringify_base_images(
		int gen_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::stringify_base_images" << endl;
		cout << "sims::stringify_base_images A = " << A->label << endl;
	}
	string s;
	int i, bi, j;

	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		j = get_image(bi, gen_idx);
		s += std::to_string(j);
		if (i < A->base_len() - 1) {
			s += ",";
		}
	}
	return s;
}


int sims::depth_in_stabilizer_chain(
		int *elt, int verbose_level)
// returns the index of the first base point 
// which is moved by the given element
// previously called generator_depth
// this function is not used anywhere!
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::depth_in_stabilizer_chain" << endl;
	}
	int i, bi, j;
	
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		j = get_image(bi, elt);
		if (f_v) {
			cout << "sims::depth_in_stabilizer_chain "
					"i = " << i << " : bi = " << bi << " : j = " << j << endl;
		}
		if (j != bi) {
			if (f_v) {
				cout << "sims::depth_in_stabilizer_chain "
						"depth is equal to " << i << endl;
			}
			return i;
		}
	}
	if (f_v) {
		cout << "sims::depth_in_stabilizer_chain "
				"depth is equal to " << A->base_len() << endl;
	}
	return A->base_len();
}

void sims::group_order(
		algebra::ring_theory::longinteger_object &go)
{
	algebra::ring_theory::longinteger_domain D;

	//cout << "sims::group_order before D.multiply_up" << endl;
	//cout << "A->base_len=" << A->base_len << endl;
	//cout << "orbit_len=";
	//int_vec_print(cout, orbit_len, A->base_len);
	//cout << endl;
	D.multiply_up(go, orbit_len, my_base_len /*A->base_len()*/, 0);
	//cout << "sims::group_order after D.multiply_up" << endl;
}

std::string sims::stringify_group_order()
{
	std::string s;
	algebra::ring_theory::longinteger_object go;

	group_order(go);

	s = go.stringify() + " = ";
	//cout << "Order " << go << " = ";
	//int_vec_print(cout, Sims->orbit_len, base_len());
	for (int t = 0; t < A->base_len(); t++) {
		s += std::to_string(get_orbit_length(t));
		if (t < A->base_len() - 1) {
			s += " * ";
		}
	}
	return s;
}

void sims::group_order_verbose(
		algebra::ring_theory::longinteger_object &go,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_domain D;
	
	if (f_v) {
		cout << "sims::group_order_verbose" << endl;
	}
	//cout << "sims::group_order before D.multiply_up" << endl;
	//cout << "A->base_len=" << A->base_len << endl;
	//cout << "orbit_len=";
	//int_vec_print(cout, orbit_len, A->base_len);
	//cout << endl;
	D.multiply_up(go, orbit_len, my_base_len /*A->base_len()*/, verbose_level);
	//cout << "sims::group_order after D.multiply_up" << endl;
	if (f_v) {
		cout << "sims::group_order_verbose done" << endl;
	}
}

void sims::subgroup_order_verbose(
		algebra::ring_theory::longinteger_object &go,
		int level, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	algebra::ring_theory::longinteger_domain D;

	if (f_v) {
		cout << "sims::subgroup_order_verbose" << endl;
	}
	//cout << "sims::group_order before D.multiply_up" << endl;
	//cout << "A->base_len=" << A->base_len << endl;
	//cout << "orbit_len=";
	//int_vec_print(cout, orbit_len, A->base_len);
	//cout << endl;
	D.multiply_up(go, orbit_len + level, my_base_len - level, verbose_level);
	//cout << "sims::group_order after D.multiply_up" << endl;
	if (f_v) {
		cout << "sims::subgroup_order_verbose done" << endl;
	}
}

long int sims::group_order_lint()
{
	algebra::ring_theory::longinteger_object go;

	group_order(go);
	return go.as_lint();
}

int sims::is_trivial_group()
{
	int j = last_moved_base_point();
	
	if (j == -1) {
		return true;
	}
	else {
		return false;
	}
}

int sims::last_moved_base_point()
// j == -1 means the group is trivial
{
	int j;
	
	for (j = A->base_len() - 1; j >= 0; j--) {
		if (orbit_len[j] != 1) {
			break;
		}
	}
	return j;
}

int sims::get_image(
		int i, int gen_idx)
// get the image of a point i under
// generator gen_idx, goes through a
// table of stored images by default.
// Computes the image only if not yet available.
{
	int a;
	
#if 0
	if (nb_images == 0 || images == NULL) {
		a = A->Group_element->element_image_of(i, gens.ith(gen_idx), 0);
		return a;
		//cout << "sims::get_image() images == NULL" << endl;
		//exit(1);
	}
	a = images[gen_idx][i];
#endif
#if 0
	if (a == -1) {
		a = A->Group_element->element_image_of(i, gens.ith(gen_idx), 0);
		images[gen_idx][i] = a;
		//images[gen_idx][A->degree + a] = i;
	}
#endif

	a = A->Group_element->element_image_of(i, gens.ith(gen_idx), 0);
	return a;
}

int sims::get_image(
		int i, int *elt)
// get the image of a point i under a given group element,
// does not goes through a table.
{
	return A->Group_element->element_image_of(i, elt, false);
}

void sims::swap_points(
		int lvl, int i, int j)
// swaps two points given by their cosets
{
	int pi, pj;
	
	pi = orbit[lvl][i];
	pj = orbit[lvl][j];
	orbit[lvl][i] = pj;
	orbit[lvl][j] = pi;
	orbit_inv[lvl][pi] = j;
	orbit_inv[lvl][pj] = i;
}

void sims::path_unrank_lint(
		long int a)
{
	long int h, l;
	
	for (h = A->base_len() - 1; h >= 0; h--) {
		l = orbit_len[h];

		path[h] = a % l;
		a = a / l;
	}
}

int sims::first_moved_based_on_path(
		long int a)
{

	path_unrank_lint(a);

	int h;

	for (h = 0; h < A->base_len(); h++) {
		if (path[h] != 0) {
			return h;
		}
	}
	return A->base_len();
}

int sims::advance_at_given_level(
		long int a, long int &b, int level)
{

	path_unrank_lint(a);

	int l, t;


	l = orbit_len[level];

	if (path[level] < l - 1) {
		path[level]++;
		for (t = level + 1; t < A->base_len(); t++) {
			path[t] = 0;
		}
		b = path_rank_lint();
		return true;
	}
	else {
		for (t = level + 1; t < A->base_len(); t++) {
			l = orbit_len[t];
			path[t] = l - 1;
		}
		b = path_rank_lint();
		return false;
#if 0
		for (h = level - 1; h >= 0; h--) {
			l = orbit_len[h];
			if (path[h] < l - 1) {
				path[h]++;
				for (t = h + 1; t < A->base_len(); t++) {
					path[t] = 0;
				}
				b = path_rank_lint();
				return true;
			}
		}
		b = a + 1;
		return false;
#endif
	}
}



void sims::make_path(
		algebra::ring_theory::longinteger_object &a,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ii, l, r;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object q;

	if (f_v) {
		cout << "sims::make_path rk=" << a << endl;
	}
	for (ii = A->base_len() - 1; ii >= 0; ii--) {
		l = orbit_len[ii];

		D.integral_division_by_int(a, l, q, r);
		q.assign_to(a);

		path[ii] = r;
		//cout << r << " ";
	}
}

long int sims::path_rank_lint()
{
	long int h, a;
	
	a = 0;
	for (h = 0; h < A->base_len(); h++) {
		if (h) {
			a *= orbit_len[h];
		}
		a += path[h];
	}
	return a;
}

void sims::element_from_path(
		int *elt, int verbose_level)
// given coset representatives in path[], the corresponding 
// element is multiplied.
// uses eltrk1, eltrk2, eltrk3
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;
	
	if (f_v) {
		cout << "sims::element_from_path" << endl;
	}
	if (f_vv) {
		cout << "path=";
		Int_vec_print(cout, path, A->base_len());
		cout << endl;
		cout << "A->degree=" << A->degree << endl;
	}
#if 0
	if (f_v) {
		cout << "i : orbit[0][i] : orbit_inv[0][i] : "
				"prev[0][i] : label[0][i]" << endl;
		for (i = 0; i < A->degree; i++) {
			cout << setw(5) << i 
				<< " : " << setw(5) << orbit[0][i] 
				<< " : " << setw(5) << orbit_inv[0][i] 
				<< " : " << setw(5) << prev[0][i] 
				<< " : " << setw(5) << label[0][i] 
				<< endl;
			}
		}
#endif
	
	A->Group_element->element_one(eltrk1, false);

	for (i = 0; i < A->base_len(); i++) {
		j = path[i];
		if (f_v) {
			cout << "sims::element_from_path level "
					<< i << " coset " << j << " before coset_rep" << endl;
		}
		coset_rep(eltrk3, i, j, verbose_level);


		if (f_v) {
			cout << "sims::element_from_path level "
					<< i << " coset " << j << " after coset_rep" << endl;
		}

		if (f_vv) {
			cout << "sims::element_from_path level "
					<< i << " coset " << j << ":" << endl;
			cout << "cosetrep:" << endl;
			A->Group_element->element_print_quick(eltrk3, cout);
			cout << endl;
		}
		
		//A->element_print_as_permutation(cosetrep, cout);
		//cout << endl;
		
		// pre multiply the coset representative because
		// we are going in oder of increasing base depth

		A->Group_element->element_mult(eltrk3, eltrk1, eltrk2, 0);
		A->Group_element->element_move(eltrk2, eltrk1, 0);
	}
	A->Group_element->element_move(eltrk1, elt, 0);
	if (f_v) {
		cout << "sims::element_from_path done" << endl;
	}
}

void sims::element_from_path_inv(
		int *elt)
// very specialized routine, used in backtrack.cpp
// action_is_minimal_recursion
// used coset_rep_inv instead of coset_rep,
// multiplies left-to-right
//
// given coset representatives in path[],
// the corresponding
// element is multiplied.
// uses eltrk1, eltrk2
{
	int i, j;
	
#if 0
	cout << "sims::element_from_path() path=";
	for (i = 0; i < A->base_len; i++) {
		cout << path[i] << " ";
	}
	cout << endl;
#endif
	A->Group_element->element_one(eltrk1, false);
	for (i = 0; i < A->base_len(); i++) {
		j = path[i];
		
		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);
		
		//A->element_print_as_permutation(eltrk3, cout);
		//cout << endl;
		
		// pre multiply the coset representative:
		A->Group_element->element_mult(eltrk1, eltrk3, eltrk2, false);
		A->Group_element->element_move(eltrk2, eltrk1, false);
	}
	A->Group_element->element_move(eltrk1, elt, false);
}


void sims::element_unrank(
		algebra::ring_theory::longinteger_object &a,
		int *elt, int verbose_level)
// Returns group element whose rank is a. 
// the elements represented by the chain
// are enumerated 0, ... go - 1
// with the convention that 0 always stands
// for the identity element.
// The computed group element will be computed into Elt1
{
	int f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "sims::element_unrank rk=" << a << endl;
	}

	make_path(
			a,
			verbose_level);

#if 0
	int ii, l, r;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object q;

	for (ii = A->base_len() - 1; ii >= 0; ii--) {
		l = orbit_len[ii];

		D.integral_division_by_int(a, l, q, r);
		q.assign_to(a);
		
		path[ii] = r;
		//cout << r << " ";
	}
	//cout << endl;
#endif

	if (f_v) {
		cout << "sims::element_unrank path=";
		Int_vec_print(cout, path, A->base_len());
		cout << endl;
	}
	element_from_path(elt, 0);
	if (f_v) {
		cout << "sims::element_unrank done" << endl;
	}
}

void sims::element_unrank(
		algebra::ring_theory::longinteger_object &a, int *elt)
// Returns group element whose rank is a. 
// the elements represented by the chain
// are enumerated 0, ... go - 1
// with the convention that 0 always stands
// for the identity element.
// The computed group element will be computed into Elt1
{
	int ii, l, r;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object q;
	
	for (ii = A->base_len() - 1; ii >= 0; ii--) {
		l = orbit_len[ii];

		D.integral_division_by_int(a, l, q, r);
		q.assign_to(a);
		
		path[ii] = r;
		//cout << r << " ";
	}
	//cout << endl;
	element_from_path(elt, 0);
}

void sims::element_rank(
		algebra::ring_theory::longinteger_object &a, int *elt)
// Computes the rank of the element in elt into a.
// uses eltrk1, eltrk2
{
	long int i, j, bi, jj, l;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object b, c;
	
	A->Group_element->element_move(elt, eltrk1, false);
	a.zero();
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		l = orbit_len[i];
		
		if (i > 0) {
			b.create(l);
			D.mult(a, b, c);
			c.assign_to(a);
		}
		
		jj = A->Group_element->element_image_of(bi, eltrk1, false);
		//cout << "at level " << i << ", maps bi = "
		// << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {
			cout << "sims::element_rank j >= orbit_len[i]" << endl;
			cout << "i=" << i << endl;
			cout << "jj=bi^elt=" << jj << endl;
			cout << "j=orbit_inv[i][jj]=" << j << endl;
			cout << "base=";
			Lint_vec_print(cout, A->get_base(), A->base_len());
			cout << endl;
			cout << "orbit_len=";
			Int_vec_print(cout, orbit_len, A->base_len());
			cout << endl;
			cout << "elt=" << endl;
			A->Group_element->element_print(eltrk1, cout);
			exit(1);
		}
		b.create(j);
		D.add(a, b, c);
		c.assign_to(a);
		
		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);

		A->Group_element->element_mult(eltrk1, eltrk3, eltrk2, false);
		A->Group_element->element_move(eltrk2, eltrk1, false);
	}
}


int sims::test_membership_and_rank_element(
		algebra::ring_theory::longinteger_object &a,
		int *elt, int verbose_level)
// Computes the rank of the element in elt into a.
// uses eltrk1, eltrk2
// returns false if the element is not a member of the group
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::test_membership_and_rank_element" << endl;
	}

	long int i, j, bi, jj, l;
	algebra::ring_theory::longinteger_domain D;
	algebra::ring_theory::longinteger_object b, c;

	A->Group_element->element_move(elt, eltrk1, false);
	a.zero();
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		l = orbit_len[i];

		if (i > 0) {
			b.create(l);
			D.mult(a, b, c);
			c.assign_to(a);
		}

		jj = A->Group_element->element_image_of(bi, eltrk1, false);
		//cout << "at level " << i << ", maps bi = "
		// << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {

			if (f_v) {
				cout << "sims::test_membership_and_rank_element "
						"element is not a member of the group" << endl;
			}
			return false;

#if 0
			cout << "sims::element_rank j >= orbit_len[i]" << endl;
			cout << "i=" << i << endl;
			cout << "jj=bi^elt=" << jj << endl;
			cout << "j=orbit_inv[i][jj]=" << j << endl;
			cout << "base=";
			Lint_vec_print(cout, A->get_base(), A->base_len());
			cout << endl;
			cout << "orbit_len=";
			Int_vec_print(cout, orbit_len, A->base_len());
			cout << endl;
			cout << "elt=" << endl;
			A->Group_element->element_print(eltrk1, cout);
			exit(1);
#endif
		}
		b.create(j);
		D.add(a, b, c);
		c.assign_to(a);

		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);

		A->Group_element->element_mult(eltrk1, eltrk3, eltrk2, false);
		A->Group_element->element_move(eltrk2, eltrk1, false);
	}
	if (f_v) {
		cout << "sims::test_membership_and_rank_element done" << endl;
	}
	return true;
}



void sims::element_unrank_lint(
		long int rk, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//longinteger_object a;

	if (f_v) {
		cout << "sims::element_unrank_lint rk=" << rk << endl;
	}
	//a.create(rk);
	//element_unrank(a, Elt);
	path_unrank_lint(rk);
	if (f_v) {
		cout << "sims::element_unrank_lint path=";
		Int_vec_print(cout, path, A->base_len());
		cout << endl;
	}
	element_from_path(Elt, 0);

}

void sims::element_unrank_lint(
		long int rk, int *Elt)
{
	//longinteger_object a;
	
	//a.create(rk);
	element_unrank_lint(rk, Elt, 0);
}

long int sims::element_rank_lint(
		int *Elt)
{
	algebra::ring_theory::longinteger_object a;
	
	element_rank(a, Elt);
	return a.as_lint();
}

int sims::is_element_of(
		int *elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, bi, jj; //, l;
	int ret = true;
	
	if (f_v) {
		cout << "sims::is_element_of" << endl;
	}
	A->Group_element->element_move(elt, eltrk1, false);
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		//l = orbit_len[i];
		
		
		jj = A->Group_element->element_image_of(
				bi, eltrk1, false);
		//cout << "at level " << i << ", maps bi = "
		// << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {
			ret = false;
			break;
		}
		
		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);

		A->Group_element->element_mult(eltrk1, eltrk3, eltrk2, false);
		A->Group_element->element_move(eltrk2, eltrk1, false);
	}
	if (f_v) {
		cout << "sims::is_element_of done ret = " << ret << endl;
	}
	return ret;
}

void sims::test_element_rank_unrank()
{
	algebra::ring_theory::longinteger_object go, a, b;
	int i, j, goi;
	int *elt = NEW_int(A->elt_size_in_int);
	
	group_order(go);
	goi = go.as_int();
	for (i = 0; i < goi; i++) {
		a.create(i);
		element_unrank(a, elt);
		cout << i << " : " << endl;
		A->Group_element->element_print(elt, cout);
		cout << " : ";
		A->Group_element->element_print_as_permutation(elt, cout);
		element_rank(b, elt);
		j = b.as_int();
		cout << " : " << j << endl;
		if (i != j) {
			cout << "error in sims::test_element_rank_unrank" << endl;
			exit(1);
		}
	}
	FREE_int(elt);
}

void sims::coset_rep(
		int *Elt, int i, int j,
		int verbose_level)
// computes a coset representative in
// transversal i which maps
// the i-th base point to the point
// which is in coset j of the i-th basic orbit.
// j is a coset, not a point
// result is in cosetrep
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int bi0, bij;
	
	if (f_v) {
		cout << "sims::coset_rep i=" << i << " j=" << j
				<< " verbose_level=" << verbose_level << endl;
	}

	bi0 = get_orbit(i, 0);
	bij = get_orbit(i, j);
	if (f_v) {
		cout << "sims::coset_rep bi0=" << bi0 << " bij=" << bij << endl;
	}

	int depth;
	int *Path;
	int *Label;
	int *gen;
	int h, a;

	if (f_v) {
		cout << "sims::coset_rep "
				"before compute_coset_rep_path" << endl;
	}
	compute_coset_rep_path(
			i, j, depth, Path, Label,
			0 /*verbose_level - 2*/);

	// Path[depth + 1]
	// Label[depth]

	if (f_v) {
		cout << "sims::coset_rep "
				"after compute_coset_rep_path" << endl;
	}
	if (f_vv) {
		cout << "sims::coset_rep depth=" << depth << endl;
		cout << "sims::coset_rep Path=";
		Int_vec_print(cout, Path, depth + 1);
		cout << endl;
		cout << "sims::coset_rep Label=";
		Int_vec_print(cout, Label, depth);
		cout << endl;
	}

	A->Group_element->element_one(cosetrep, 0);

	for (h = 0; h < depth; h++) {

		if (f_v) {
			cout << "sims::coset_rep " << h << " / " << depth
					<< " Label[" << h << "]=" << Label[h] << endl;
		}

		gen = gens.ith(Label[h]);
		if (f_vv) {
			cout << "sims::coset_rep gen=:" << endl;
			A->Group_element->element_print_quick(gen, cout);
		}
		A->Group_element->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->Group_element->element_move(cosetrep_tmp, cosetrep, 0);
		a = A->Group_element->element_image_of(
				orbit[i][0], cosetrep,
				0 /* verbose_level */);
		if (f_vv) {
			cout << "sims::coset_rep cosetrep * gen = " << endl;
			A->Group_element->element_print_quick(cosetrep, cout);
		}
		if (f_v) {
			cout << "sims::coset_rep " << bi0 << " -> " << a << endl;
		}
	}
	
	
	FREE_int(Path);
	FREE_int(Label);

	if (f_v) {
		cout << "sims::coset_rep i=" << i << " j=" << j << endl;
	}
	a = A->Group_element->element_image_of(bi0, cosetrep, 0 /* verbose_level */);
	if (f_v) {
		cout << "sims::coset_rep " << bi0 << " -> " << a << endl;
	}
	if (a != bij) {
		cout << "sims::coset_rep a != bij" << endl;
		exit(1);
	}
	A->Group_element->element_move(cosetrep, Elt, 0);
	if (f_vv) {
		cout << "sims::coset_rep cosetrep:" << endl;
		A->Group_element->element_print_quick(Elt, cout);
	}
	if (f_v) {
		cout << "sims::coset_rep done" << endl;
	}
}

int sims::compute_coset_rep_depth(
		int i, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 1);
	int p, depth, jj;
	
	if (f_v) {
		cout << "sims::compute_coset_rep_depth "
				"i=" << i << " j=" << j << endl;
	}
	if (j >= orbit_len[i]) {
		cout << "sims::compute_coset_rep_depth "
				"fatal: j >= orbit_len[i]" << endl;
		cout << "sims::compute_coset_rep_depth "
				"i=" << i << " j=" << j << endl;
		cout << "orbit_len[i]=" << orbit_len[i] << endl;
		exit(1);
	}
	depth = 0;
	jj = j;
	while (true) {
		p = prev[i][jj];
		if (p == -1) {
			break;
		}
		jj = orbit_inv[i][p];
		depth++;
	}
	if (f_v) {
		cout << "sims::compute_coset_rep_depth "
				"i=" << i << " j=" << j
				<< " depth = " << depth << endl;
	}
	return depth;
}

void sims::compute_coset_rep_path(
		int i, int j, int &depth,
		int *&Path, int *&Label,
		int verbose_level)
// Path[depth + 1]
// Label[depth]
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int p, d, jj;
	
	if (f_v) {
		cout << "sims::compute_coset_rep_path "
				"i=" << i << " j=" << j << endl;
	}


	if (f_v) {
		cout << "sims::compute_coset_rep_path "
				"before compute_coset_rep_depth" << endl;
	}
	depth = compute_coset_rep_depth(i, j, verbose_level - 2);
	if (f_v) {
		cout << "sims::compute_coset_rep_path "
				"after compute_coset_rep_depth" << endl;
	}

	if (f_v) {
		cout << "sims::compute_coset_rep_path "
				"depth = " << depth << endl;
	}
	

	Path = NEW_int(depth + 1);
	Label = NEW_int(depth);

	jj = j;
	d = 0;
	while (true) {

		if (false) {
			cout << "Path[" << depth - d
					<< "]=" << jj << endl;
		}
		Path[depth - d] = jj;

		p = prev[i][jj];
		if (false) {
			cout << "p=" << p << endl;
		}

		if (p == -1) {
			break;
		}
		else {
			if (false) {
				cout << "Label[" << depth - 1 - d
						<< "]=" << label[i][jj] << endl;
			}
			Label[depth - 1 - d] = label[i][jj];
		}
		jj = orbit_inv[i][p];
		if (false) {
			cout << "jj=" << jj << endl;
		}
		d++;
	}
	if (d != depth) {
		cout << "sims::compute_coset_rep_path "
				"d != depth" << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "sims::compute_coset_rep_path path = ";
		cout << "sims::compute_coset_rep_path "
				"i=" << i << " j=" << j
				<< " depth = " << depth << endl;
		cout << "sims::compute_coset_rep_path path = ";
		Int_vec_print(cout, Path, depth + 1);
		cout << endl;
		cout << "sims::compute_coset_rep_path label = ";
		Int_vec_print(cout, Label, depth);
		cout << endl;
	}
	if (f_v) {
		cout << "sims::compute_coset_rep_path "
				"i=" << i << " j=" << j
				<< " depth = " << depth << " done" << endl;
	}
}

void sims::coset_rep_inv(
		int *Elt, int i, int j,
		int verbose_level)
// computes the inverse element of what coset_rep computes,
// i.e. an element which maps the j-th point in the orbit to the 
// i-th base point.
// j is a coset, not a point
// result is in cosetrep
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int a;
	int bi0, bij;
	
	if (f_v) {
		cout << "sims::coset_rep_inv i=" << i << " j=" << j
				<< " verbose_level=" << verbose_level << endl;
	}

	bi0 = get_orbit(i, 0);
	bij = get_orbit(i, j);
	if (f_v) {
		cout << "sims::coset_rep_inv bi0=" << bi0 << endl;
		cout << "sims::coset_rep_inv bij=" << bij << endl;
	}


	int offset = 0;
	int f_do_it_anyway_even_for_big_degree = true;
	int f_print_cycles_of_length_one = true;


	if (f_v) {
		cout << "sims::coset_rep_inv "
				"before coset_rep(i,j)" << endl;
	}
	coset_rep(Elt, i, j, 0 /*verbose_level - 2*/);
	if (f_v) {
		cout << "sims::coset_rep_inv "
				"after coset_rep(i,j)" << endl;
	}
	if (f_vv) {

		cout << "cosetrep:" << endl;
		A->Group_element->element_print_quick(Elt, cout);

		A->Group_element->element_print_for_make_element(Elt, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
			Elt, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;

	}
	a = A->Group_element->element_image_of(bi0, Elt, 0 /* verbose_level */);

	if (a != bij) {

		cout << "sims::coset_rep_inv a != get_orbit(i, 0)" << endl;
		cout << "sims::coset_rep_inv i=" << i << " j=" << j << endl;
		cout << "sims::coset_rep_inv get_orbit(i, 0)=bi0=" << bi0 << endl;
		cout << "sims::coset_rep_inv get_orbit(i, j)=bij=" << bij << endl;
		cout << "sims::coset_rep_inv a=" << a << endl;

		cout << "sims::coset_rep_inv cosetrep:" << endl;
		A->Group_element->element_print_quick(Elt, cout);

		A->Group_element->element_print_for_make_element(Elt, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
			Elt, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;


		exit(1);
	}

	A->Group_element->element_invert(Elt, cosetrep_tmp, 0 /* verbose_level */);
	A->Group_element->element_move(cosetrep_tmp, Elt, 0 /* verbose_level */);

	if (f_vv) {
		cout << "sims::coset_rep_inv cosetrep^-1=:" << endl;
		A->Group_element->element_print_quick(Elt, cout);

		A->Group_element->element_print_for_make_element(Elt, cout);
		cout << endl;

		A->Group_element->element_print_as_permutation_with_offset(
			Elt, cout,
			offset, f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0/*verbose_level*/);
		cout << endl;


	}
	a = A->Group_element->element_image_of(bij, Elt, 0 /* verbose_level */);
	if (f_v) {
		cout << "sims::coset_rep_inv cosetrep^-1 maps " << orbit[i][j]
			<< " to " << a << endl;
	}
	if (a != bi0) {
		cout << "sims::coset_rep_inv a != bi0" << endl;
		cout << "cosetrep^-1 maps " << bij
			<< " to " << a << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "sims::coset_rep_inv cosetrep:" << endl;
		A->Group_element->element_print_quick(Elt, cout);
	}
	if (f_v) {
		cout << "sims::coset_rep_inv i=" << i
			<< " j=" << j << " done" << endl;
	}
}

void sims::extract_strong_generators_in_order(
		data_structures_groups::vector_ge &SG,
		int *tl, int verbose_level)
{
	int i, nbg, nbg1, j, k = 0, gen_idx;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "sims::extract_strong_generators_in_order" << endl;
		cout << "A->base_len=" << A->base_len() << endl;
		cout << "gens.len=" << gens.len << endl;
		cout << "extract_strong_generators_in_order nb_gen=" << endl;
		Int_vec_print(cout, nb_gen, A->base_len() + 1);
		cout << endl;
		cout << "extract_strong_generators_in_order gen_perm=" << endl;
		Int_vec_print(cout, gen_perm, gens.len);
		cout << endl;
		print_generator_depth_and_perm();
		//if (f_vv) {
			//print(0);
			//}
	}
	
	SG.init(A, verbose_level - 2);
	SG.allocate(gens.len, verbose_level - 2);
	for (i = A->base_len() - 1; i >= 0; i--) {
		if (f_v) {
			cout << "sims::extract_strong_generators_in_order "
					"level i=" << i << endl;
		}
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		//cout << "i=" << i << " nbg1=" << nbg1
		// << " nbg=" << nbg << endl;
		if (f_v) {
			cout << "sims::extract_strong_generators_in_order level i=" << i
					<< " nbg1=" << nbg1 << " nbg=" << nbg << endl;
		}
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];
			//cout << "gen_idx=" << gen_idx << endl;
			if (f_v) {
				cout << "sims::extract_strong_generators_in_order "
						"the " << k << "-th strong generator "
						"is generator "
					<< j << " at position " << gen_idx << endl;
				
				cout << "moving generator " << gen_idx
				 << " to position " << k << endl;
				cout << "before:" << endl;
				A->Group_element->element_print(gens.ith(gen_idx), cout);
				cout << endl;
			}
			A->Group_element->element_move(gens.ith(gen_idx), SG.ith(k), false);
			if (f_vv) {
				cout << "sims::extract_strong_generators_in_order "
						"the " << k << "-th strong "
						"generator is generator "
					<< j << " at position " << gen_idx << endl;
				A->Group_element->element_print(SG.ith(k), cout);
				cout << endl;
			}
			k++;
		}
		tl[i] = orbit_len[i];
	}
	if (k < SG.len) {
		cout << "sims::extract_strong_generators_in_order warning" << endl;
		cout << "k = " << k << endl;
		cout << "SG.len = " << SG.len << endl;
		SG.len = k;
		exit(1);
	}
	if (f_v) {
		cout << "sims::extract_strong_generators_in_order done, "
				"found " << SG.len << " strong generators" << endl;
	}
	if (f_v) {
		cout << "sims::extract_strong_generators_in_order" << endl;
		cout << "transversal length:" << endl;
		for (i = 0; i < A->base_len(); i++) {
			cout << tl[i];
			if (i < A->base_len() - 1) {
				cout << ", ";
			}
		}
		cout << endl;
		cout << "sims::extract_strong_generators_in_order "
				"strong generators are:" << endl;
		SG.print(cout);
		cout << endl;
	}
	if (f_v) {
		cout << "sims::extract_strong_generators_in_order done" << endl;
	}
}

void sims::random_schreier_generator(
		int *Elt, int verbose_level)
// computes random Schreier generator
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);

	if (f_v) {
		cout << "sims::random_schreier_generator" << endl;
		cout << "sims::random_schreier_generator verbose_level = " << verbose_level << endl;
		cout << "sims::random_schreier_generator "
				"my_base_len=" << my_base_len << endl;
		cout << "sims::random_schreier_generator orbit_len=";
		Int_vec_print(cout, orbit_len, my_base_len);
		cout << endl;
		cout << "sims::random_schreier_generator base:" << endl;
		for (int i = 0; i < my_base_len; i++) {
			cout << i << " : " << get_orbit(i, 0) << endl;
		}
	}

	other::orbiter_kernel_system::os_interface Os;
	int i, r1, r2, pt, pt1, pt1b, pt2, pt2_coset;
	int *gen, gen_idx, nbg;

	if (nb_gen[0] == 0) {
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"nb_gen[0] == 0, choosing the identity" << endl;
		}
		A->Group_element->element_one(Elt, 0 /* verbose_level */);
		goto finish;
	}
	while (true) {
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"iteration" << endl;
			cout << "sims::random_schreier_generator orbit_len=";
			Int_vec_print(cout, orbit_len, my_base_len);
			cout << endl;
		}
		// get a random level:
		i = Os.random_integer(my_base_len);
		pt = get_orbit(i, 0);
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"i=" << i << " orbit_len[i]=" << orbit_len[i]
					<< " base_pt=" << pt
					<< " nb_gen[i]=" << nb_gen[i] << endl;
		}
	
		// get a random coset:
		r1 = Os.random_integer(orbit_len[i]);
		pt1 = get_orbit(i, r1);
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"picking coset r1=" << r1 << " / " << orbit_len[i]
					<< " with image point " << pt1 << endl;
		}
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"random level " << i << ", base pt " << pt
				<< ", random coset " << r1 << " of an orbit of length " 
				<< orbit_len[i] << ", image pt " << pt1 << endl;
		}
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"before coset_rep" << endl;
		}
	
		coset_rep(eltrk3, i, r1, verbose_level - 2);
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"after coset_rep" << endl;
			cout << "checking image of pt=" << pt << endl;
		}
		pt1b = A->Group_element->element_image_of(pt, eltrk3, 0/*verbose_level*/);

		if (f_vvv) {
			cout << "sims::random_schreier_generator "
					"coset rep maps " << pt << " to " << pt1b << endl;
		}
		if (pt1b != pt1) {
			cout << "sims::random_schreier_generator "
					"fatal: not the same point" << endl;
			cout << "action " << A->label << endl;
			cout << "level i=" << i << endl;
			cout << "coset r1=" << r1 << endl;
			cout << "base pt=" << pt << endl;
			cout << "image pt1=" << pt1 << endl;
			cout << "image  under cosetrep pt1b=" << pt1b << endl;
			cout << "basic orbit " << i << ":" << endl;
			print_basic_orbit(i);
			pt1b = A->Group_element->element_image_of(pt, eltrk3, false);
			cout << "cosetrep:" << endl;
			A->Group_element->element_print(eltrk3, cout);
			cout << endl;
			coset_rep(eltrk3, i, r1, 10 /* verbose_level */);
			exit(1);
		}
		
		// get a random generator:
		nbg = nb_gen[i];
		if (nbg == 0) {
			continue;
		}
		r2 = Os.random_integer(nbg);
		if (f_vv) {
			cout << "sims::random_schreier_generator picking "
					"generator " << r2 << " / " << nbg << endl;
		}
		gen_idx = gen_perm[r2];
		if (f_vv) {
			cout << "sims::random_schreier_generator picking "
					"generator " << r2 << " / " << nbg
					<< " gen_idx=" << gen_idx << endl;
		}
		gen = gens.ith(gen_idx);
		if (f_vvv) {
			cout << "sims::random_schreier_generator "
					"random level " << i << ", random coset " << r1
				<< ", random generator " << r2
				<< " of " << nb_gen[i] << endl;
			cout << "sims::random_schreier_generator "
					"gen = " << endl;
			A->Group_element->element_print(gen, cout);
		}
		break;
	}
	
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "cosetrep:" << endl;
		//A->element_print(eltrk3, cout);
		cout << "maps " << pt << " to " << pt1
				<< " : checking: " << pt << " -> ";
		pt1b = A->Group_element->element_image_of(pt, eltrk3, false);
		cout << pt1b;
		cout << endl;
	}
	
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "gen=" << endl;
		A->Group_element->element_print(gen, cout);
	}
	A->Group_element->element_mult(eltrk3, gen, schreier_gen1, 0);
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "sims::random_schreier_generator "
				"cosetrep * gen=" << endl;
		//A->element_print(schreier_gen1, cout);
	}
	pt2 = A->Group_element->element_image_of(pt, schreier_gen1, 0);


	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"cosetrep * gen maps " << pt
				<< " to " << pt2 << endl;
	}


	//cout << "maps " << pt << " to " << pt2 << endl;
	pt2_coset = orbit_inv[i][pt2];
	
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"pt2_coset = " << pt2_coset << endl;
	}

	if (pt2_coset >= orbit_len[i]) {


		A->Group_element->element_move(schreier_gen1, Elt, 0);
		cout << "sims::random_schreier_generator "
				"schreier generator is " << endl;
		A->Group_element->element_print(Elt, cout);
		cout << endl;
		
		if (f_v) {
			cout << "sims::random_schreier_generator "
					"done early" << endl;
		}
		return;

#if 0
		cout << "sims::random_schreier_generator "
				"fatal: additional pt " << pt2 << " reached from "
				<< pt << " under generator " << i << endl;
		print(true);
		cout << "level = " << i << endl;
		cout << "coset1 = " << r1 << endl;
		cout << "generator = " << r2 << endl;
		cout << "pt2 = " << pt2 << endl;
		cout << "coset2 = " << pt2_coset << endl;
		cout << "orbit_len = " << orbit_len[i] << endl;
		cout << "cosetrep: " << endl;
		A->element_print(cosetrep, cout);
		cout << endl;
		cout << "gen: " << endl;
		A->element_print(gen, cout);
		cout << endl;
		exit(1);
#endif

	}
	
	coset_rep_inv(eltrk3, i, pt2_coset, 0 /*verbose_level - 2*/);

	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"cosetrep(" << pt2 << ")^-1=" << endl;
		A->Group_element->element_print(eltrk3, cout);
	}
	int pt2b;

	pt2b = A->Group_element->element_image_of(pt2, eltrk3, false);
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"cosetrep(" << pt2 << ")^-1 maps "
				<< pt2 << " to " << pt2b << endl;
	}
	if (pt2b != pt) {
		cout << "sims::random_schreier_generator "
				"pt2b != pt" << endl;
		cout << "cosetrep(" << pt2 << ")^-1 maps "
				<< pt2 << " to " << pt2b << endl;
		cout << "pt=" << pt << endl;
		exit(1);
	}
	
	A->Group_element->element_mult(
			schreier_gen1, eltrk3, schreier_gen, 0);
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "cosetrep * gen * cosetrep("
				<< pt2 << ")^-1=" << endl;
		A->Group_element->element_print(schreier_gen, cout);
	}

	if (A->Group_element->element_image_of(pt, schreier_gen, 0) != pt) {
		int im;
		
		cout << "sims::random_schreier_generator "
				"fatal: schreier generator does not stabilize pt" << endl;
		cout << "pt=" << pt << endl;
		cout << "sims::random_schreier_generator action=" << endl;
		A->print_info();
		cout << "schreier generator:" << endl;
		A->Group_element->element_print(schreier_gen, cout);
		im = A->Group_element->element_image_of(pt, schreier_gen, true);
		cout << "pt = " << pt << endl;
		cout << "im = " << im << endl;
		exit(1);
	}
	A->Group_element->element_move(schreier_gen, Elt, 0);

finish:
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"random Schreier generator:" << endl;
		A->Group_element->element_print(Elt, cout);
	}
	if (f_v) {
		cout << "sims::random_schreier_generator done" << endl;
	}
}

void sims::element_as_permutation(
		actions::action *A_special,
		long int elt_rk, int *perm,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::element_as_permutation" << endl;
	}
	int *Elt;
	
	Elt = NEW_int(A->elt_size_in_int);
	
	element_unrank_lint(elt_rk, Elt);

	A_special->Group_element->compute_permutation(Elt, perm, 0);
	

	FREE_int(Elt);
	if (f_v) {
		cout << "sims::element_as_permutation done" << endl;
	}
}


int sims::least_moved_point_at_level(
		int lvl, int verbose_level)
// returns -1 if there are no generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h, nbg, gen_idx, least_moved_point = -1;
	
	if (f_v) {
		cout << "sims::least_moved_point_at_level "
				"lvl=" << lvl << endl;
		//cout << "sims::least_moved_point_at_level "
		//		"this=" << this << endl;
	}
	nbg = nb_gen[lvl];
	if (false) {
		cout << "sims::least_moved_point_at_level "
				"nbg=" << nbg << endl;
	}
	for (h = 0; h < nbg; h++) {
		gen_idx = gen_perm[h];
		if (f_vv) {
			cout << "sims::least_moved_point_at_level "
					"h=" << h << endl;
			cout << "gen_idx=" << gen_idx << endl;
			A->Group_element->element_print_quick(gens.ith(gen_idx), cout);
		}
		for (i = 0; i < A->degree; i++) {
			j = A->Group_element->element_image_of(i, gens.ith(gen_idx), 0);
			if (j != i) {
				break;
			}
		}
		if (i < A->degree) {
			if (least_moved_point == -1 || i < least_moved_point) {
				least_moved_point = i;
			}
		}
	}
	if (f_v) {
		cout << "sims::least_moved_point_at_level lvl = "
				<< lvl << ", least moved point = "
				<< least_moved_point << endl;
	}
	return least_moved_point;
}

int sims::get_orbit(
		int i, int j)
{
	if (orbit == NULL) {
		cout << "sims::get_orbit orbit == NULL "
				"i=" << i << " j=" << j << endl;
		exit(1);
	}
	if (i < 0 || i >= my_base_len) {
		cout << "sims::get_orbit out of range, "
				"i = " << i << endl;
		cout << "my_base_len=" << my_base_len << endl;
		exit(1);
	}
	if (orbit[i] == NULL) {
		cout << "sims::get_orbit orbit[i] == NULL "
				"i=" << i << " j=" << j << endl;
		exit(1);
	}
#if 1
	if (j >= transversal_length) {
		cout << "sims::get_orbit  j >= transversal_length" << endl;
		cout << "j=" << j << endl;
		cout << "transversal_length=" << transversal_length << endl;
		exit(1);
	}
#endif
	return orbit[i][j];
}

int sims::get_orbit_inv(
		int i, int j)
{
	if (orbit_inv == NULL) {
		cout << "sims::get_orbit_inv orbit_inv == NULL "
				"i=" << i << " j=" << j << endl;
		exit(1);
	}
	if (i < 0 || i >= my_base_len) {
		cout << "sims::get_orbit "
				"out of range, i = " << i << endl;
		cout << "my_base_len=" << my_base_len << endl;
		exit(1);
	}
	if (orbit_inv[i] == NULL) {
		cout << "sims::get_orbit_inv "
				"orbit_inv[i] == NULL "
				"i=" << i << " j=" << j << endl;
		exit(1);
	}
#if 1
	if (j >= transversal_length) {
		cout << "sims::get_orbit_inv "
				"j >= transversal_length" << endl;
		cout << "my_base_len=" << my_base_len << endl;
		cout << "i=" << i << endl;
		cout << "j=" << j << endl;
		cout << "transversal_length=" << transversal_length << endl;
		exit(1);
	}
#endif
	return orbit_inv[i][j];
}

int sims::get_orbit_length(
		int i)
{
	if (i < 0 || i >= my_base_len) {
		cout << "sims::get_orbit_length out of range "
				"i = " << i << endl;
		cout << "my_base_len=" << my_base_len << endl;
		exit(1);
	}
	return orbit_len[i];
}

void sims::get_orbit(
		int orbit_idx, std::vector<int> &Orb,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::get_orbit" << endl;
	}
	int len, i;

	len = orbit_len[orbit_idx];
	for (i = 0; i < len; i++) {
		Orb.push_back(orbit[orbit_idx][i]);
	}
	if (f_v) {
		cout << "sims::get_orbit done" << endl;
	}
}

void sims::all_elements(
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::all_elements" << endl;
	}




	algebra::ring_theory::longinteger_object go;
	long int i, goi;

	group_order(go);
	goi = go.as_int();

	vec = NEW_OBJECT(data_structures_groups::vector_ge);
	vec->init(A, 0 /*verbose_level*/);
	vec->allocate(goi, verbose_level);


	for (i = 0; i < goi; i++) {
		element_unrank_lint(i, vec->ith(i));
	}

	if (f_v) {
		cout << "sims::all_elements done" << endl;
	}
}

void sims::select_elements(
		long int *Index_of_elements, int nb_elements,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{

	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::select_elements" << endl;
	}

	algebra::ring_theory::longinteger_object go;
	long int i, goi, a;

	group_order(go);
	goi = go.as_int();

	vec = NEW_OBJECT(data_structures_groups::vector_ge);
	vec->init(A, 0 /*verbose_level*/);
	vec->allocate(nb_elements, verbose_level);


	for (i = 0; i < nb_elements; i++) {
		a = Index_of_elements[i];
		if (a < 0 || a >= goi) {
			cout << "sims::select_elements "
					"element index is out of range" << endl;
			exit(1);
		}
		element_unrank_lint(a, vec->ith(i));
	}

	if (f_v) {
		cout << "sims::select_elements done" << endl;
	}
}



void sims::all_elements_save_csv(
		std::string &fname, int verbose_level)
{

	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "sims::all_elements_save_csv" << endl;
	}

	data_structures_groups::vector_ge *vec;

	all_elements(vec, verbose_level);


	vec->save_csv(fname, verbose_level);
	if (f_v) {
		cout << "sims::all_elements_save_csv "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	FREE_OBJECT(vec);

	if (f_v) {
		cout << "sims::all_elements_save_csv done" << endl;
	}
}



void sims::all_elements_export_inversion_graphs(
		std::string &fname, int verbose_level)
{

	int f_v = (verbose_level >= 1);
	other::orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "sims::all_elements_export_inversion_graphs" << endl;
	}

	data_structures_groups::vector_ge *vec;

	all_elements(vec, verbose_level);


	vec->export_inversion_graphs(fname, verbose_level);
	if (f_v) {
		cout << "sims::all_elements_export_inversion_graphs "
				"Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	FREE_OBJECT(vec);

	if (f_v) {
		cout << "sims::all_elements_export_inversion_graphs done" << endl;
	}
}

void sims::get_non_trivial_base_orbits(
		std::vector<int> &base_orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::get_non_trivial_base_orbits" << endl;
	}

	int i;

	for (i = 0; i < A->base_len(); i++) {
		if (orbit_len[i] > 1) {
			base_orbit_idx.push_back(i);
		}
	}
	if (f_v) {
		cout << "sims::get_non_trivial_base_orbits done" << endl;
	}
}

void sims::get_all_base_orbits(
		std::vector<int> &base_orbit_idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::get_all_base_orbits" << endl;
	}

	int i;

	for (i = 0; i < A->base_len(); i++) {
		base_orbit_idx.push_back(i);
	}
	if (f_v) {
		cout << "sims::get_all_base_orbits done" << endl;
	}
}

int sims::element_from_path_and_test_permutation_property(
		int *elt, int &fail_depth, int verbose_level)
// given coset representatives in path[], the corresponding
// element is multiplied.
// uses eltrk1, eltrk2, eltrk3
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j;

	if (f_v) {
		cout << "sims::element_from_path_and_test_permutation_property" << endl;
	}

	if (!A->is_matrix_group()) {
		cout << "sims::element_from_path_and_test_permutation_property "
				"not a matrix group" << endl;
		exit(1);
	}

	if (f_vv) {
		cout << "path=";
		Int_vec_print(cout, path, A->base_len());
		cout << endl;
		cout << "A->degree=" << A->degree << endl;
	}

	fail_depth = A->base_len();
	A->Group_element->element_one(eltrk1, false);
	for (i = 0; i < A->base_len(); i++) {
		j = path[i];
		if (f_v) {
			cout << "sims::element_from_path_and_test_permutation_property level "
					<< i << " coset " << j << " before coset_rep" << endl;
		}
		coset_rep(eltrk3, i, j, verbose_level);


		if (f_v) {
			cout << "sims::element_from_path_and_test_permutation_property level "
					<< i << " coset " << j << " after coset_rep" << endl;
		}

		if (f_vv) {
			cout << "sims::element_from_path_and_test_permutation_property level "
					<< i << " coset " << j << ":" << endl;
			cout << "cosetrep:" << endl;
			A->Group_element->element_print_quick(eltrk3, cout);
			cout << endl;
		}

		//A->element_print_as_permutation(cosetrep, cout);
		//cout << endl;

		// pre multiply the coset representative:
		A->Group_element->element_mult(eltrk3, eltrk1, eltrk2, 0);

		// test unit vector property in row i:
		algebra::basic_algebra::matrix_group *Matrix_group;

		Matrix_group = A->get_matrix_group();
		int d;
		int idx_nonzero;

		d = Matrix_group->n;
		if (!Int_vec_of_Hamming_weight_one(eltrk2 + i * d, idx_nonzero, d)) {
			fail_depth = i;
			return false;
		}



		A->Group_element->element_move(eltrk2, eltrk1, 0);
	}
	A->Group_element->element_move(eltrk1, elt, 0);
	if (f_v) {
		cout << "sims::element_from_path_and_test_permutation_property done" << endl;
	}
	return true;
}

void sims::permutation_subgroup(
		std::vector<long int> &Gens,
		int verbose_level)
{
	algebra::ring_theory::longinteger_object go;
	long int i, j, goi, cnt;
	int *Elt;
	int fail_depth;


	Elt = NEW_int(A->elt_size_in_int);

	group_order(go);
	goi = go.as_lint();

	cnt = 0;
	for (i = 0; i < goi; i++) {

		path_unrank_lint(i);

		if ((i % 10000) == 0) {
			cout << "sims::permutation_subgroup i=" << i << " / " << goi << endl;
		}

		if (element_from_path_and_test_permutation_property(
				Elt, fail_depth, verbose_level - 2)) {

			cout << "sims::permutation_subgroup " << i << " : "
					<< " has the permutation property, cnt = " << cnt << endl;
			A->Group_element->element_print(Elt, cout);

			Gens.push_back(i);

			cnt++;
		}

#if 1
		else {

			if (advance_at_given_level(
					i, j, fail_depth)) {
				i = j - 1;
			}
			else {
				i = j;
			}

		}
#endif
	}
	cout << "sims::permutation_subgroup "
			"the order of the permutation subgroup is " << cnt << endl;
	cout << "Number of generators = " << Gens.size() << endl;
	cout << "Generators = ";
	Lint_vec_stl_print(cout, Gens);
	cout << endl;
	FREE_int(Elt);
}





}}}



