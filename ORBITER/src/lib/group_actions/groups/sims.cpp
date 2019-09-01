// sims.cpp
//
// Anton Betten
// December 21, 2003

#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

sims::sims()
{
	A = NULL;
	my_base_len = 0;
	nb_images = 0;
	images = NULL;
	gen_depth = NULL;
	gen_perm = NULL;
	nb_gen = NULL;
	transversal_length = 0;
	path = NULL;
	images = NULL;
	orbit_len = NULL;
	orbit = NULL;
	orbit_inv = NULL;
	prev = NULL;
	label = NULL;
	//Path = NULL;
	//Label = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	Elt4 = NULL;
	strip1 = NULL;
	strip2 = NULL;
	eltrk1 = NULL;
	eltrk2 = NULL;
	eltrk3 = NULL;
	cosetrep = NULL;
	cosetrep_tmp = NULL;
	schreier_gen = NULL;
	schreier_gen1 = NULL;
	//null();

}

void sims::null()
{
}

sims::sims(action *A, int verbose_level)
{
	init(A, verbose_level);
}

sims::~sims()
{
	freeself();
}

void sims::freeself()
{
	int i;
	int f_v = FALSE;
	
	if (f_v) {
		cout << "sims::freeself freeing gen_depth" << endl;
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
		cout << "sims::freeself freeing orbit, "
				"my_base_len=" << my_base_len << endl;
		}
	if (orbit) {
		for (i = 0; i < my_base_len; i++) {
			if (f_v) {
				cout << "sims::freeself freeing orbit i=" << i << endl;
				}
			if (f_v) {
				cout << "sims::freeself freeing orbit[i]" << endl;
				}
			FREE_int(orbit[i]);
			if (f_v) {
				cout << "sims::freeself freeing orbit_inv[i]" << endl;
				}
			FREE_int(orbit_inv[i]);
			if (f_v) {
				cout << "sims::freeself freeing prev[i]" << endl;
				}
			FREE_int(prev[i]);
			if (f_v) {
				cout << "sims::freeself freeing label[i]" << endl;
				}
			FREE_int(label[i]);
			}
		if (f_v) {
			cout << "sims::freeself freeing orbit"<< endl;
			}
		FREE_pint(orbit);
		FREE_pint(orbit_inv);
		FREE_pint(prev);
		FREE_pint(label);
		}
#if 0
	if (Path) {
		FREE_int(Path);
	}
	if (Label) {
		FREE_int(Label);
	}
#endif
	if (f_v) {
		cout << "sims::freeself freeing orbit_len" << endl;
		}
	if (orbit_len)
		FREE_int(orbit_len);
	if (Elt1)
		FREE_int(Elt1);
	if (Elt2)
		FREE_int(Elt2);
	if (Elt3)
		FREE_int(Elt3);
	if (Elt4)
		FREE_int(Elt4);
	if (strip1)
		FREE_int(strip1);
	if (strip2)
		FREE_int(strip2);
	if (eltrk1)
		FREE_int(eltrk1);
	if (eltrk2)
		FREE_int(eltrk2);
	if (eltrk3)
		FREE_int(eltrk3);
	if (cosetrep)
		FREE_int(cosetrep);
	if (cosetrep_tmp)
		FREE_int(cosetrep_tmp);
	if (schreier_gen)
		FREE_int(schreier_gen);
	if (schreier_gen1)
		FREE_int(schreier_gen1);
	A = NULL;
	if (f_v) {
		cout << "sims::freeself before delete_images" << endl;
		}
	delete_images();
	if (f_v) {
		cout << "sims::freeself before after_images" << endl;
		}
	null();
	if (f_v) {
		cout << "sims::freeself done" << endl;
		}
};

void sims::delete_images()
{
	int i;
	
	if (images) {
		for (i = 0; i < nb_images; i++) {
			FREE_int(images[i]);
			}
		nb_images = 0;
		FREE_pint(images);
		images = NULL;
		}
}

void sims::init_images(int nb_images)
{
#if 0
	int i, j; //, a;
	
	cout << "sims::init_images" << endl;
	if (A == NULL) {
		cout << "sims::init_images() action is NULL" << endl;
		exit(1);
		}
	delete_images();
	sims::nb_images = nb_images;
	images = NEW_pint(nb_images);
	for (i = 0; i < nb_images; i++) {
		images[i] = NEW_int(A->degree);
		for (j = 0; j < A->degree; j++) {
			images[i][j] = -1;
			//a = A->image_of(gens.ith(i), j);
			//images[i][j] = a;
			//images[i][A->degree + a] = j;
			}
		}
#else
	cout << "sims::init_images doing nothing" << endl;
#endif
}

void sims::images_append()
{
#if 0
	int **new_images = NEW_pint(nb_images + 1);
	int i, j; //, a;
	
	new_images[nb_images] = NEW_int(A->degree);
	for (j = 0; j < A->degree; j++) {
		new_images[nb_images][j] = -1;
		}
	//for (j = 0; j < 2 * A->degree; j++) {
		//new_images[i][j] = -1;
		//a = A->image_of(gens.ith(nb_images), j);
		//new_images[nb_images][j] = a;
		//new_images[nb_images][A->degree + a] = j;
		//}
	for (i = 0; i < nb_images; i++) {
		new_images[i] = images[i];
		}
	if (images)
		FREE_pint(images);
	images = new_images;
	nb_images++;
#else
	cout << "sims::images_append doing nothing" << endl;

#endif
}

void sims::init(action *A, int verbose_level)
// initializes the trivial group with the base as given in A
{
	int i;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::init action=" << A->label << endl;
	}
	
	if (f_v) {
		cout << "sims::init before init_without_base" << endl;
	}
	init_without_base(A, verbose_level);
	if (f_v) {
		cout << "sims::init after init_without_base" << endl;
	}
	
#if 0
	if (A->Stabilizer_chain) {
		my_base_len = A->base_len();
	}
	else {
		cout << "sims::init A->Stabilizer_chain == NULL, setting my_base_len to degree" << endl;
		my_base_len = A->degree;
	}
#endif
	
	if (f_v) {
		cout << "sims::init my_base_len=" << my_base_len << endl;
	}
	if (f_v) {
		cout << "sims::init allocating orbit" << endl;
	}
	orbit = NEW_pint(my_base_len);
	if (f_v) {
		cout << "sims::init allocating orbit_inv" << endl;
	}
	orbit_inv = NEW_pint(my_base_len);
	if (f_v) {
		cout << "sims::init allocating prev" << endl;
	}
	prev = NEW_pint(my_base_len);
	if (f_v) {
		cout << "sims::init allocating label" << endl;
	}
	label = NEW_pint(my_base_len);

	if (A->degree == 4294967320) {
		transversal_length = 1000;
	}
	else {
		transversal_length = A->degree;
	}
	if (f_v) {
		cout << "sims::init transversal_length=" << transversal_length << endl;
	}
	for (i = 0; i < my_base_len; i++) {
		if (f_v) {
			cout << "sims::init allocating orbit " << i << " / " << my_base_len << endl;
		}
		orbit[i] = NEW_int(transversal_length);
		orbit_inv[i] = NEW_int(transversal_length);
		prev[i] = NEW_int(transversal_length);
		label[i] = NEW_int(transversal_length);
		}
	//Path = NEW_int(A->degree + 1);
	//Label = NEW_int(A->degree + 1);
	
	FREE_int(nb_gen);
	
	
	nb_gen = NEW_int(my_base_len + 1);
	for (i = 0; i <= my_base_len; i++) {
		nb_gen[i] = 0;
		}
	
	path = NEW_int(my_base_len);

	orbit_len = NEW_int(my_base_len);
	
	for (i = 0; i < my_base_len; i++) {
		if (f_v) {
			cout << "sims::init before initialize_table " << i << " / " << my_base_len << endl;
		}
		initialize_table(i, verbose_level);
		orbit_len[i] = 1;
		}
	if (f_v) {
		cout << "sims::init done" << endl;
	}
	
}

void sims::init_without_base(action *A, int verbose_level)
{	
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::init_without_base action=" << A->label << endl;
	}
	sims::A = A;
	nb_images = 0;
	images = NULL;
	
	my_base_len = A->base_len();
	if (f_v) {
		cout << "sims::init_without_base my_base_len=" << my_base_len << endl;
	}
	
	gens.init(A, verbose_level - 2);
	gens_inv.init(A, verbose_level - 2);
	
#if 0
	Path = NEW_int(my_base_len + 1);
	Label = NEW_int(my_base_len + 1);
#endif

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

void sims::reallocate_base(int old_base_len, int verbose_level)
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
#if 0
	int *old_Path = Path;
	int *old_Label = Label;
#endif
	
	if (f_v) {
		cout << "sims::reallocate_base from " 
			<< old_base_len << " to " << A->base_len() << endl;
		}

	my_base_len = A->base_len();
	if (f_v) {
		cout << "sims::reallocate_base my_base_len=" << my_base_len << endl;
	}
	
	nb_gen = NEW_int(my_base_len + 1);
	path = NEW_int(my_base_len);
	orbit_len = NEW_int(my_base_len);
	orbit = NEW_pint(my_base_len);
	orbit_inv = NEW_pint(my_base_len);
	prev = NEW_pint(my_base_len);
	label = NEW_pint(my_base_len);
#if 0
	Path = NEW_int(my_base_len + 1);
	Label = NEW_int(my_base_len + 1);
#endif
	for (i = 0; i < old_base_len; i++) {
		nb_gen[i] = old_nb_gen[i];
		path[i] = old_path[i];
		orbit_len[i] = old_orbit_len[i];
		orbit[i] = old_orbit[i];
		orbit_inv[i] = old_orbit_inv[i];
		prev[i] = old_prev[i];
		label[i] = old_label[i];
		}
	for (i = old_base_len; i < my_base_len; i++) {
		nb_gen[i] = 0;
		path[i] = 0;
		orbit[i] = NEW_int(A->degree);
		orbit_inv[i] = NEW_int(A->degree);
		prev[i] = NEW_int(A->degree);
		label[i] = NEW_int(A->degree);
		initialize_table(i, 0 /* verbose_level */);
		init_trivial_orbit(i, 0 /* verbose_level */);
		}
	nb_gen[my_base_len] = 0;
#if 0
	nb_gen[A->base_len - 1] = old_nb_gen[A->base_len - 1];
	nb_gen[A->base_len] = 0;
	path[A->base_len - 1] = 0;
	orbit[A->base_len - 1] = NEW_int(A->degree);
	orbit_inv[A->base_len - 1] = NEW_int(A->degree);
	prev[A->base_len - 1] = NEW_int(A->degree);
	label[A->base_len - 1] = NEW_int(A->degree);
	initialize_table(A->base_len - 1);
	init_trivial_orbit(A->base_len - 1);
#endif
	if (old_nb_gen)
		FREE_int(old_nb_gen);
	if (old_path)
		FREE_int(old_path);
	if (old_orbit_len)
		FREE_int(old_orbit_len);
	if (old_orbit)
		FREE_pint(old_orbit);
	if (old_orbit_inv)
		FREE_pint(old_orbit_inv);
	if (old_prev)
		FREE_pint(old_prev);
	if (old_label)
		FREE_pint(old_label);
#if 0
	if (old_Path)
		FREE_int(old_Path);
	if (old_Label)
		FREE_int(old_Label);
#endif
}

void sims::initialize_table(int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	combinatorics_domain Combi;
	int j;
	
	if (f_v) {
		cout << "sims::initialize_table" << endl;
	}
	Combi.perm_identity(orbit[i], transversal_length);
	Combi.perm_identity(orbit_inv[i], transversal_length);
	for (j = 0; j < transversal_length; j++) {
		prev[i][j] = -1;
		label[i][j] = -1;
		}
	orbit_len[i] = 0;
	if (f_v) {
		cout << "sims::initialize_table done" << endl;
	}
}

void sims::init_trivial_group(int verbose_level)
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
	if (A->Stabilizer_chain == NULL) {
		cout << "sims::init_trivial_group A->Stabilizer_chain == NULL" << endl;
		return;
	}
	if (my_base_len != A->base_len()) {
		cout << "sims::init_trivial_group: "
				"my_base_len != A->base_len" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "before init_generators" << endl;
		}
	init_generators(0, NULL, verbose_level - 3);
	for (i = 0; i < my_base_len; i++) {
		if (f_vv) {
			cout << "before init_trivial_orbit i=" << i << endl;
			}
		init_trivial_orbit(i, verbose_level);
		}
}

void sims::init_trivial_orbit(int i, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int coset_of_base_point;
	int bi;
	
	if (f_v) {
		cout << "sims::init_trivial_orbit i=" << i << endl;
	}
	if (my_base_len != A->base_len()) {
		cout << "sims::init_trivial_orbit: "
				"my_base_len != A->base_len" << endl;
		exit(1);
		}
	bi = A->base_i(i);
	if (f_v) {
		cout << "sims::init_trivial_orbit bi=" << bi << endl;
	}

	coset_of_base_point = get_orbit_inv(i, bi);

	if (coset_of_base_point) {
		swap_points(i, coset_of_base_point, 0);
		}
	//cout << "sims::init_trivial_orbit " << i
	// << " : " << A->base[i] << endl;
	//cout << "orbit[i][0] = " << orbit[i][0] << endl;
	orbit_len[i] = 1;
	if (f_v) {
		cout << "sims::init_trivial_orbit i=" << i << " done" << endl;
	}
}

void sims::init_generators(vector_ge &generators,
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
	else	{
		init_generators(generators.len, NULL, verbose_level);
		}
	if (f_v) {
		cout << "sims::init_generators done" << endl;
		}
}

void sims::init_generators(int nb, int *elt,
		int verbose_level)
// copies the given elements into the generator array, 
// then computes depth and perm
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	int i;
	
	if (f_v) {
		cout << "sims::init_generators nb = " << nb << endl;
		}
	gens.allocate(nb, verbose_level - 2);
	gens_inv.allocate(nb, verbose_level - 2);
	for (i = 0; i < nb; i++) {
		if (f_vv) {
			cout << "sims::init_generators "
					"i = " << i << " / " << nb << ":" << endl;
			}
		gens.copy_in(i, elt + i * A->elt_size_in_int);
		if (f_vvv) {
			A->element_print_quick(
					elt + i * A->elt_size_in_int, cout);
			}
		A->element_invert(elt + i * A->elt_size_in_int,
				gens_inv.ith(i), FALSE);
		}
	if (f_v) {
		cout << "sims::init_generators before init_images" << endl;
		}
	init_images(nb);
	if (f_v) {
		cout << "sims::init_generators "
				"before init_generator_depth_and_perm" << endl;
		}
	init_generator_depth_and_perm(verbose_level);
	if (f_v) {
		cout << "sims::init_generators done" << endl;
		}
}

void sims::init_generators_by_hdl(int nb_gen, int *gen_hdl, int verbose_level)
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
		A->element_retrieve(gen_hdl[i], gens.ith(i), FALSE);
		A->element_invert(gens.ith(i), gens_inv.ith(i), FALSE);
		}
	init_images(nb_gen);	
	init_generator_depth_and_perm(FALSE);
	if (f_v) {
		cout << "sims::init_generators_by_hdl done" << endl;
	}

}

void sims::init_generator_depth_and_perm(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, d;
	
	if (f_v) {
		cout << "sims::init_generator_depth_and_perm" << endl;
		cout << "gens.len=" << gens.len << endl;
		cout << "action=" << A->label << endl;
		}
	if (my_base_len != A->base_len()) {
		cout << "sims::init_generator_depth_and_perm "
				"my_base_len != A->base_len" << endl;
		exit(1);
		}

	for (i = 0; i <= A->base_len(); i++) {
		nb_gen[i] = 0;
		}
	gen_depth = NEW_int(gens.len);
	gen_perm = NEW_int(gens.len);
	for (i = 0; i < gens.len; i++) {
		gen_perm[i] = i;
		gen_depth[i] = generator_depth(i);
		if (f_vv) {
			cout << "generator " << i
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
}

void sims::add_generator(int *elt, int verbose_level)
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
				"generator no " << gens.len << endl;
		//A->element_print_quick(elt, cout);
		//cout << endl;
		cout << "sims::add_generator my_base_len=" << my_base_len << endl;
		cout << "sims::add_generator A->base_len()=" << A->base_len() << endl;
		}
	if (f_v) {
		cout << "sims::add_generator before adding the generator:" << endl;
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
	A->element_invert(elt, gens_inv.ith(idx), FALSE);
	
	if (f_v) {
		cout << "sims::add_generator "
				"before images_append" << endl;
		}
	images_append();
	if (f_v) {
		cout << "sims::add_generator "
				"after images_append" << endl;
		}

	
	if (f_v) {
		cout << "sims::add_generator "
				"before generator_depth(idx)" << endl;
	}
	depth = generator_depth(idx);
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
		cout << "sims::add_generator after adding the generator:" << endl;
		print_generator_depth_and_perm();
	}

	if (f_v) {
		cout << "sims::add_generator done" << endl;
	}
}




int sims::generator_depth(int gen_idx)
// returns the index of the first base point 
// which is moved by a given generator. 
{
	int i, bi, j;
	
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		j = get_image(bi, gen_idx);
		if (j != bi) {
			return i;
		}
	}
	return A->base_len();
}

int sims::generator_depth(int *elt)
// returns the index of the first base point 
// which is moved by the given element
{
	int i, bi, j;
	
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		j = get_image(bi, elt);
		if (j != bi)
			return i;
		}
	return A->base_len();
}

void sims::group_order(longinteger_object &go)
{
	longinteger_domain D;

	//cout << "sims::group_order before D.multiply_up" << endl;
	//cout << "A->base_len=" << A->base_len << endl;
	//cout << "orbit_len=";
	//int_vec_print(cout, orbit_len, A->base_len);
	//cout << endl;
	D.multiply_up(go, orbit_len, my_base_len /*A->base_len()*/, 0);
	//cout << "sims::group_order after D.multiply_up" << endl;
}

void sims::group_order_verbose(longinteger_object &go, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_domain D;
	
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

int sims::group_order_int()
{
	longinteger_object go;

	group_order(go);
	return go.as_int();
}

int sims::is_trivial_group()
{
	int j = last_moved_base_point();
	
	if (j == -1)
		return TRUE;
	else 
		return FALSE;
}

int sims::last_moved_base_point()
// j == -1 means the group is trivial
{
	int j;
	
	for (j = A->base_len() - 1; j >= 0; j--) {
		if (orbit_len[j] != 1)
			break;
		}
	return j;
}

int sims::get_image(int i, int gen_idx)
// get the image of a point i under
// generator gen_idx, goes through a
// table of stored images by default.
// Computes the image only if not yet available.
{
	int a;
	
	if (nb_images == 0 || images == NULL) {
		a = A->element_image_of(i, gens.ith(gen_idx), 0);
		return a;
		//cout << "sims::get_image() images == NULL" << endl;
		//exit(1);
		}
	a = images[gen_idx][i];
#if 1
	if (a == -1) {
		a = A->element_image_of(i, gens.ith(gen_idx), 0);
		images[gen_idx][i] = a;
		//images[gen_idx][A->degree + a] = i;
		}
#endif
	return a;
}

int sims::get_image(int i, int *elt)
// get the image of a point i under a given group element,
// does not goes through a table.
{
	return A->element_image_of(i, elt, FALSE);
}

void sims::swap_points(int lvl, int i, int j)
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

void sims::path_unrank_int(int a)
{
	int h, l;
	
	for (h = A->base_len() - 1; h >= 0; h--) {
		l = orbit_len[h];

		path[h] = a % l;
		a = a / l;
		}
}

int sims::path_rank_int()
{
	int h, a;
	
	a = 0;
	for (h = 0; h < A->base_len(); h++) {
		if (h) {
			a *= orbit_len[h];
			}
		a += path[h];
		}
	return a;
}

void sims::element_from_path(int *elt, int verbose_level)
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
		int_vec_print(cout, path, A->base_len());
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
	
	A->element_one(eltrk1, FALSE);
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
			A->element_print_quick(eltrk3, cout);
			cout << endl;
			}
		
		//A->element_print_as_permutation(cosetrep, cout);
		//cout << endl;
		
		// pre multiply the coset representative:
		A->element_mult(eltrk3, eltrk1, eltrk2, 0);
		A->element_move(eltrk2, eltrk1, 0);
		}
	A->element_move(eltrk1, elt, 0);
	if (f_v) {
		cout << "sims::element_from_path done" << endl;
		}
}

void sims::element_from_path_inv(int *elt)
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
	A->element_one(eltrk1, FALSE);
	for (i = 0; i < A->base_len(); i++) {
		j = path[i];
		
		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);
		
		//A->element_print_as_permutation(eltrk3, cout);
		//cout << endl;
		
		// pre multiply the coset representative:
		A->element_mult(eltrk1, eltrk3, eltrk2, FALSE);
		A->element_move(eltrk2, eltrk1, FALSE);
		}
	A->element_move(eltrk1, elt, FALSE);
}

void sims::element_unrank(longinteger_object &a,
		int *elt, int verbose_level)
// Returns group element whose rank is a. 
// the elements represented by the chain
// are enumerated 0, ... go - 1
// with the convention that 0 always stands
// for the identity element.
// The computed group element will be computed into Elt1
{
	int f_v = (verbose_level >= 1);
	int ii, l, r;
	longinteger_domain D;
	longinteger_object q;
	
	if (f_v) {
		cout << "sims::element_unrank rk=" << a << endl;
		}
	for (ii = A->base_len() - 1; ii >= 0; ii--) {
		l = orbit_len[ii];

		D.integral_division_by_int(a, l, q, r);
		q.assign_to(a);
		
		path[ii] = r;
		//cout << r << " ";
		}
	//cout << endl;
	if (f_v) {
		cout << "sims::element_unrank path=";
		int_vec_print(cout, path, A->base_len());
		cout << endl;
		}
	element_from_path(elt, 0);
	if (f_v) {
		cout << "sims::element_unrank done" << endl;
		}
}

void sims::element_unrank(longinteger_object &a, int *elt)
// Returns group element whose rank is a. 
// the elements represented by the chain
// are enumerated 0, ... go - 1
// with the convention that 0 always stands
// for the identity element.
// The computed group element will be computed into Elt1
{
	int ii, l, r;
	longinteger_domain D;
	longinteger_object q;
	
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

void sims::element_rank(longinteger_object &a, int *elt)
// Computes the rank of the element in elt into a.
// uses eltrk1, eltrk2
{
	int i, j, bi, jj, l;
	longinteger_domain D;
	longinteger_object b, c;
	
	A->element_move(elt, eltrk1, FALSE);
	a.zero();
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		l = orbit_len[i];
		
		if (i > 0) {
			b.create(l);
			D.mult(a, b, c);
			c.assign_to(a);
			}
		
		jj = A->element_image_of(bi, eltrk1, FALSE);
		//cout << "at level " << i << ", maps bi = "
		// << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {
			cout << "sims::element_rank() j >= orbit_len[i]" << endl;
			cout << "i=" << i << endl;
			cout << "jj=bi^elt=" << jj << endl;
			cout << "j=orbit_inv[i][jj]=" << j << endl;
			cout << "base=";
			int_vec_print(cout, A->get_base(), A->base_len());
			cout << endl;
			cout << "orbit_len=";
			int_vec_print(cout, orbit_len, A->base_len());
			cout << endl;
			cout << "elt=" << endl;
			A->element_print(eltrk1, cout);
			exit(1);
			}
		b.create(j);
		D.add(a, b, c);
		c.assign_to(a);
		
		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);

		A->element_mult(eltrk1, eltrk3, eltrk2, FALSE);
		A->element_move(eltrk2, eltrk1, FALSE);
		}
}

void sims::element_unrank_int(int rk, int *Elt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//longinteger_object a;

	if (f_v) {
		cout << "sims::element_unrank_int rk=" << rk << endl;
		}
	//a.create(rk);
	//element_unrank(a, Elt);
	path_unrank_int(rk);
	if (f_v) {
		cout << "sims::element_unrank path=";
		int_vec_print(cout, path, A->base_len());
		cout << endl;
		}
	element_from_path(Elt, 0);

}

void sims::element_unrank_int(int rk, int *Elt)
{
	//longinteger_object a;
	
	//a.create(rk);
	element_unrank_int(rk, Elt, 0);
}

int sims::element_rank_int(int *Elt)
{
	longinteger_object a;
	
	element_rank(a, Elt);
	return a.as_int();
}

int sims::is_element_of(int *elt)
{
	int i, j, bi, jj; //, l;
	
	A->element_move(elt, eltrk1, FALSE);
	for (i = 0; i < A->base_len(); i++) {
		bi = A->base_i(i);
		//l = orbit_len[i];
		
		
		jj = A->element_image_of(bi, eltrk1, FALSE);
		//cout << "at level " << i << ", maps bi = "
		// << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {
			return FALSE;
			}
		
		coset_rep_inv(eltrk3, i, j, 0 /* verbose_level */);

		A->element_mult(eltrk1, eltrk3, eltrk2, FALSE);
		A->element_move(eltrk2, eltrk1, FALSE);
		}
	return TRUE;
}

void sims::test_element_rank_unrank()
{
	longinteger_object go, a, b;
	int i, j, goi;
	int *elt = NEW_int(A->elt_size_in_int);
	
	group_order(go);
	goi = go.as_int();
	for (i = 0; i < goi; i++) {
		a.create(i);
		element_unrank(a, elt);
		cout << i << " : " << endl;
		A->element_print(elt, cout);
		cout << " : ";
		A->element_print_as_permutation(elt, cout);
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

void sims::coset_rep(int *Elt, int i, int j, int verbose_level)
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
		cout << "sims::coset_rep i=" << i << " j=" << j << endl;
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
	compute_coset_rep_path(i, j, depth, Path, Label, verbose_level - 2);
	if (f_v) {
		cout << "sims::coset_rep "
				"after compute_coset_rep_path" << endl;
		cout << "sims::coset_rep depth=" << depth << endl;
		cout << "sims::coset_rep Path=";
		int_vec_print(cout, Path, depth);
		cout << endl;
		cout << "sims::coset_rep Label=";
		int_vec_print(cout, Label, depth);
		cout << endl;
	}

	A->element_one(cosetrep, 0);
	for (h = 0; h < depth; h++) {
		if (f_v) {
			cout << "sims::coset_rep " << h << " / " << depth
					<< " Label[" << h << "]=" << Label[h] << endl;
		}
		gen = gens.ith(Label[h]);
		if (f_vv) {
			cout << "sims::coset_rep gen=:" << endl;
			A->element_print_quick(gen, cout);
		}
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		a = A->element_image_of(orbit[i][0], cosetrep,
				0 /* verbose_level */);
		if (f_vv) {
			cout << "sims::coset_rep cosetrep*gen=:" << endl;
			A->element_print_quick(cosetrep, cout);
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
	a = A->element_image_of(bi0, cosetrep, 0 /* verbose_level */);
	if (f_v) {
		cout << "sims::coset_rep " << bi0 << " -> " << a << endl;
	}
	if (a != bij) {
		cout << "sims::coset_rep a != bij" << endl;
		exit(1);
	}
	A->element_move(cosetrep, Elt, 0);
	if (f_vv) {
		cout << "sims::coset_rep cosetrep:" << endl;
		A->element_print_quick(Elt, cout);
	}
	if (f_v) {
		cout << "sims::coset_rep cosetrep done" << endl;
	}
}

int sims::compute_coset_rep_depth(int i, int j, int verbose_level)
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
	while (TRUE) {
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

void sims::compute_coset_rep_path(int i, int j, int &depth,
		int *&Path, int *&Label,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = FALSE; // (verbose_level >= 2);
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

	if (f_vv) {
		cout << "sims::compute_coset_rep_path "
				"depth = " << depth << endl;
		}
	

	Path = NEW_int(depth + 1);
	Label = NEW_int(depth);

	jj = j;
	d = 0;
	while (TRUE) {

		if (f_vv) {
			cout << "Path[" << depth - d
					<< "]=" << jj << endl;
			}
		Path[depth - d] = jj;

		p = prev[i][jj];
		if (f_vv) {
			cout << "p=" << p << endl;
			}

		if (p == -1) {
			break;
			}
		else {
			if (f_vv) {
				cout << "Label[" << depth - 1 - d
						<< "]=" << label[i][jj] << endl;
				}
			Label[depth - 1 - d] = label[i][jj];
			}
		jj = orbit_inv[i][p];
		if (f_vv) {
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
		int_vec_print(cout, Path, depth + 1);
		cout << endl;
		cout << "sims::compute_coset_rep_path label = ";
		int_vec_print(cout, Label, depth);
		cout << endl;
		}
	if (f_v) {
		cout << "sims::compute_coset_rep_path "
				"i=" << i << " j=" << j
				<< " depth = " << depth << " done" << endl;
		}
}

void sims::coset_rep_inv(int *Elt, int i, int j,
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
		cout << "sims::coset_rep_inv i=" << i << " j=" << j << endl;
		}

	bi0 = get_orbit(i, 0);
	bij = get_orbit(i, j);
	if (f_v) {
		cout << "sims::coset_rep_inv i=" << i << " j=" << j << " bi0=" << bi0 << " bij=" << bij << endl;
		}

	if (f_v) {
		cout << "sims::coset_rep_inv before "
				"coset_rep(i,j)" << endl;
		}
	coset_rep(Elt, i, j, verbose_level - 2);
	if (f_v) {
		cout << "sims::coset_rep_inv "
				"coset_rep(i=" << i << " j=" << j << ") done" << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(Elt, cout);
		}
	a = A->element_image_of(bi0, Elt, 0 /* verbose_level */);
	if (a != bij) {

		cout << "sims::coset_rep_inv a != get_orbit(i, 0)" << endl;
		cout << "i=" << i << " j=" << j << endl;
		cout << "get_orbit(i, 0)=" << bi0
			<< " get_orbit(i, j)=" << bij << endl;
		cout << "a=" << a << endl;

		cout << "cosetrep:" << endl;
		A->element_print_quick(Elt, cout);
		exit(1);
		}

	A->element_invert(Elt, cosetrep_tmp, 0 /* verbose_level */);
	A->element_move(cosetrep_tmp, Elt, 0 /* verbose_level */);
	if (f_vv) {
		cout << "cosetrep^-1=:" << endl;
		A->element_print_quick(Elt, cout);
		}
	a = A->element_image_of(bij, Elt, 0 /* verbose_level */);
	if (f_v) {
		cout << "cosetrep^-1 maps " << orbit[i][j]
			<< " to " << a << endl;
		}
	if (a != bi0) {
		cout << "sims::coset_rep_inv a != bi0" << endl;
		cout << "cosetrep^-1 maps " << bij
			<< " to " << a << endl;
		exit(1);
		}

	if (f_vv) {
		cout << "cosetrep:" << endl;
		A->element_print_quick(Elt, cout);
		}
	if (f_v) {
		cout << "sims::coset_rep_inv i=" << i
			<< " j=" << j << " done" << endl;
		}
}

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
		int_vec_print(cout, orbit_len, A->base_len());
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
		int_vec_print(cout, tl, A->base_len());
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
	int f_vvv = FALSE; //(verbose_level >= 3);
	int i, cur, cur_pt, total, total0;
	int next_pt, next_pt_loc, gen_idx, nbg;
	
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << endl;
		}
	cur = 0;
	total = total0 = orbit_len[lvl];
	nbg = nb_gen[lvl];
	while (cur < total) {
		cur_pt = orbit[lvl][cur];
		if (f_vvv) {
			cout << "sims::extend_base_orbit: "
					"applying generator to " << cur_pt << endl;
			}
		for (i = 0; i < nbg; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
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
						"n e w pt " << next_pt << " reached from "
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
		cout << "sims::extend_base_orbit " << lvl
				<< " finished" << endl;
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
			if (next_pt_loc < orbit_len[lvl])
				continue;
			if (FALSE) {
				cout << "n e w pt " << next_pt << " reached from "
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
			} else {
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
			if (next_pt_loc < orbit_len[lvl])
				continue;
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length "
						"n e w pt " << next_pt << " reached from "
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
			if (i < orbit_len[lvl] - 1)
				cout << ", ";
			}
		cout << " }" << endl;
		}
}

void sims::extract_strong_generators_in_order(vector_ge &SG,
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
		int_vec_print(cout, nb_gen, A->base_len() + 1);
		cout << endl;
		cout << "extract_strong_generators_in_order gen_perm=" << endl;
		int_vec_print(cout, gen_perm, gens.len);
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
			cout << "sims::extract_strong_generators_in_order level i=" << i << endl;
		}
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		//cout << "i=" << i << " nbg1=" << nbg1
		// << " nbg=" << nbg << endl;
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];
			//cout << "gen_idx=" << gen_idx << endl;
			if (f_vv) {
				cout << "sims::extract_strong_generators_in_order the " << k << "-th strong generator "
						"is generator "
					<< j << " at position " << gen_idx << endl;
				
				cout << "moving generator " << gen_idx
				 << " to position " << k << endl;
				cout << "before:" << endl;
				A->element_print(gens.ith(gen_idx), cout);
				cout << endl;
				}
			A->element_move(gens.ith(gen_idx), SG.ith(k), FALSE);
			if (f_vv) {
				cout << "sims::extract_strong_generators_in_order the " << k << "-th strong "
						"generator is generator "
					<< j << " at position " << gen_idx << endl;
				A->element_print(SG.ith(k), cout);
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
		cout << "sims::extract_strong_generators_in_order strong generators are:" << endl;
		SG.print(cout);
		cout << endl;
	}
	if (f_v) {
		cout << "sims::extract_strong_generators_in_order done" << endl;
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
	add_generator_at_level(residue,
			drop_out_level, verbose_level);
	//add_generator_at_level_only(residue, drop_out_level, verbose_level);
	// !!! this was add_generator_at_level previously
	
	if (FALSE) {
		cout << "sims::strip_and_add n e w set of generators:" << endl;
		gens.print(cout);
		gens.print_as_permutation(cout);
		}
	if (f_v) {
		cout << "sims::strip_and_add finished, n e w group order is ";
		print_group_order(cout);
		cout << endl;
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
		cout << "sims::add_generator_at_level adding generator at level " << lvl << endl;
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
		cout << "adding generator at level " << lvl << endl;
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
}

void sims::random_schreier_generator(int *Elt, int verbose_level)
// computes random Schreier generator into schreier_gen
{
	int i, r1, r2, pt, pt1, pt1b, pt2, pt2_coset;
	int *gen, gen_idx, nbg;
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "sims:random_schreier_generator" << endl;
		cout << "sims:random_schreier_generator my_base_len=" << my_base_len << endl;
		cout << "sims:random_schreier_generator orbit_len=";
		int_vec_print(cout, orbit_len, my_base_len);
		cout << endl;
		cout << "sims:random_schreier_generator base:" << endl;
		for (i = 0; i < my_base_len; i++) {
			cout << i << " : " << get_orbit(i, 0) << endl;
		}
		}
	if (nb_gen[0] == 0) {
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"nb_gen[0] == 0, choosing the identity" << endl;
			}
		A->element_one(Elt, 0 /* verbose_level */);
		//A->element_one(schreier_gen, 0 /* verbose_level */);
		goto finish;
		}
	while (TRUE) {
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"iteration" << endl;
			cout << "sims:random_schreier_generator orbit_len=";
			int_vec_print(cout, orbit_len, my_base_len);
			cout << endl;
			}
		// get a random level:
		i = random_integer(my_base_len);
		pt = get_orbit(i, 0);
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"i=" << i << " orbit_len[i]=" << orbit_len[i] << " base_pt=" << pt << " nb_gen[i]=" << nb_gen[i] << endl;
		}
	
		// get a random coset:
		r1 = random_integer(orbit_len[i]);
		pt1 = get_orbit(i, r1);
		if (f_vv) {
			cout << "sims::random_schreier_generator "
					"picking coset r1=" << r1 << " / " << orbit_len[i] << " with image point " << pt1 << endl;
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
		// coset rep now in cosetrep
		pt1b = A->element_image_of(pt, eltrk3,
				0/*verbose_level*/);

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
			pt1b = A->element_image_of(pt, eltrk3, FALSE);
			cout << "cosetrep:" << endl;
			A->element_print(eltrk3, cout);
			cout << endl;
			coset_rep(eltrk3, i, r1, 10 /* verbose_level */);
			exit(1);
			}
		
		// get a random generator:
		nbg = nb_gen[i];
		if (nbg == 0) {
			continue;
			}
		r2 = random_integer(nbg);
		if (f_vv) {
			cout << "sims::random_schreier_generator picking generator " << r2 << " / " << nbg << endl;
		}
		gen_idx = gen_perm[r2];
		if (f_vv) {
			cout << "sims::random_schreier_generator picking generator " << r2 << " / " << nbg << " gen_idx=" << gen_idx << endl;
		}
		gen = gens.ith(gen_idx);
		if (f_vvv) {
			cout << "sims::random_schreier_generator "
					"random level " << i << ", random coset " << r1
				<< ", random generator " << r2
				<< " of " << nb_gen[i] << endl;
			cout << "sims::random_schreier_generator "
					"gen = " << endl;
			A->element_print(gen, cout);
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
		pt1b = A->element_image_of(pt, eltrk3, FALSE);
		cout << pt1b;
		cout << endl;
		}
	
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "gen=" << endl;
		A->element_print(gen, cout);
		}
	A->element_mult(eltrk3, gen, schreier_gen1, 0);
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "sims::random_schreier_generator cosetrep * gen=" << endl;
		//A->element_print(schreier_gen1, cout);
		}
	pt2 = A->element_image_of(pt, schreier_gen1, 0);


	if (f_vv) {
		cout << "sims::random_schreier_generator cosetrep * gen maps " << pt
				<< " to " << pt2 << endl;
		}


	//cout << "maps " << pt << " to " << pt2 << endl;
	pt2_coset = orbit_inv[i][pt2];
	
	if (f_vv) {
		cout << "sims::random_schreier_generator pt2_coset = " << pt2_coset << endl;
		}

	if (pt2_coset >= orbit_len[i]) {


		A->element_move(schreier_gen1, Elt, 0);
		cout << "sims::random_schreier_generator schreier generator is " << endl;
		A->element_print(Elt, cout);
		cout << endl;
		
		if (f_v) {
			cout << "sims::random_schreier_generator done early" << endl;
			}
		return;

#if 0
		cout << "sims::random_schreier_generator "
				"fatal: n e w pt " << pt2 << " reached from "
				<< pt << " under generator " << i << endl;
		print(TRUE);
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
		cout << "sims::random_schreier_generator cosetrep(" << pt2 << ")^-1=" << endl;
		A->element_print(eltrk3, cout);
		}
	int pt2b;

	pt2b = A->element_image_of(pt2, eltrk3, FALSE);
	if (f_vv) {
		cout << "sims::random_schreier_generator cosetrep(" << pt2 << ")^-1 maps "
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
	
	A->element_mult(schreier_gen1, eltrk3, schreier_gen, 0);
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"after the while loop" << endl;
		cout << "cosetrep * gen * cosetrep("
				<< pt2 << ")^-1=" << endl;
		A->element_print(schreier_gen, cout);
		//cout << endl;
		}

	if (A->element_image_of(pt, schreier_gen, 0) != pt) {
		int im;
		
		cout << "sims::random_schreier_generator() "
				"fatal: schreier generator does not stabilize pt" << endl;
		cout << "pt=" << pt << endl;
		cout << "schreier generator:" << endl;
		A->element_print(schreier_gen, cout);
		//cout << endl;
		im = A->element_image_of(pt, schreier_gen, TRUE);
		cout << "im = " << im << endl;
		exit(1);
		}
	A->element_move(schreier_gen, Elt, 0);

finish:
	if (f_vv) {
		cout << "sims::random_schreier_generator "
				"random Schreier generator:" << endl;
		A->element_print(Elt, cout);
		//cout << endl;
		}
	if (f_v) {
		cout << "sims::random_schreier_generator done" << endl;
		}
}

void sims::build_up_group_random_process_no_kernel(
		sims *old_G, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	longinteger_object go, go1;
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
		sims *extending_by_G, longinteger_object &target_go,
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
}

void sims::build_up_group_random_process(sims *K, 
	sims *old_G, 
	longinteger_object &target_go, 
	int f_override_choose_next_base_point,
	int (*choose_next_base_point_method)(action *A,
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
	longinteger_domain D;
	longinteger_object go, G_order, K_order, KG_order, quo, rem;
	int drop_out_level, image, cnt, b, c, /*f_added,*/ old_base_len;
	action *GA;
	action *KA;
	int *Elt;
	
	if (f_v) {
		cout << "sims::build_up_group_random_process" << endl;
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
		else if ((cnt % 2) == 1){
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"choosing random element in the group by "
						"which we extend" << endl;
				}
			old_G->random_element(GA->Elt1, verbose_level - 5);
			if (f_vv) {
				cout << "sims::build_up_group_random_process: "
						"random element chosen, path = ";
				int_vec_print(cout, old_G->path, old_G->A->base_len());
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
							"next suggested base point is " << b << endl;
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
					K->add_generator_at_level(GA->Elt3,
							drop_out_level, 0/*verbose_level - 3*/);
					if (f_vvv) {
						cout << "sims::build_up_group_random_process: "
								"the residue has been added as kernel "
								"generator at level " << drop_out_level
								<< endl;
						}
					//f_added = TRUE;
					}
				else {
					if (f_vvv) {
						cout << "sims::build_up_group_random_process: "
								"choosing n e w base point " << b << endl;
						}
					old_base_len = GA->base_len();
					GA->Stabilizer_chain->reallocate_base(b);
					if (f_vvv) {
						//cout << "after reallocate_base 1" << endl;
						}
					reallocate_base(old_base_len, verbose_level - 1);
					if (f_vvv) {
						//cout << "after reallocate_base 2" << endl;
						}
					if (f_vv) {
						cout << "sims::build_up_group_random_process: "
								"n e w base point " << b
							<< " chosen, n e w base has length "
							<< GA->base_len() << endl;
						cout << "sims::build_up_group_random_process: "
								"calling add_generator_at_level" << endl;
						}
					add_generator_at_level(GA->Elt2,
							GA->base_len() - 1, 0/*verbose_level - 3*/);
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
			add_generator_at_level(GA->Elt2, drop_out_level,
					0/*verbose_level - 3*/);
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
					"n e w group order is " << KG_order
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
				cout << "n e w group order is " << KG_order
					<< " = " << G_order << " * " << K_order << endl;
				}
			//break;
			exit(1);
			}
		}
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
	vector_ge *gens,
	int f_target_go, longinteger_object *target_go, 
	int f_override_choose_next_base_point,
	int (*choose_next_base_point_method)(action *A,
			int *Elt, int verbose_level),
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
	//int f_vvvv = (verbose_level >= 4);
	longinteger_domain D;
	longinteger_object G_order, K_order, KG_order;
	int drop_out_level, image, f_added, j;
	int level, base_point, b, old_base_len;
	action *GA;
	action *KA;
	vector_ge subset_of_gens;
	
	GA = A;
	KA = K->A;
	
	
	if (f_v) {
		cout << "sims::build_up_group_from_generators base: ";
		int_vec_print(cout, GA->get_base(), GA->base_len());
		cout << endl;

#if 0
		cout << "generators:" << endl;
		gens->print(cout);
		cout << endl;
#endif

		if (f_target_go) {
			cout << "target group order: " << *target_go << endl;
			}
		else {
			cout << "no target group order given" << endl;
			}
		cout << "verbose_level=" << verbose_level << endl;
		}
	group_order(G_order);
	K->group_order(K_order);
	D.mult(G_order, K_order, KG_order);
	for (level = GA->base_len() - 1; level >= 0; level--) {
		base_point = GA->base_i(level);
		if (f_vv) {
			cout << "level " << level << " base point "
					<< base_point << endl;
			}
		GA->find_strong_generators_at_level(
			GA->base_len(), GA->get_base(), level,
			*gens, subset_of_gens, verbose_level - 3);
		
		{
		schreier O;
	
		if (f_v) {
			cout << "calling O.init" << endl;
			}
		
		O.init(GA, verbose_level - 2);
		
		if (f_v) {
			cout << "calling O.init_generators" << endl;
			}
		O.init_generators(subset_of_gens, verbose_level - 2);
		
		if (f_vvv) {
			cout << "generators in schreier" << endl;
			O.print_generators();
			}

		if (f_vv) {
			cout << "computing orbit of point " << base_point << endl;
			}
		O.compute_point_orbit(base_point, 0);
		if (f_vv) {
			cout << "point " << base_point << " lies in an orbit "
					"of size " << O.orbit_len[0] << endl;
			if (FALSE) {
				O.print(cout);
				O.print_tables(cout, FALSE);
				}
			}
		for (j = 0; j < O.orbit_len[0]; j++) {
			if (FALSE) {
				cout << "level " << level << " coset rep " << j << endl;
				}
			O.coset_rep(j);
			if (FALSE) {
				GA->element_print(O.cosetrep, cout);
				cout << endl;
				}
			if (strip(O.cosetrep, GA->Elt2 /* residue */,
					drop_out_level, image, 0 /*verbose_level - 1*/)) {
				if (f_vv) {
					cout << "element strips through" << endl;
					if (FALSE /*f_vvv */) {
						cout << "residue=" << endl;
						GA->element_print_quick(GA->Elt2, cout);
						cout << endl;
						}
					}
				if (FALSE) {
					cout << "element strips through." << endl;
					cout << "if it is the identity element, that's OK,"
							<< endl;
					cout << "otherwise please add another base point,"
							<< endl;
					cout << "a point which is moved by the residue"
							<< endl;
					GA->element_print(GA->Elt2, cout);
					}
				if (!GA->element_is_one(GA->Elt2, FALSE)) {
					if (f_vvv) {
						cout << "the residue is not trivial, "
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
								cout << "element strips through kernel, "
										"residue = " << endl;
								if (f_vv) {
									KA->element_print(GA->Elt3, cout);
									cout << endl;
									}
								K->print(FALSE);
								K->print_basic_orbits();
								cout << "residue" << endl;
								KA->element_print_image_of_set(
										GA->Elt3, KA->base_len(), KA->get_base());
								cout << "Elt2" << endl;
								KA->element_print_image_of_set(
										GA->Elt2, KA->base_len(), KA->get_base());
								}
							if (!KA->element_is_one(GA->Elt3, FALSE)) {
								cout << "but the element is not the identity, "
										"something is wrong" << endl;
								GA->element_print(GA->Elt3, cout);
								cout << endl;
								exit(1);
								}
							}
						K->add_generator_at_level(GA->Elt3,
								drop_out_level, verbose_level - 3);
						if (f_vv) {
							cout << "the residue has been added as "
									"kernel generator at level "
									<< drop_out_level << endl;
							}
						f_added = TRUE;
						}
					else {
						if (f_vv) {
							cout << "action::induced_action: "
									"choosing n e w base point "
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
							cout << "n e w base point " << b
								<< " chosen, n e w base has length "
								<< GA->base_len() << endl;
							cout << "calling add_generator_at_level" << endl;
							}
						add_generator_at_level(GA->Elt2,
								GA->base_len() - 1, verbose_level - 3);
						if (f_vv) {
							cout << "the residue has been added at level "
									<< GA->base_len() - 1 << endl;
							}
						} // if b
					} // if ! element is one
				else {
					if (f_vv) {
						cout << "the residue is trivial" << endl;
						}
					}

				f_added = FALSE;
				}
			else {
				f_added = TRUE;
				add_generator_at_level(GA->Elt2,
						drop_out_level, 0 /*verbose_level - 1*/);
				}

			group_order(G_order);
			K->group_order(K_order);
			D.mult(G_order, K_order, KG_order);


			if (f_v && f_added) {
				cout << "level " << level << " coset " << j 
					<< " group of order increased to " << KG_order 
					<< " = " << G_order << " * " << K_order << endl;
				}
			if (f_vv) {
				cout << "level " << level << " coset " << j 
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
					cout << "reached the full group after "
							<< cnt << " iterations" << endl;
					}
				break;
				}
			if (c < 0) {
				if (TRUE) {
					cout << "sims::build_up_group_from_generators "
							"overshooting the expected group after "
							<< cnt << " iterations" << endl;
					cout << "n e w group order is " << KG_order
						<< " = " << G_order << " * " << K_order << endl;
					}
				//break;
				exit(1);
				}
			if (cnt > 10000) {
				cout << "sims::build_up_group_from_generators after "
						<< cnt << " iterations, we seem to be having "
						"problems reaching the target group order" << endl;
				cout << "group order = " << KG_order << endl;
				cout << "target group order = " << *target_go << endl;
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
		cout << "the n e w action has base_length = " << GA->base_len()
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
		cout << "found a group of order " << G_order << endl;
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
	longinteger_object old_go, go, go1;
	
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
		// now in S->schreier_gen
		group_order(go);
		A->element_move(Elt3, Elt2, 0);
		if (strip_and_add(Elt2, Elt1 /* residue */,
				verbose_level - 3)) {
			group_order(go1);
			if (f_vv) {
				cout << "closure_group: iteration " << i
						<< " the group has been extended, old order "
					<< go << " n e w group order " << go1 << endl;
				print_transversal_lengths();
				}
			if (f_v3) {
				cout << "original element:" << endl;
				A->element_print_quick(Elt3, cout);
				cout << "n e w generator:" << endl;
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

void sims::element_as_permutation(action *A_special,
		int elt_rk, int *perm, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::element_as_permutation" << endl;
		}
	int *Elt;
	
	Elt = NEW_int(A->elt_size_in_int);
	
	element_unrank_int(elt_rk, Elt);

	A_special->element_as_permutation(Elt, perm, 0);
	

	FREE_int(Elt);
	if (f_v) {
		cout << "sims::element_as_permutation done" << endl;
		}
}


int sims::least_moved_point_at_level(int lvl, int verbose_level)
// returns -1 if there are no generators
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, j, h, nbg, gen_idx, least_moved_point = -1;
	
	if (f_v) {
		cout << "sims::least_moved_point_at_level  lvl=" << lvl << endl;
		}
	nbg = nb_gen[lvl];
	if (f_v) {
		cout << "sims::least_moved_point_at_level  nbg=" << nbg << endl;
		}
	for (h = 0; h < nbg; h++) {
		gen_idx = gen_perm[h];
		if (f_vv) {
			cout << "sims::least_moved_point_at_level  h=" << h << endl;
			cout << "gen_idx=" << gen_idx << endl;
			A->element_print_quick(gens.ith(gen_idx), cout);
			}
		for (i = 0; i < A->degree; i++) {
			j = A->element_image_of(i, gens.ith(gen_idx), 0);
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

#if 0
int sims::identify_group(char *path_t144,
		char *discreta_home, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int group_idx;
	int h, i, j, *Elt;
	longinteger_object go;
	const char *fname = "group_generators.txt";

	if (f_v) {
		cout << "sims::identify_group" << endl;
		}
	group_order(go);
	{
	ofstream f(fname);

		// generators start from one

	f << gens.len << " " << A->degree << endl;
	for (h = 0; h < gens.len; h++) {
		Elt = gens.ith(h);
		for (i = 0; i < A->degree; i++) {
			j = A->element_image_of(i, Elt, 0);
			f << j + 1 << " ";
			}
		f << endl;
		}
	}
	if (f_v) {
		cout << "sims::identify_group written file "
				<< fname << " of size " << file_size(fname) << endl;
		}
	char cmd[2000];

	sprintf(cmd, "%s/t144.out -discreta_home %s "
			"group_generators.txt >log.tmp",
			path_t144, discreta_home);
	
	if (f_v) {
		cout << "sims::identify_group calling '"
				<< cmd << "'" << endl;
		}

	system(cmd);
		
	{
	ifstream f("result.txt");
	f >> group_idx;
	}
	if (f_v) {
		cout << "sims::identify_group: the group is "
				"isomorphic to group " << go << "#"
				<< group_idx << endl;
		}
	return group_idx;
}
#endif

int sims::mult_by_rank(int rk_a, int rk_b, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk_c;

	if (f_v) {
		cout << "sims::mult_by_rank" << endl;
		}
	element_unrank_int(rk_a, Elt1);
	element_unrank_int(rk_b, Elt2);
	A->element_mult(Elt1, Elt2, Elt3, 0);
	rk_c = element_rank_int(Elt3);
	return rk_c;
}

int sims::mult_by_rank(int rk_a, int rk_b)
{
	int rk_c;

	rk_c = mult_by_rank(rk_a, rk_b, 0);
	return rk_c;
}

int sims::invert_by_rank(int rk_a, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int rk_b;

	if (f_v) {
		cout << "sims::invert_by_rank" << endl;
		}
	element_unrank_int(rk_a, Elt1);
	A->element_invert(Elt1, Elt2, 0);
	rk_b = element_rank_int(Elt2);
	return rk_b;
}

int sims::conjugate_by_rank(int rk_a, int rk_b,
		int verbose_level)
// comutes b^{-1} * a * b
{
	int f_v = (verbose_level >= 1);
	int rk_c;

	if (f_v) {
		cout << "sims::conjugate_by_rank" << endl;
		}
	element_unrank_int(rk_a, Elt1); // Elt1 = a
	element_unrank_int(rk_b, Elt2); // Elt2 = b
	A->element_invert(Elt2, Elt3, 0); // Elt3 = b^{-1}
	A->element_mult(Elt3, Elt1, Elt4, 0);
	A->element_mult(Elt4, Elt2, Elt3, 0);
	rk_c = element_rank_int(Elt3);
	return rk_c;
}

int sims::conjugate_by_rank_b_bv_given(int rk_a,
		int *Elt_b, int *Elt_bv, int verbose_level)
// comutes b^{-1} * a * b
{
	int f_v = (verbose_level >= 1);
	int rk_c;

	if (f_v) {
		cout << "sims::conjugate_by_rank_b_bv_given" << endl;
		}
	element_unrank_int(rk_a, Elt1); // Elt1 = a
	A->element_mult(Elt_bv, Elt1, Elt4, 0);
	A->element_mult(Elt4, Elt_b, Elt3, 0);
	rk_c = element_rank_int(Elt3);
	return rk_c;
}

int sims::get_orbit(int i, int j)
{
	if (orbit == NULL) {
		cout << "sims::get_orbit orbit == NULL i=" << i << " j=" << j << endl;
		exit(1);
	}
	if (i < 0 || i >= my_base_len) {
		cout << "sims::get_orbit i < 0 || i >= A->base_len(), i = " << i << endl;
		cout << "my_base_len=" << my_base_len << endl;
		exit(1);
	}
	if (orbit[i] == NULL) {
		cout << "sims::get_orbit orbit[i] == NULL i=" << i << " j=" << j << endl;
		exit(1);
	}
	if (j >= transversal_length) {
		cout << "sims::get_orbit  j >= transversal_length" << endl;
		cout << "j=" << j << endl;
		cout << "transversal_length=" << transversal_length << endl;
		exit(1);
	}
	return orbit[i][j];
}

int sims::get_orbit_inv(int i, int j)
{
	if (orbit_inv == NULL) {
		cout << "sims::get_orbit_inv orbit_inv == NULL i=" << i << " j=" << j << endl;
		exit(1);
	}
	if (i < 0 || i >= my_base_len) {
		cout << "sims::get_orbit i < 0 || i >= my_base_len, i = " << i << endl;
		cout << "my_base_len=" << my_base_len << endl;
		exit(1);
	}
	if (orbit_inv[i] == NULL) {
		cout << "sims::get_orbit_inv orbit_inv[i] == NULL i=" << i << " j=" << j << endl;
		exit(1);
	}
	if (j >= transversal_length) {
		cout << "sims::get_orbit_inv  j >= transversal_length" << endl;
		cout << "j=" << j << endl;
		cout << "transversal_length=" << transversal_length << endl;
		exit(1);
	}
	return orbit_inv[i][j];
}

int sims::get_orbit_length(int i)
{
	if (i < 0 || i >= my_base_len) {
		cout << "sims::get_orbit_length i < 0 || i >= my_base_len, i = " << i << endl;
		cout << "my_base_len=" << my_base_len << endl;
		exit(1);
	}
	return orbit_len[i];
}

}}


