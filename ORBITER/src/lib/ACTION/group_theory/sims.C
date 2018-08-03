// sims.C
//
// Anton Betten
// December 21, 2003

#include "GALOIS/galois.h"
#include "action.h"

sims::sims()
{
	null();
}

void sims::null()
{
	A = NULL;
	my_base_len = 0;
	nb_images = 0;
	images = NULL;
	gen_depth = NULL;
	gen_perm = NULL;
	nb_gen = NULL;
	path = NULL;
	images = NULL;
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
	cosetrep = NULL;
	cosetrep_tmp = NULL;
	schreier_gen = NULL;
	schreier_gen1 = NULL;
}

sims::sims(action *A)
{
	init(A);
}

sims::~sims()
{
	freeself();
}

void sims::freeself()
{
	INT i;
	INT f_v = FALSE;
	
	if (f_v) {
		cout << "sims::freeself freeing gen_depth" << endl;
		}
	if (gen_depth)
		FREE_INT(gen_depth);
	if (gen_perm)
		FREE_INT(gen_perm);
	if (nb_gen)
		FREE_INT(nb_gen);
	if (path)
		FREE_INT(path);
		
	if (f_v) {
		cout << "sims::freeself freeing orbit, my_base_len=" << my_base_len << endl;
		}
	if (orbit) {
		for (i = 0; i < my_base_len; i++) {
			if (f_v) {
				cout << "sims::freeself freeing orbit i=" << i << endl;
				}
			if (f_v) {
				cout << "sims::freeself freeing orbit[i]" << endl;
				}
			FREE_INT(orbit[i]);
			if (f_v) {
				cout << "sims::freeself freeing orbit_inv[i]" << endl;
				}
			FREE_INT(orbit_inv[i]);
			if (f_v) {
				cout << "sims::freeself freeing prev[i]" << endl;
				}
			FREE_INT(prev[i]);
			if (f_v) {
				cout << "sims::freeself freeing label[i]" << endl;
				}
			FREE_INT(label[i]);
			}
		if (f_v) {
			cout << "sims::freeself freeing orbit"<< endl;
			}
		FREE_PINT(orbit);
		FREE_PINT(orbit_inv);
		FREE_PINT(prev);
		FREE_PINT(label);
		}

	if (f_v) {
		cout << "sims::freeself freeing orbit_len" << endl;
		}
	if (orbit_len)
		FREE_INT(orbit_len);
	if (Elt1)
		FREE_INT(Elt1);
	if (Elt2)
		FREE_INT(Elt2);
	if (Elt3)
		FREE_INT(Elt3);
	if (Elt4)
		FREE_INT(Elt4);
	if (strip1)
		FREE_INT(strip1);
	if (strip2)
		FREE_INT(strip2);
	if (eltrk1)
		FREE_INT(eltrk1);
	if (eltrk2)
		FREE_INT(eltrk2);
	if (cosetrep)
		FREE_INT(cosetrep);
	if (cosetrep_tmp)
		FREE_INT(cosetrep_tmp);
	if (schreier_gen)
		FREE_INT(schreier_gen);
	if (schreier_gen1)
		FREE_INT(schreier_gen1);
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
	INT i;
	
	if (images) {
		for (i = 0; i < nb_images; i++) {
			FREE_INT(images[i]);
			}
		nb_images = 0;
		FREE_PINT(images);
		images = NULL;
		}
}

void sims::init_images(INT nb_images)
{
#if 1
	INT i, j; //, a;
	
	if (A == NULL) {
		cout << "sims::init_images() action is NULL" << endl;
		exit(1);
		}
	delete_images();
	sims::nb_images = nb_images;
	images = NEW_PINT(nb_images);
	for (i = 0; i < nb_images; i++) {
		images[i] = NEW_INT(A->degree);
		for (j = 0; j < A->degree; j++) {
			images[i][j] = -1;
			//a = A->image_of(gens.ith(i), j);
			//images[i][j] = a;
			//images[i][A->degree + a] = j;
			}
		}
#endif
}

void sims::images_append()
{
#if 1
	INT **new_images = NEW_PINT(nb_images + 1);
	INT i, j; //, a;
	
	new_images[nb_images] = NEW_INT(A->degree);
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
		FREE_PINT(images);
	images = new_images;
	nb_images++;
#endif
}

void sims::init(action *A)
// initializes the trivial group with the base as given in A
{
	INT i;
	
	init_without_base(A);
	
	my_base_len = A->base_len;
	
	orbit = NEW_PINT(my_base_len);
	orbit_inv = NEW_PINT(my_base_len);
	prev = NEW_PINT(my_base_len);
	label = NEW_PINT(my_base_len);
	for (i = 0; i < my_base_len; i++) {
		orbit[i] = NEW_INT(A->degree);
		orbit_inv[i] = NEW_INT(A->degree);
		prev[i] = NEW_INT(A->degree);
		label[i] = NEW_INT(A->degree);
		}
	
	FREE_INT(nb_gen);
	
	
	nb_gen = NEW_INT(my_base_len + 1);
	for (i = 0; i <= my_base_len; i++) {
		nb_gen[i] = 0;
		}
	
	path = NEW_INT(my_base_len);

	orbit_len = NEW_INT(my_base_len);
	
	for (i = 0; i < my_base_len; i++) {
		initialize_table(i);
		}
	
}

void sims::init_without_base(action *A)
{	
	sims::A = A;
	nb_images = 0;
	images = NULL;
	
	
	gens.init(A);
	gens_inv.init(A);
	
	nb_gen = NEW_INT(1);

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	Elt4 = NEW_INT(A->elt_size_in_INT);
	strip1 = NEW_INT(A->elt_size_in_INT);
	strip2 = NEW_INT(A->elt_size_in_INT);
	eltrk1 = NEW_INT(A->elt_size_in_INT);
	eltrk2 = NEW_INT(A->elt_size_in_INT);
	cosetrep = NEW_INT(A->elt_size_in_INT);
	cosetrep_tmp = NEW_INT(A->elt_size_in_INT);
	schreier_gen = NEW_INT(A->elt_size_in_INT);
	schreier_gen1 = NEW_INT(A->elt_size_in_INT);
}

void sims::reallocate_base(INT old_base_len, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	INT *old_nb_gen = nb_gen;
	INT *old_path = path;
	INT *old_orbit_len = orbit_len;
	INT **old_orbit = orbit;
	INT **old_orbit_inv = orbit_inv;
	INT **old_prev = prev;
	INT **old_label = label;
	
	if (f_v) {
		cout << "sims::reallocate_base from " 
			<< old_base_len << " to " << A->base_len << endl;
		}

	my_base_len = A->base_len;
	
	nb_gen = NEW_INT(my_base_len + 1);
	path = NEW_INT(my_base_len);
	orbit_len = NEW_INT(my_base_len);
	orbit = NEW_PINT(my_base_len);
	orbit_inv = NEW_PINT(my_base_len);
	prev = NEW_PINT(my_base_len);
	label = NEW_PINT(my_base_len);
	for (i = 0; i < old_base_len /* A->base_len - 1 */; i++) {
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
		orbit[i] = NEW_INT(A->degree);
		orbit_inv[i] = NEW_INT(A->degree);
		prev[i] = NEW_INT(A->degree);
		label[i] = NEW_INT(A->degree);
		initialize_table(i);
		init_trivial_orbit(i);
		}
	nb_gen[my_base_len] = 0;
#if 0
	nb_gen[A->base_len - 1] = old_nb_gen[A->base_len - 1];
	nb_gen[A->base_len] = 0;
	path[A->base_len - 1] = 0;
	orbit[A->base_len - 1] = NEW_INT(A->degree);
	orbit_inv[A->base_len - 1] = NEW_INT(A->degree);
	prev[A->base_len - 1] = NEW_INT(A->degree);
	label[A->base_len - 1] = NEW_INT(A->degree);
	initialize_table(A->base_len - 1);
	init_trivial_orbit(A->base_len - 1);
#endif
	if (old_nb_gen)
		FREE_INT(old_nb_gen);
	if (old_path)
		FREE_INT(old_path);
	if (old_orbit_len)
		FREE_INT(old_orbit_len);
	if (old_orbit)
		FREE_PINT(old_orbit);
	if (old_orbit_inv)
		FREE_PINT(old_orbit_inv);
	if (old_prev)
		FREE_PINT(old_prev);
	if (old_label)
		FREE_PINT(old_label);
}

void sims::initialize_table(INT i)
{
	INT j;
	
	perm_identity(orbit[i], A->degree);
	perm_identity(orbit_inv[i], A->degree);
	for (j = 0; j < A->degree; j++) {
		prev[i][j] = -1;
		label[i][j] = -1;
		}
	orbit_len[i] = 0;
}

void sims::init_trivial_group(INT verbose_level)
// clears the generators array, 
// and sets the i-th transversal to contain
// only the i-th base point (for all i).
{
	INT f_v = (verbose_level >= 2);
	INT f_vv = (verbose_level >= 3);
	INT i;
	
	if (f_v) {
		cout << "sims::init_trivial_group" << endl;
		}
	if (my_base_len != A->base_len) {
		cout << "sims::init_trivial_group: my_base_len != A->base_len" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "before init_generators" << endl;
		}
	init_generators(0, NULL, 0/*verbose_level - 3*/);
	for (i = 0; i < A->base_len; i++) {
		if (f_vv) {
			cout << "before init_trivial_orbit i=" << i << endl;
			}
		init_trivial_orbit(i);
		}
}

void sims::init_trivial_orbit(INT i)
{
	INT coset_of_base_point;
	
	if (my_base_len != A->base_len) {
		cout << "sims::init_trivial_orbit: my_base_len != A->base_len" << endl;
		exit(1);
		}
	coset_of_base_point = orbit_inv[i][A->base[i]];
	if (coset_of_base_point) {
		swap_points(i, coset_of_base_point, 0);
		}
	//cout << "sims::init_trivial_orbit " << i << " : " << A->base[i] << endl;
	//cout << "orbit[i][0] = " << orbit[i][0] << endl;
	orbit_len[i] = 1;
}

void sims::init_generators(vector_ge &generators, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "sims::init_generators" << endl;
		cout << "generators.len=" << generators.len << endl;
		}
	if (generators.len) {
		init_generators(generators.len, generators.ith(0), verbose_level);
		}
	else	{
		init_generators(generators.len, NULL, verbose_level);
		}
}

void sims::init_generators(INT nb, INT *elt, INT verbose_level)
// copies the given elements into the generator array, 
// then computes depth and perm
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i;
	
	if (f_v) {
		cout << "sims::init_generators() nb = " << nb << endl;
		}
	gens.allocate(nb);
	gens_inv.allocate(nb);
	for (i = 0; i < nb; i++) {
		if (f_vv) {
			cout << "sims::init_generators i = " << i << endl;
			}
		gens.copy_in(i, elt + i * A->elt_size_in_INT);
		if (f_vvv) {
			A->element_print_quick(elt + i * A->elt_size_in_INT, cout);
			}
		A->element_invert(elt + i * A->elt_size_in_INT, gens_inv.ith(i), FALSE);
		}
	if (f_v) {
		cout << "sims::init_images()" << endl;
		}
	init_images(nb);
	if (f_v) {
		cout << "sims::init_generator_depth_and_perm()" << endl;
		}
	init_generator_depth_and_perm(verbose_level);
	if (f_v) {
		cout << "sims::init_generator_depth_and_perm() done" << endl;
		}
}

void sims::init_generators_by_hdl(INT nb_gen, INT *gen_hdl)
{
	INT i;
	
	gens.allocate(nb_gen);
	gens_inv.allocate(nb_gen);

	for (i = 0; i < nb_gen; i++) {
		//cout << "sims::init_generators i = " << i << endl;
		A->element_retrieve(gen_hdl[i], gens.ith(i), FALSE);
		A->element_invert(gens.ith(i), gens_inv.ith(i), FALSE);
		}
	init_images(nb_gen);	
	init_generator_depth_and_perm(FALSE);

}

void sims::init_generator_depth_and_perm(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, d;
	
	if (f_v) {
		cout << "sims::init_generator_depth_and_perm" << endl;
		cout << "gens.len=" << gens.len << endl;
		}
	if (my_base_len != A->base_len) {
		cout << "sims::init_generator_depth_and_perm: my_base_len != A->base_len" << endl;
		exit(1);
		}
	for (i = 0; i <= A->base_len; i++) {
		nb_gen[i] = 0;
		}
	gen_depth = NEW_INT(gens.len);
	gen_perm = NEW_INT(gens.len);
	for (i = 0; i < gens.len; i++) {
		gen_perm[i] = i;
		gen_depth[i] = generator_depth(i);
		if (f_vv) {
			cout << "generator " << i << " has depth " << gen_depth[i] << endl;
			}
		if (i) {
			if (gen_depth[i] > gen_depth[i - 1]) {
				cout << "sims::init_generator_depth_and_perm() generators must be of decreasing depth" << endl;
				exit(1);
				}
			}
		}
	nb_gen[A->base_len] = 0;
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

void sims::add_generator(INT *elt, INT verbose_level)
// adds elt to list of generators, 
// computes the depth of the element, 
// updates the arrays gen_depth, gen_perm and nb_gen accordingly
// does not change the transversals
{
	INT f_v = (verbose_level >= 1);
	INT old_nb_gen, idx, depth, i;
	INT *new_gen_depth = NEW_INT(gens.len + 1);
	INT *new_gen_perm = NEW_INT(gens.len + 1);
	
	if (f_v) {
		cout << "sims::add_generator generator no " << gens.len << endl;
		A->element_print_quick(elt, cout);
		cout << endl;
		}
	if (my_base_len != A->base_len) {
		cout << "sims::add_generator: my_base_len != A->base_len" << endl;
		exit(1);
		}
	old_nb_gen = idx = gens.len;
	for (i = 0; i < old_nb_gen; i++) {
		new_gen_depth[i] = gen_depth[i];
		new_gen_perm[i] = gen_perm[i];
		}
	if (gen_depth) {
		FREE_INT(gen_depth);
		FREE_INT(gen_perm);
		}
	gen_depth = new_gen_depth;
	gen_perm = new_gen_perm;
	
	gens.append(elt);
	gens_inv.append(elt);
	A->element_invert(elt, gens_inv.ith(idx), FALSE);
	
	images_append();

	
	depth = generator_depth(idx);
	new_gen_depth[idx] = depth;
	for (i = old_nb_gen - 1; i >= nb_gen[depth]; i--) {
		gen_perm[i + 1] = gen_perm[i];
		}
	gen_perm[nb_gen[depth]] = idx;
	for (i = depth; i >= 0; i--) {
		nb_gen[i]++;
		}
}

void sims::create_group_tree(const BYTE *fname, INT f_full, INT verbose_level)
{
	INT i, j, h, go, l;
	longinteger_object Go;
	INT *Elt;
	INT *Fst;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(Go);
	go = Go.as_INT();
	//cout << "Group of order " << go << endl;

	{
	ofstream fp(fname);
	
	l = last_moved_base_point() + 1;
	Fst = NEW_INT(l);
	for (i = 0; i < l; i++) {
		if (i == 0) {
			Fst[i] = 0;
			}
		else {
			Fst[i] = Fst[i - 1] + orbit_len[i - 1];
			}
		}
	if (f_full) {
		for (h = 0; h < go; h++) {
			element_unrank_INT(h, Elt);
			//cout << "Element " << h << " / " << go << " : path=";
			//INT_vec_print(cout, S->path, S->A->base_len);
			//cout << endl;
			fp << l;
			for (i = 0; i < l; i++) {
				j = path[i];
				fp << " " << Fst[i] + j;
				coset_rep(i, j, 0 /* verbose_level*/);
				// coset rep now in cosetrep
				if (i) {
					//cout << "*" << endl;
					}
				//A->element_print_quick(S->cosetrep, cout);
				}
			fp << endl;
			//cout << "=" << endl;
			//A->element_print_quick(Elt, cout);
			//A->element_print_as_permutation(Elt, cout);
			//cout << endl;
			}
		}
	else {
		for (h = l - 1; h >= 0; h--) {
			for (j = 0; j < orbit_len[h]; j++) {
				if (h < l - 1 && j == 0) {
					continue;
					}
				INT_vec_zero(path, l);
				path[h] = j;
				fp << h + 1;
				for (i = 0; i <= h; i++) {
					j = path[i];
					fp << " " << orbit[i][j];
					}
				fp << endl;
				}
			}
		}
	fp << "-1" << endl;
	}
	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	FREE_INT(Elt);
	FREE_INT(Fst);
}

void sims::print_transversals()
{
	INT i, j, l;
	cout << "sims data structure, a group of order ";
	print_group_order(cout);
	cout << endl;
	l = last_moved_base_point();
	for (i = 0 ; i <= l; i++) {
		for (j = 0; j < orbit_len[i]; j++) {
			cout << orbit[i][j] << " ";
			}
		cout << "(length " << orbit_len[i] << ")" << endl;
#if 0
		cout << "printing orbit_inv:" << endl;
		for (j = 0; j < A->degree; j++) {
			cout << orbit_inv[i][j] << " ";
			}
#endif
		cout << endl;
		}
}

void sims::print_transversals_short()
{
	INT i, j, l;

	if (my_base_len != A->base_len) {
		cout << "sims::print_transversals_short: my_base_len != A->base_len" << endl;
		exit(1);
		}
	l = last_moved_base_point();
	for (i = 0 ; i <= l; i++) {
		cout << "(";
		for (j = 0; j < orbit_len[i]; j++) {
			cout << orbit[i][j];
			if (j < orbit_len[i] - 1)
				cout << ",";
			}
		cout << ")";
		if (i < l)
			cout << ", ";
		}
	cout << endl;
}

void sims::print_transversal_lengths()
{
	INT_vec_print(cout, orbit_len, A->base_len);
	cout << endl;
#if 0
	INT i, l;
	
	l = last_moved_base_point();
	for (i = 0 ; i <= l; i++) {
		cout << i << " base point " << orbit[i][0] << " length " << orbit_len[i] << endl;
		}
#endif
}

void sims::print_orbit_len()
{
	INT i;
	
	for (i = 0; i < A->base_len; i++) {
		cout << orbit_len[i] << " ";
		}
	cout << endl;
}



void sims::print(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i, j;
	
	j = last_moved_base_point();
	cout << "sims data structure, a group of order ";
	print_group_order(cout);
	cout << endl;
	cout << "number of generators = " << nb_gen[0] << endl;
	cout << "last moved base point = " << j << endl;
	if (f_v) {
		cout << "depth : base pt : transversal length : # generators" << endl;
		for (i = 0; i <= j; i++) {
			cout << i << " : " << A->base[i] << " : " << orbit_len[i] << " : " << nb_gen[i] - nb_gen[i + 1] << endl;
			}
		cout << endl;
		}
	if (f_vv) {
		print_generator_depth_and_perm();
		}
	if (f_vvv) {
		print_generators();
		print_basic_orbits();
		}
}

void sims::print_generators()
{
	INT i, j, nbg, nbg1, gen_idx;

	cout << "generators are:" << endl;
	//gens.print(cout);

	for (i = A->base_len - 1; i >= 0; i--) {
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		cout << "level " << i << ":" << endl;
		//cout << "i=" << i << " nbg1=" << nbg1 << " nbg=" << nbg << endl;
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];

			cout << "generator " << gen_idx << ":" << endl;
			A->element_print(gens.ith(gen_idx), cout);
			cout << endl;
			}
		cout << "orbit_len[" << i << "]=" << orbit_len[i] << endl;
		}
	cout << endl;
}

void sims::print_generators_tex(ostream &ost)
{
	INT i, j, nbg, nbg1, gen_idx, cnt, f_first;

	ost << "\\begin{align*}" << endl;
	cnt = 0;
	f_first = TRUE;
	for (i = A->base_len - 1; i >= 0; i--) {
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		//cout << "i=" << i << " nbg1=" << nbg1 << " nbg=" << nbg << endl;
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];

			if ((cnt % 3) == 0) {
				if (!f_first) {
					ost << "\\\\" << endl;
					}
				ost << "&" << endl;
				}
			A->element_print_latex(gens.ith(gen_idx), ost);
			cnt++;
			f_first = FALSE;
			if (j < nbg - 1) {
				ost << ", \\; " << endl;
				}
			}
		ost << ";_{" << orbit_len[i] << "}" << endl;
		}
	ost << "\\end{align*}" << endl;
}


void sims::print_generators_as_permutations()
{
	INT i, l;
	
	cout << "generators as permutations:" << endl;
	l = gens.len;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		A->element_print_as_permutation(gens.ith(i), cout);
		cout << endl;
		}
	cout << endl;
}

void sims::print_generators_as_permutations_override_action(action *A)
{
	INT i, l;
	
	cout << "generators as permutations (override action):" << endl;
	l = gens.len;
	for (i = 0; i < l; i++) {
		cout << i << " : ";
		A->element_print_as_permutation(gens.ith(i), cout);
		cout << endl;
		}
	cout << endl;
}

void sims::print_basic_orbits()
{
	INT i;
	
	if (my_base_len != A->base_len) {
		cout << "sims::print_basic_orbits: my_base_len != A->base_len" << endl;
		exit(1);
		}
	for (i = 0 ; i < A->base_len /* <= j */; i++) {
		print_basic_orbit(i);
		}
}

void sims::print_basic_orbit(INT i)
{
	INT j;
	
	cout << "basic orbit " << i << " of length " << orbit_len[i] << endl; 
	cout << "j : orbit[i][j] : prev[i][j] : label[i][j]" << endl;
	for (j = 0; j < orbit_len[i] /* A->degree */; j++) {
		//coset_rep[i][j];
		//coset_rep_inv(i);
		if (j == orbit_len[i])
			cout << "======================================" << endl;
		cout << setw(5) << j << " : " 
			<< setw(5) << orbit[i][j] << " : " 
			<< setw(5) << prev[i][j] << " : " 
			<< setw(5) << label[i][j];
		cout << endl;
		}
	cout << endl;
}

#if 0
void sims::sort()
{
	INT i, j;
	
	j = last_moved_base_point();
	for (i = 0 ; i <= j; i++) {
		sort_basic_orbit(i);
		}
}

void sims::sort_basic_orbit(INT i)
{
	INT j1, j2, a, b;
	
	cout << "sims::sort_basic_orbit Warning, this function uses bubble sort for an orbit of size " << orbit_len[i] << endl;
	for (j1 = 0; j1 < orbit_len[i]; j1++) {
		for (j2 = j1 + 1; j2 < orbit_len[i]; j2++) {
			if (orbit[i][j1] > orbit[i][j2]) {
				if (j1 == 0) {
					cout << "sims::sort_basic_orbit() root must be first" << endl;
					exit(1);
					}
				a = orbit[i][j1];
				b = orbit[i][j2];
				orbit[i][j1] = b;
				orbit[i][j2] = a;
				orbit_inv[i][a] = j2;
				orbit_inv[i][b] = j1;
				a = prev[i][j1];
				b = prev[i][j2];
				prev[i][j1] = b;
				prev[i][j2] = a;
				a = label[i][j1];
				b = label[i][j2];
				label[i][j1] = b;
				label[i][j2] = a;
				}
			}
		}
}
#endif

void sims::print_generator_depth_and_perm()
{
	INT i;
	
	cout << "depth:" << endl;
	for (i = 0; i < gens.len; i++) {
		cout << gen_depth[i];
		if (i < gens.len - 1)
			cout << ", ";
		}
	cout << endl;
	cout << "perm:" << endl;
	for (i = 0; i < gens.len; i++) {
		cout << gen_perm[i];
		if (i < gens.len - 1)
			cout << ", ";
		}
	cout << endl;
	cout << "nb_gen:" << endl;
	for (i = 0; i <= A->base_len; i++) {
		cout << nb_gen[i];
		if (i < A->base_len)
			cout << ", ";
		}
	cout << endl;
}

INT sims::generator_depth(INT gen_idx)
// returns the index of the first base point 
// which is moved by a given generator. 
{
	INT i, bi, j;
	
	for (i = 0; i < A->base_len; i++) {
		bi = A->base[i];
		j = get_image(bi, gen_idx);
		if (j != bi)
			return i;
		}
	return A->base_len;
}

INT sims::generator_depth(INT *elt)
// returns the index of the first base point 
// which is moved by the given element
{
	INT i, bi, j;
	
	for (i = 0; i < A->base_len; i++) {
		bi = A->base[i];
		j = get_image(bi, elt);
		if (j != bi)
			return i;
		}
	return A->base_len;
}

void sims::group_order(longinteger_object &go)
{
	longinteger_domain D;
	
	//cout << "sims::group_order before D.multiply_up" << endl;
	//cout << "A->base_len=" << A->base_len << endl;
	//cout << "orbit_len=";
	//INT_vec_print(cout, orbit_len, A->base_len);
	//cout << endl;
	D.multiply_up(go, orbit_len, A->base_len);
	//cout << "sims::group_order after D.multiply_up" << endl;
	
}

INT sims::group_order_INT()
{
	longinteger_object go;

	group_order(go);
	return go.as_INT();
}

void sims::print_group_order(ostream &ost)
{
	longinteger_object go;
	group_order(go);
	cout << go;
}

void sims::print_group_order_factored(ostream &ost)
{
	INT i, j, f_first = TRUE;
	
	j = last_moved_base_point();
	for (i = 0; i <= j; i++) {
		if (f_first) {
			f_first = FALSE;
			}
		else {
			cout << " * ";
			}
		cout << orbit_len[i];
		}
}

INT sims::is_trivial_group()
{
	INT j = last_moved_base_point();
	
	if (j == -1)
		return TRUE;
	else 
		return FALSE;
}

INT sims::last_moved_base_point()
// j == -1 means the group is trivial
{
	INT j;
	
	for (j = A->base_len - 1; j >= 0; j--) {
		if (orbit_len[j] != 1)
			break;
		}
	return j;
}

INT sims::get_image(INT i, INT gen_idx)
// get the image of a point i under generator gen_idx, goes through a 
// table of stored images by default. Computes the image only if not yet available.
{
	INT a;
	
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

INT sims::get_image(INT i, INT *elt)
// get the image of a point i under a given group element, does not goes through a table.
{
	return A->element_image_of(i, elt, FALSE);
}

void sims::swap_points(INT lvl, INT i, INT j)
// swaps two points given by their cosets
{
	INT pi, pj;
	
	pi = orbit[lvl][i];
	pj = orbit[lvl][j];
	orbit[lvl][i] = pj;
	orbit[lvl][j] = pi;
	orbit_inv[lvl][pi] = j;
	orbit_inv[lvl][pj] = i;
}

void sims::random_element(INT *elt, INT verbose_level)
// compute a random element among the group elements represented by the chain
// (chooses random cosets along the stabilizer chain)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "sims::random_element" << endl;
		cout << "sims::random_element orbit_len=";
		INT_vec_print(cout, orbit_len, A->base_len);
		cout << endl;
		//cout << "transversals:" << endl;
		//print_transversals();
		}
	for (i = 0; i < A->base_len; i++) {
		path[i] = random_integer(orbit_len[i]);
		}
	if (f_v) {
		cout << "sims::random_element" << endl;
		cout << "path=";
		INT_vec_print(cout, path, A->base_len);
		cout << endl;
		}
	element_from_path(elt, verbose_level /*- 1 */);
	if (f_v) {
		cout << "sims::random_element done" << endl;
		}
}

void sims::random_element_of_order(INT *elt, INT order, INT verbose_level)
{
	INT f_v = (verbose_level >=1);
	INT f_vv = (verbose_level >=2);
	INT o, n, cnt;
	
	if (f_v) {
		cout << "sims::random_element_of_order " << order << endl;
		}
	cnt = 0;
	while (TRUE) {
		cnt++;
		random_element(elt, verbose_level - 1);
		o = A->element_order(elt);
		if ((o % order) == 0) {
			break;
			}
		}
	if (f_v) {
		cout << "sims::random_element_of_order " << o << " found with " << cnt << " trials" << endl;
		}
	if (f_vv) {
		A->element_print_quick(elt, cout);
		}
	n = o / order;
	if (f_v) {
		cout << "sims::random_element_of_order we will raise to the " << n << "-th power" << endl;
		}
	A->element_power_INT_in_place(elt, n, 0);
	if (f_vv) {
		A->element_print_quick(elt, cout);
		}
}

void sims::random_elements_of_order(vector_ge *elts, INT *orders, INT nb, INT verbose_level)
{
	INT i;
	
	elts->init(A);
	elts->allocate(nb);
	for (i = 0; i < nb; i++) {
		random_element_of_order(elts->ith(i), orders[i], verbose_level);
		}
}

void sims::path_unrank_INT(INT a)
{
	INT h, l;
	
	for (h = A->base_len - 1; h >= 0; h--) {
		l = orbit_len[h];

		path[h] = a % l;
		a = a / l;
		}
}

INT sims::path_rank_INT()
{
	INT h, a;
	
	a = 0;
	for (h = 0; h < A->base_len; h++) {
		if (h) {
			a *= orbit_len[h];
			}
		a += path[h];
		}
	return a;
}

void sims::element_from_path(INT *elt, INT verbose_level)
// given coset representatives in path[], the corresponding 
// element is multiplied.
// uses eltrk1, eltrk2
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j;
	
	if (f_v) {
		cout << "sims::element_from_path" << endl;
		}
	if (f_vv) {
		cout << "path=";
		INT_vec_print(cout, path, A->base_len);
		cout << endl;
		cout << "A->degree=" << A->degree << endl;
		}
#if 0
	if (f_v) {
		cout << "i : orbit[0][i] : orbit_inv[0][i] : prev[0][i] : label[0][i]" << endl;
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
	for (i = 0; i < A->base_len; i++) {
		j = path[i];
		if (f_v) {
			cout << "sims::element_from_path level " << i << " coset " << j << " before coset_rep" << endl;
			}
		coset_rep(i, j, verbose_level);
		// coset rep now in cosetrep


		if (f_v) {
			cout << "sims::element_from_path level " << i << " coset " << j << " after coset_rep" << endl;
			}

		if (f_vv) {
			cout << "sims::element_from_path level " << i << " coset " << j << ":" << endl;
			cout << "cosetrep:" << endl;
			A->element_print_quick(cosetrep, cout);
			cout << endl;
			}
		
		//A->element_print_as_permutation(cosetrep, cout);
		//cout << endl;
		
		// pre multiply the coset representative:
		A->element_mult(cosetrep, eltrk1, eltrk2, 0);
		A->element_move(eltrk2, eltrk1, 0);
		}
	A->element_move(eltrk1, elt, 0);
	if (f_v) {
		cout << "sims::element_from_path done" << endl;
		}
}

void sims::element_from_path_inv(INT *elt)
// very specialized routine, used in backtrack.C action_is_minimal_recursion
// used coset_rep_inv instead of coset_rep, multiplies left-to-right
//
// given coset representatives in path[], the corresponding 
// element is multiplied.
// uses eltrk1, eltrk2
{
	INT i, j;
	
#if 0
	cout << "sims::element_from_path() path=";
	for (i = 0; i < A->base_len; i++) {
		cout << path[i] << " ";
		}
	cout << endl;
#endif
	A->element_one(eltrk1, FALSE);
	for (i = 0; i < A->base_len; i++) {
		j = path[i];
		
		coset_rep_inv(i, j, 0 /* verbose_level */);
		// coset rep now in cosetrep
		
		//A->element_print_as_permutation(cosetrep, cout);
		//cout << endl;
		
		// pre multiply the coset representative:
		A->element_mult(eltrk1, cosetrep, eltrk2, FALSE);
		A->element_move(eltrk2, eltrk1, FALSE);
		}
	A->element_move(eltrk1, elt, FALSE);
}

void sims::element_unrank(longinteger_object &a, INT *elt, INT verbose_level)
// Returns group element whose rank is a. 
// the elements represented by the chain are enumerated 0, ... go - 1
// with the convention that 0 always stands for the identity element.
// The computed group element will be computed into Elt1
{
	INT f_v = (verbose_level >= 1);
	INT ii, l, r;
	longinteger_domain D;
	longinteger_object q;
	
	if (f_v) {
		cout << "sims::element_unrank rk=" << a << endl;
		}
	for (ii = A->base_len - 1; ii >= 0; ii--) {
		l = orbit_len[ii];

		D.integral_division_by_INT(a, l, q, r);
		q.assign_to(a);
		
		path[ii] = r;
		//cout << r << " ";
		}
	//cout << endl;
	if (f_v) {
		cout << "sims::element_unrank path=";
		INT_vec_print(cout, path, A->base_len);
		cout << endl;
		}
	element_from_path(elt, 0);
	if (f_v) {
		cout << "sims::element_unrank done" << endl;
		}
}

void sims::element_unrank(longinteger_object &a, INT *elt)
// Returns group element whose rank is a. 
// the elements represented by the chain are enumerated 0, ... go - 1
// with the convention that 0 always stands for the identity element.
// The computed group element will be computed into Elt1
{
	INT ii, l, r;
	longinteger_domain D;
	longinteger_object q;
	
	for (ii = A->base_len - 1; ii >= 0; ii--) {
		l = orbit_len[ii];

		D.integral_division_by_INT(a, l, q, r);
		q.assign_to(a);
		
		path[ii] = r;
		//cout << r << " ";
		}
	//cout << endl;
	element_from_path(elt, 0);
}

void sims::element_rank(longinteger_object &a, INT *elt)
// Computes the rank of the element in elt into a.
// uses eltrk1, eltrk2
{
	INT i, j, bi, jj, l;
	longinteger_domain D;
	longinteger_object b, c;
	
	A->element_move(elt, eltrk1, FALSE);
	a.zero();
	for (i = 0; i < A->base_len; i++) {
		bi = A->base[i];
		l = orbit_len[i];
		
		if (i > 0) {
			b.create(l);
			D.mult(a, b, c);
			c.assign_to(a);
			}
		
		jj = A->element_image_of(bi, eltrk1, FALSE);
		//cout << "at level " << i << ", maps bi = " << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {
			cout << "sims::element_rank() j >= orbit_len[i]" << endl;
			cout << "i=" << i << endl;
			cout << "jj=bi^elt=" << jj << endl;
			cout << "j=orbit_inv[i][jj]=" << j << endl;
			cout << "base=";
			INT_vec_print(cout, A->base, A->base_len);
			cout << endl;
			cout << "orbit_len=";
			INT_vec_print(cout, orbit_len, A->base_len);
			cout << endl;
			cout << "elt=" << endl;
			A->element_print(eltrk1, cout);
			exit(1);
			}
		b.create(j);
		D.add(a, b, c);
		c.assign_to(a);
		
		coset_rep_inv(i, j, 0 /* verbose_level */);

		A->element_mult(eltrk1, cosetrep, eltrk2, FALSE);
		A->element_move(eltrk2, eltrk1, FALSE);
		}
}

void sims::element_unrank_INT(INT rk, INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//longinteger_object a;

	if (f_v) {
		cout << "sims::element_unrank_INT rk=" << rk << endl;
		}
	//a.create(rk);
	//element_unrank(a, Elt);
	path_unrank_INT(rk);
	if (f_v) {
		cout << "sims::element_unrank path=";
		INT_vec_print(cout, path, A->base_len);
		cout << endl;
		}
	element_from_path(Elt, 0);

}

void sims::element_unrank_INT(INT rk, INT *Elt)
{
	//longinteger_object a;
	
	//a.create(rk);
	element_unrank_INT(rk, Elt, 0);
}

INT sims::element_rank_INT(INT *Elt)
{
	longinteger_object a;
	
	element_rank(a, Elt);
	return a.as_INT();
}

INT sims::is_element_of(INT *elt)
{
	INT i, j, bi, jj; //, l;
	
	A->element_move(elt, eltrk1, FALSE);
	for (i = 0; i < A->base_len; i++) {
		bi = A->base[i];
		//l = orbit_len[i];
		
		
		jj = A->element_image_of(bi, eltrk1, FALSE);
		//cout << "at level " << i << ", maps bi = " << bi << " to " << jj << endl;
		j = orbit_inv[i][jj];
		if (j >= orbit_len[i]) {
			return FALSE;
			}
		
		coset_rep_inv(i, j, 0 /* verbose_level */);

		A->element_mult(eltrk1, cosetrep, eltrk2, FALSE);
		A->element_move(eltrk2, eltrk1, FALSE);
		}
	return TRUE;
}

void sims::test_element_rank_unrank()
{
	longinteger_object go, a, b;
	INT i, j, goi;
	INT *elt = NEW_INT(A->elt_size_in_INT);
	
	group_order(go);
	goi = go.as_INT();
	for (i = 0; i < goi; i++) {
		a.create(i);
		element_unrank(a, elt);
		cout << i << " : " << endl;
		A->element_print(elt, cout);
		cout << " : ";
		A->element_print_as_permutation(elt, cout);
		element_rank(b, elt);
		j = b.as_INT();
		cout << " : " << j << endl;
		if (i != j) {
			cout << "error in sims::test_element_rank_unrank()" << endl;
			exit(1);
			}
		}
	FREE_INT(elt);
}

void sims::coset_rep(INT i, INT j, INT verbose_level)
// computes a coset representative in transversal i which maps
// the i-th base point to the point which is in coset j of the i-th basic orbit.
// j is a coset, not a point
// result is in cosetrep
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "sims::coset_rep i=" << i << " j=" << j << endl;
		cout << "orbit[i][0]=" << orbit[i][0] << " orbit[i][j]=" << orbit[i][j] << endl;
		}

	INT depth;
	INT *Path;
	INT *Label;
	INT *gen;
	INT h, a;

	if (f_v) {
		cout << "sims::coset_rep before compute_coset_rep_path" << endl;
		}
	compute_coset_rep_path(i, j, Path, Label, depth, verbose_level - 2);
	if (f_v) {
		cout << "sims::coset_rep after compute_coset_rep_path" << endl;
		cout << "depth=" << depth << endl;
		cout << "Path=";
		INT_vec_print(cout, Path, depth);
		cout << endl;
		cout << "Label=";
		INT_vec_print(cout, Label, depth);
		cout << endl;
		}

	A->element_one(cosetrep, 0);
	for (h = 0; h < depth; h++) {
		if (f_v) {
			cout << h << " / " << depth << " Label[" << h << "]=" << Label[h] << endl;
			}
		gen = gens.ith(Label[h]);
		if (f_vv) {
			cout << "gen=:" << endl;
			A->element_print_quick(gen, cout);
			}
		A->element_mult(cosetrep, gen, cosetrep_tmp, 0);
		A->element_move(cosetrep_tmp, cosetrep, 0);
		a = A->element_image_of(orbit[i][0], cosetrep, 0 /* verbose_level */);
		if (f_vv) {
			cout << "cosetrep*gen=:" << endl;
			A->element_print_quick(cosetrep, cout);
			}
		if (f_v) {
			cout << orbit[i][0] << " -> " << a << endl;
			}
		}
	
	
	FREE_INT(Path);
	FREE_INT(Label);

	if (f_v) {
		cout << "sims::coset_rep i=" << i << " j=" << j << endl;
		}
	a = A->element_image_of(orbit[i][0], cosetrep, 0 /* verbose_level */);
	if (f_v) {
		cout << orbit[i][0] << " -> " << a << endl;
		}
	if (a != orbit[i][j]) {
		cout << "sims::coset_rep a != orbit[i][j]" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		}
}

INT sims::compute_coset_rep_depth(INT i, INT j, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 1);
	INT p, depth, jj;
	
	if (f_v) {
		cout << "sims::compute_coset_rep_depth i=" << i << " j=" << j << endl;
		}
	if (j >= orbit_len[i]) {
		cout << "sims::compute_coset_rep_depth fatal: j >= orbit_len[i]" << endl;
		cout << "sims::compute_coset_rep_depth i=" << i << " j=" << j << endl;
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
		cout << "sims::compute_coset_rep_depth i=" << i << " j=" << j << " depth = " << depth << endl;
		}
	return depth;
}

void sims::compute_coset_rep_path(INT i, INT j, INT *&Path, INT *&Label, INT &depth, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE; // (verbose_level >= 2);
	INT p, d, jj;
	
	if (f_v) {
		cout << "sims::compute_coset_rep_path i=" << i << " j=" << j << endl;
		}


	depth = compute_coset_rep_depth(i, j, verbose_level - 2);

	if (f_vv) {
		cout << "sims::compute_coset_rep_path depth = " << depth << endl;
		}
	

	Path = NEW_INT(depth + 1);
	Label = NEW_INT(depth);

	jj = j;
	d = 0;
	while (TRUE) {

		Path[depth - d] = jj;
		if (f_vv) {
			cout << "Path[" << depth - d << "]=" << jj << endl;
			}

		p = prev[i][jj];
		if (f_vv) {
			cout << "p=" << p << endl;
			}

		if (p == -1) {
			break;
			}
		else {
			if (f_vv) {
				cout << "Label[" << depth - 1 - d << "]=" << label[i][jj] << endl;
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
		cout << "sims::compute_coset_rep_path d != depth" << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "sims::compute_coset_rep_path path = ";
		INT_vec_print(cout, Path, depth + 1);
		cout << endl;
		cout << "sims::compute_coset_rep_path label = ";
		INT_vec_print(cout, Label, depth);
		cout << endl;
		}
	if (f_v) {
		cout << "sims::compute_coset_rep_path i=" << i << " j=" << j << " depth = " << depth << " done" << endl;
		}
}

void sims::coset_rep_inv(INT i, INT j, INT verbose_level)
// computes the inverse element of what coset_rep computes,
// i.e. an element which maps the j-th point in the orbit to the 
// i-th base point.
// j is a coset, not a point
// result is in cosetrep
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT a;
	
	if (f_v) {
		cout << "coset_rep_inv i=" << i << " j=" << j << endl;
		}
	if (f_v) {
		cout << "coset_rep_inv before coset_rep(i,j)" << endl;
		}
	coset_rep(i, j, verbose_level - 2);
	if (f_v) {
		cout << "sims::coset_rep_inv coset_rep(i=" << i << " j=" << j << ") done" << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		}

	a = A->element_image_of(orbit[i][0], cosetrep, 0 /* verbose_level */);
	if (a != orbit[i][j]) {
		cout << "sims::coset_rep_inv a != orbit[i][j]" << endl;
		cout << "i=" << i << " j=" << j << endl;
		cout << "orbit[i][0]=" << orbit[i][0] << " orbit[i][j]=" << orbit[i][j] << endl;
		cout << "a=" << a << endl;
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		exit(1);
		}

	A->element_invert(cosetrep, cosetrep_tmp, 0 /* verbose_level */);
	A->element_move(cosetrep_tmp, cosetrep, 0 /* verbose_level */);
	if (f_vv) {
		cout << "cosetrep^-1=:" << endl;
		A->element_print_quick(cosetrep, cout);
		}
	a = A->element_image_of(orbit[i][j], cosetrep, 0 /* verbose_level */);
	if (f_v) {
		cout << "cosetrep^-1 maps " << orbit[i][j] << " to " << a << endl;
		}
	if (a != orbit[i][0]) {
		cout << "sims::coset_rep_inv a != orbit[i][0]" << endl;
		cout << "cosetrep^-1 maps " << orbit[i][j] << " to " << a << endl;
		exit(1);
		}

	if (f_v) {
		cout << "sims::coset_rep_inv i=" << i << " j=" << j << " done" << endl;
		}
	if (f_vv) {
		cout << "cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
		}
}

void sims::compute_base_orbits(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "sims::compute_base_orbits" << endl;
		}
	if (f_vv) {
		cout << "sims::compute_base_orbits base_len=" << A->base_len << endl;
		}
	for (i = A->base_len - 1; i >= 0; i--) {
		if (FALSE) {
			cout << "sims::compute_base_orbits level " << i << endl;
			}
		compute_base_orbit(i, 0/*verbose_level - 1*/);
		if (f_vv) {
			cout << "sims::compute_base_orbits level " << i 
				<< " base point " << A->base[i] 
				<< " orbit length " << orbit_len[i] << endl;
			}
		}
	if (f_v) {
		cout << "sims::compute_base_orbits done, orbit_len=";
		INT_vec_print(cout, orbit_len, A->base_len);
		cout << endl;
		}
}

void sims::compute_base_orbits_known_length(INT *tl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT i;
	
	if (f_v) {
		cout << "sims::compute_base_orbits_known_length: ";
		INT_vec_print(cout, tl, A->base_len);
		cout << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	for (i = A->base_len - 1; i >= 0; i--) {
		if (f_v) {
			cout << "sims::compute_base_orbits_known_length computing level " << i << endl;
			}
		compute_base_orbit_known_length(i, tl[i], verbose_level);
		if (f_v) {
			cout << "sims::compute_base_orbits_known_length level " << i 
				<< " base point " << A->base[i] 
				<< " orbit length " << orbit_len[i] << " has been computed" << endl;
			}
		if (orbit_len[i] != tl[i]) {
			cout << "sims::compute_base_orbits_known_length orbit_len[i] != tl[i]" << endl;
			exit(1);
			}
		}
	if (f_v) {
		cout << "sims::compute_base_orbits_known_length done" << endl;
		}
}

void sims::extend_base_orbit(INT new_gen_idx, INT lvl, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = FALSE; //(verbose_level >= 3);
	INT i, cur, cur_pt, total, total0, next_pt, next_pt_loc, gen_idx, nbg;
	
	if (f_v) {
		cout << "sims::extend_base_orbit " << lvl << endl;
		}
	cur = 0;
	total = total0 = orbit_len[lvl];
	nbg = nb_gen[lvl];
	while (cur < total) {
		cur_pt = orbit[lvl][cur];
		if (f_vvv) {
			cout << "sims::extend_base_orbit: applying generator to " << cur_pt << endl;
			}
		for (i = 0; i < nbg; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (f_vvv) {
				cout << "sims::extend_base_orbit generator " << gen_idx << " maps " << cur_pt << " to " << next_pt << endl;
				}
			if (next_pt_loc < total)
				continue;
			if (f_vvv) {
				cout << "sims::extend_base_orbit new pt " << next_pt << " reached from " << cur_pt << " under generator " << i << endl;
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
		cout << lvl << "-th base point " << A->base[lvl] << " orbit extended to length " << orbit_len[lvl];
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

void sims::compute_base_orbit(INT lvl, INT verbose_level)
// applies all generators at the given level to compute
// the corresponding basic orbit.
// the generators are the first nb_gen[lvl] in the generator arry
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	INT pt, pt_loc, cur, cur_pt, i, next_pt, next_pt_loc, gen_idx;
	
	pt = A->base[lvl];
	pt_loc = orbit_inv[lvl][pt];
	if (f_v) {
		cout << "sims::compute_base_orbit: computing orbit of " << lvl << "-th base point " << pt << " with " << nb_gen[lvl] << " generators" << endl;
		}
	if (pt_loc > 0) {
		swap_points(lvl, 0, pt_loc);
		}
	cur = 0;
	orbit_len[lvl] = 1;
	while (cur < orbit_len[lvl]) {
		cur_pt = orbit[lvl][cur];
		if (FALSE) {
			cout << "sims::compute_base_orbit applying generator to " << cur_pt << endl;
			}
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (FALSE) {
				cout << "sims::compute_base_orbit generator " << i << " maps " << cur_pt << " to " << next_pt << endl;
				}
			if (next_pt_loc < orbit_len[lvl])
				continue;
			if (FALSE) {
				cout << "new pt " << next_pt << " reached from " << cur_pt << " under generator " << i << endl;
				}
			swap_points(lvl, orbit_len[lvl], next_pt_loc);
			prev[lvl][orbit_len[lvl]] = cur_pt;
			label[lvl][orbit_len[lvl]] = gen_idx;
			orbit_len[lvl]++;
			if (FALSE) {
				cout << "sims::compute_base_orbit cur = " << cur << endl;
				cout << "sims::compute_base_orbit orbit_len[lvl] = " << orbit_len[lvl] << endl;
				//print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	if (f_v) {
		cout << "sims::compute_base_orbit finished, " << lvl << "-th base orbit of length " << orbit_len[lvl] << endl;
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

void sims::compute_base_orbit_known_length(INT lvl, INT target_length, INT verbose_level)
// applies all generators at the given level to compute
// the corresponding basic orbit.
// the generators are the first nb_gen[lvl] in the generator arry
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = FALSE;
	//INT f_vvv = (verbose_level >= 3);
	//INT f_v10 = FALSE; // (verbose_level >= 10);
	INT pt, pt_loc, cur, cur_pt, i, next_pt, next_pt_loc, gen_idx;
	
	pt = A->base[lvl];
	pt_loc = orbit_inv[lvl][pt];
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length: computing orbit of " << lvl 
			<< "-th base point " << pt 
			<< " target_length = " << target_length << " nb_gens=" << nb_gen[lvl] << endl;
		}
	if (target_length > 1000000) {
		f_vv = TRUE;
		}
	if (FALSE) {
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			cout << "sims::compute_base_orbit_known_length generator " << i << ":" << endl;
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
			if (cur % ((1 << 21) - 1) == 0) {
				cout << "sims::compute_base_orbit_known_length lvl=" << lvl << " cur=" << cur << " orbit_len[lvl]=" << orbit_len[lvl] << " target_length=" << target_length << endl;
				}
			}
		if (FALSE) {
			cout << "sims::compute_base_orbit_known_length applying " << nb_gen[lvl] << " generators to " << cur_pt << " orbit_len[lvl]=" << orbit_len[lvl] << " target_length=" << target_length << endl;
			}
		for (i = 0; i < nb_gen[lvl]; i++) {
			gen_idx = gen_perm[i];
			next_pt = get_image(cur_pt, gen_idx);
			next_pt_loc = orbit_inv[lvl][next_pt];
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length generator " << i << " maps " << cur_pt << " to " << next_pt << endl;
				}
			if (next_pt_loc < orbit_len[lvl])
				continue;
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length new pt " << next_pt << " reached from " << cur_pt << " under generator " << i << endl;
				}
			swap_points(lvl, orbit_len[lvl], next_pt_loc);
			prev[lvl][orbit_len[lvl]] = cur_pt;
			label[lvl][orbit_len[lvl]] = gen_idx;
			orbit_len[lvl]++;
			if (FALSE) {
				cout << "sims::compute_base_orbit_known_length cur = " << cur << endl;
				cout << "sims::compute_base_orbit_known_length orbit_len[lvl] = " << orbit_len[lvl] << endl;
				//print_orbit(cur, total - 1);
				}
			}
		cur++;
		}
	if (f_v) {
		cout << "sims::compute_base_orbit_known_length finished, " << lvl << "-th base orbit of length " << orbit_len[lvl] << endl;
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

void sims::extract_strong_generators_in_order(vector_ge &SG, INT *tl, INT verbose_level)
{
	INT i, nbg, nbg1, j, k = 0, gen_idx;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "extract_strong_generators_in_order" << endl;
		cout << "A->base_len=" << A->base_len << endl;
		//if (f_vv) {
			//print(0);
			//}
		}
	
	SG.init(A);
	SG.allocate(gens.len);
	for (i = A->base_len - 1; i >= 0; i--) {
		nbg = nb_gen[i];
		nbg1 = nb_gen[i + 1];
		//cout << "i=" << i << " nbg1=" << nbg1 << " nbg=" << nbg << endl;
		for (j = nbg1; j < nbg; j++) {
			gen_idx = gen_perm[j];
			//cout << "gen_idx=" << gen_idx << endl;
			if (f_vv) {
				cout << "the " << k << "-th strong generator is generator " 
					<< j << " at position " << gen_idx << endl;
				
				//cout << "moving generator " << gen_idx << " to position " << k << endl;
				//cout << "before:" << endl;
				//A->element_print(gens.ith(gen_idx), cout);
				//cout << endl;
				}
			A->element_move(gens.ith(gen_idx), SG.ith(k), FALSE);
			if (f_vv) {
				cout << "the " << k << "-th strong generator is generator " 
					<< j << " at position " << gen_idx << endl;
				//A->element_print(SG.ith(k), cout);
				cout << endl;
				}
			k++;
			}
		tl[i] = orbit_len[i];
		}
	if (k < SG.len) {
		cout << "warning: sims::extract_strong_generators_in_order" << endl;
		cout << "k = " << k << endl;
		cout << "SG.len = " << SG.len << endl;
		SG.len = k;
		}
	if (f_v) {
		cout << "sims::extract_strong_generators_in_order done, found " << SG.len << " strong generators" << endl;
		}
	if (FALSE) {
		cout << "transversal length:" << endl;
		for (i = 0; i < A->base_len; i++) {
			cout << tl[i];
			if (i < A->base_len - 1)
				cout << ", ";
			}
		cout << endl;
		cout << "strong generators are:" << endl;
		SG.print(cout);
		cout << endl;
		}
}

void sims::transitive_extension(schreier &O, vector_ge &SG, INT *tl, INT verbose_level)
{
	transitive_extension_tolerant(O, SG, tl, FALSE, verbose_level);
}

INT sims::transitive_extension_tolerant(schreier &O, vector_ge &SG, INT *tl, 
	INT f_tolerant, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go, ol, ego, cur_ego, rgo, rem;
	INT orbit_len, j;
	longinteger_domain D;

	orbit_len = O.orbit_len[0];
	if (f_v) {
		cout << "sims::transitive_extension_tolerant computing transitive extension" << endl;
		cout << "f_tolerant=" << f_tolerant << endl;
		}
	group_order(go);
	ol.create(orbit_len);
	D.mult(go, ol, ego);
	if (f_v) {
		cout << "sims::transitive_extension_tolerant group order " << go << ", orbit length " << orbit_len << ", new group order " << ego << endl;
		}
	group_order(cur_ego);
	
	//if (f_vv) {
		//print(0);
		//}
	while (D.compare_unsigned(cur_ego, ego) != 0) {
		
		// we do not enter the while loop if orbit_len is 1, hence the following makes sense:
		// we want non trivial generators, hence we want j non zero.
		if (D.compare_unsigned(cur_ego, ego) > 0) {
			cout << "sims::transitive_extension_tolerant fatal: group order overshoots target" << endl;
			cout << "current group order = " << cur_ego << endl;
			cout << "target group order = " << ego << endl;
			if (f_tolerant) {
				cout << "we are tolerant, so we return FALSE" << endl;
				return FALSE;
				}
			cout << "we are not tolerant, so we exit" << endl;
			exit(1);
			}
	
		while (TRUE) {
			j = random_integer(orbit_len);
			if (j)
				break;
			}	
		
		O.coset_rep(j);
		
		random_element(Elt2, verbose_level - 1);
		
		A->element_mult(O.cosetrep, Elt2, Elt3, FALSE);
		
		if (f_vv) {
			cout << "sims::transitive_extension_tolerant choosing random coset " << j << ", random element ";
			print_set(cout, A->base_len, path);
			cout << endl;
			//A->element_print(Elt3, cout);
			//cout << endl;
			}
		
		if (!strip_and_add(Elt3, Elt1 /* residue */, 0/*verbose_level - 1*/)) {
			continue;
			}


		group_order(cur_ego);
		if (f_v) {
			cout << "sims::transitive_extension_tolerant found an extension of order " << cur_ego << " of " << ego 
				<< " with " << gens.len << " strong generators" << endl;
			D.integral_division(ego, cur_ego, rgo, rem, 0);
			cout << "remaining factor: " << rgo << " remainder " << rem << endl;
			}
		
		
		}
	if (f_v) {
		cout << "sims::transitive_extension_tolerant extracting strong generators" << endl;
		}
	extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	return TRUE;
}

void sims::transitive_extension_using_coset_representatives_extract_generators(
	INT *coset_reps, INT nb_cosets, 
	vector_ge &SG, INT *tl, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::transitive_extension_using_coset_representatives_extract_generators" << endl;
		}
	transitive_extension_using_coset_representatives(
		coset_reps, nb_cosets, 
		verbose_level);
	extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_representatives_extract_generators done" << endl;
		}
}


void sims::transitive_extension_using_coset_representatives(
	INT *coset_reps, INT nb_cosets, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go, ol, ego, cur_ego, rgo, rem;
	INT orbit_len, j;
	longinteger_domain D;

	orbit_len = nb_cosets;
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_representatives computing transitive extension" << endl;
		}
	group_order(go);
	ol.create(orbit_len);
	D.mult(go, ol, ego);
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_representatives group order " << go << ", orbit length " << orbit_len << ", new group order " << ego << endl;
		}
	group_order(cur_ego);
	
	//if (f_vv) {
		//print(0);
		//}
	while (D.compare_unsigned(cur_ego, ego) != 0) {
		
		// we do not enter the while loop if orbit_len is 1, hence the following makes sense:
		// we want non trivial generators, hence we want j non zero.
		if (D.compare_unsigned(cur_ego, ego) > 0) {
			cout << "sims::transitive_extension_using_coset_representatives fatal: group order overshoots target" << endl;
			cout << "current group order = " << cur_ego << endl;
			cout << "target group order = " << ego << endl;
			cout << "we are not tolerant, so we exit" << endl;
			exit(1);
			}
	
		while (TRUE) {
			j = random_integer(orbit_len);
			if (j) {
				break;
				}
			}	
				
		random_element(Elt2, verbose_level - 1);
		
		A->element_mult(coset_reps + j * A->elt_size_in_INT, Elt2, Elt3, 0);
		
		if (f_vv) {
			cout << "sims::transitive_extension_using_coset_representatives choosing random coset " << j << ", random element ";
			INT_vec_print(cout, path, A->base_len);
			cout << endl;
			//A->element_print(Elt3, cout);
			//cout << endl;
			}
		
		if (!strip_and_add(Elt3, Elt1 /* residue */, 0/*verbose_level - 1*/)) {
			continue;
			}


		group_order(cur_ego);
		if (f_v) {
			cout << "sims::transitive_extension_using_coset_representatives found an extension of order " << cur_ego << " of " << ego 
				<< " with " << gens.len << " strong generators" << endl;
			D.integral_division(ego, cur_ego, rgo, rem, 0);
			cout << "remaining factor: " << rgo << " remainder " << rem << endl;
			}
		
		
		}
	if (f_v) {
		cout << "sims::transitive_extension_using_coset_representatives done" << endl;
		}
}

void sims::transitive_extension_using_generators(
	INT *Elt_gens, INT nb_gens, INT subgroup_index, 
	vector_ge &SG, INT *tl, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go, ol, ego, cur_ego, rgo, rem;
	INT j;
	longinteger_domain D;

	if (f_v) {
		cout << "sims::transitive_extension_using_generators computing transitive extension" << endl;
		}
	group_order(go);
	ol.create(subgroup_index);
	D.mult(go, ol, ego);
	if (f_v) {
		cout << "sims::transitive_extension_using_generators group order " << go << ", subgroup_index " << subgroup_index << ", new group order " << ego << endl;
		}
	group_order(cur_ego);
	
	//if (f_vv) {
		//print(0);
		//}
	while (D.compare_unsigned(cur_ego, ego) != 0) {
		
		// we do not enter the while loop if orbit_len is 1, hence the following makes sense:
		// we want non trivial generators, hence we want j non zero.
		if (D.compare_unsigned(cur_ego, ego) > 0) {
			cout << "sims::transitive_extension_using_generators fatal: group order overshoots target" << endl;
			cout << "current group order = " << cur_ego << endl;
			cout << "target group order = " << ego << endl;
			cout << "we are not tolerant, so we exit" << endl;
			exit(1);
			}
	
		j = random_integer(nb_gens);
				
		random_element(Elt2, verbose_level - 1);
		
		A->element_mult(Elt_gens + j * A->elt_size_in_INT, Elt2, Elt3, 0);
		
		if (f_vv) {
			cout << "sims::transitive_extension_using_generators choosing random coset " << j << ", random element ";
			INT_vec_print(cout, path, A->base_len);
			cout << endl;
			//A->element_print(Elt3, cout);
			//cout << endl;
			}
		
		if (!strip_and_add(Elt3, Elt1 /* residue */, verbose_level - 1)) {
			continue;
			}


		group_order(cur_ego);
		if (f_v) {
			cout << "sims::transitive_extension_using_generators found an extension of order " << cur_ego << " of " << ego 
				<< " with " << gens.len << " strong generators" << endl;
			D.integral_division(ego, cur_ego, rgo, rem, 0);
			cout << "remaining factor: " << rgo << " remainder " << rem << endl;
			}
		
		
		}
	if (f_v) {
		cout << "sims::transitive_extension_using_generators extracting strong generators" << endl;
		}
	extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	//return TRUE;
}


void sims::point_stabilizer_stabchain_with_action(action *A2, sims &S, INT pt, INT verbose_level)
// first computes the orbit of the point pt in action A2 under the generators 
// that are stored at present (using a temporary schreier object),
// then sifts random schreier generators into S
{
	schreier O;
	longinteger_object go, stab_order, cur_stab_order, rgo, rem;
	INT orbit_len, r, cnt = 0, image; // d
	longinteger_domain D;

	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action computing stabilizer of point " 
			<< pt << " in action " << A2->label 
			<< " verbose_level=" << verbose_level << endl;
		cout << "internal action: " << A->label << endl;
		cout << "verbose_level=" << verbose_level << endl;
		}
	group_order(go);
	if (f_v) {
		cout << "group order = " << go << endl;
		}
	
	O.init(A2);

	O.init_generators(gens);
	
	if (f_vvv && A2->degree < 150) {
		O.print_generators();
		O.print_generators_with_permutations();
		INT j;
		for (j = 0; j < O.gens.len; j++) {
			cout << "generator " << j << ":" << endl;
			//A->element_print(gens.ith(j), cout);
			//A->element_print_quick(gens.ith(j), cout);
			A->element_print_as_permutation(O.gens.ith(j), cout);
			cout << endl;
			}
		}

	if (f_vv) {
		cout << "sims::point_stabilizer_stabchain_with_action computing point orbit" << endl;
		}
	O.compute_point_orbit(pt, 0/*verbose_level - 1*/);
	
	
	orbit_len = O.orbit_len[0];
	if (f_vv) {
		cout << "sims::point_stabilizer_stabchain_with_action found orbit of length " << orbit_len << endl;
		}
	
	if (f_vvv && A2->degree < 150) {
		O.print(cout);
		}
	D.integral_division_by_INT(go, orbit_len, stab_order, r);
	if (r != 0) {
		cout << "sims::point_stabilizer_stabchain_with_action orbit_len does not divide group order" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action group_order = " << go << " orbit_len = " << orbit_len << " target stab_order = " << stab_order << endl;
		}
	if (stab_order.is_one()) {
		if (f_v) {
			cout << "sims::point_stabilizer_stabchain_with_action stabilizer is trivial, finished" << endl;
			}
		S.init(A);
		S.init_trivial_group(verbose_level - 1);
#if 0
		for (i = 0; i < A->base_len; i++) {
			tl[i] = 1;
			}
#endif
		return;
		}


	vector_ge stab_gens(A);
	
	//stab_gens.append(O.schreier_gen);
	
	//sims S;
	//INT drop_out_level, image;
	INT *p_schreier_gen;
	
	S.init(A);
	S.init_generators(stab_gens, verbose_level - 2);
	S.compute_base_orbits(verbose_level - 1);
	if (FALSE) {
		cout << "sims::point_stabilizer_stabchain_with_action generators:" << endl;
		S.gens.print(cout);
		}
	
	S.group_order(cur_stab_order);
	if (f_vv) {
		cout << "sims::point_stabilizer_stabchain_with_action before the loop, stabilizer has order " << cur_stab_order << " of " << stab_order << endl;
		cout << "sims::point_stabilizer_stabchain_with_action creating the stabilizer using random generators" << endl;
		}
	
	while (D.compare_unsigned(cur_stab_order, stab_order) != 0) {
	

		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action loop iteration " << cnt << endl;
			}

		if (cnt % 2 || nb_gen[0] == 0) {
			if (f_vv) {
				cout << "sims::point_stabilizer_stabchain_with_action creating random generator no " << cnt + 1 << " using the Schreier vector" << endl;
				}
			O.non_trivial_random_schreier_generator(A2, 0 /*verbose_level - 5*/);
			p_schreier_gen = O.schreier_gen;
			}
		else {
			if (f_vv) {
				cout << "sims::point_stabilizer_stabchain_with_action creating random generator no " << cnt + 1 << " using the Sims chain" << endl;
				}
			S.random_schreier_generator(0 /*verbose_level - 5*/);
			p_schreier_gen = S.schreier_gen;
			}
		cnt++;
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action random generator no " << cnt << endl;
			A->element_print_quick(p_schreier_gen, cout);
			cout << endl;
			cout << "sims::point_stabilizer_stabchain_with_action random generator no " << cnt << " as permutation in natural action:" << endl;
			A->element_print_as_permutation(p_schreier_gen, cout);
			cout << endl;
			cout << "sims::point_stabilizer_stabchain_with_action random generator no " << cnt << " as permutation in chosen action:" << endl;
			A2->element_print_as_permutation(p_schreier_gen, cout);
			cout << endl;
			}
		image = A2->element_image_of(pt, p_schreier_gen, 0 /* verbose_level */);
		if (image != pt) {
			cout << "sims::point_stabilizer_stabchain_with_action image is not equal to pt" << endl;
			cout << "pt=" << pt << endl;
			cout << "image=" << image << endl;
			exit(1);
			}
		if (f_vvv) {
			A->element_print_quick(p_schreier_gen, cout);
			if (A2->degree < 150) {
				A2->element_print_as_permutation(p_schreier_gen, cout);
				cout << endl;
				}
			}
		
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action random generator no " << cnt << " before strip_and_add" << endl;
			}
		if (!S.strip_and_add(p_schreier_gen, Elt1 /* residue */, verbose_level - 3)) {
			if (f_vvv) {
				cout << "sims::point_stabilizer_stabchain_with_action strip_and_add returns FALSE" << endl;
				}
			//continue;
			}
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action random generator no " << cnt << " before strip_and_add" << endl;
			}

		S.group_order(cur_stab_order);
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action group order " << go << endl;
			cout << "orbit length " << orbit_len << endl;
			cout << "current stab_order = " << cur_stab_order << " / " << stab_order
				<< " with " << S.gens.len << " strong generators" << endl;
			}

		INT cmp;
		
		cmp = D.compare_unsigned(cur_stab_order, stab_order);
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action compare yields " << cmp << endl;
			}
		if (cmp > 0) {	
			cout << "sims::point_stabilizer_stabchain_with_action overshooting the target group order" << endl;
			cout << "current stab_order = " << cur_stab_order << " / " << stab_order << endl;
			exit(1);
			}
		D.integral_division(stab_order, cur_stab_order, rgo, rem, 0);
		if (f_vv) {
			cout << "sims::point_stabilizer_stabchain_with_action remaining factor: " << rgo << " remainder " << rem << endl;
			}

		if (D.compare_unsigned(cur_stab_order, stab_order) == 1) {
			cout << "sims::point_stabilizer_stabchain_with_action group order " << go << endl;
			cout << "orbit length " << orbit_len << endl;
			cout << "current stab_order = " << cur_stab_order << " / " << stab_order
				<< " with " << S.gens.len << " strong generators" << endl;
			D.integral_division(stab_order, cur_stab_order, rgo, rem, 0);
			cout << "remaining factor: " << rgo << " remainder " << rem << endl;
			cout << "the current stabilizer is:" << endl;
			S.print_transversals();
			cout << "sims::point_stabilizer_stabchain_with_action computing stabilizer of point " << pt << " in action " << A2->label << " verbose_level=" << verbose_level << endl;
			cout << "internal action: " << A->label << endl;
			cout << "The orbit of point " << pt << " is:" << endl;
			O.print_and_list_orbits(cout);
			//O.print_tables(cout, TRUE /* f_with_cosetrep */);
			cout << "sims::point_stabilizer_stabchain_with_action cur_stab_order > stab_order, error" << endl;
			exit(1);
			}
		
		}
	if (f_v) {
		cout << "sims::point_stabilizer_stabchain_with_action found a stabilizer of order " << cur_stab_order << " of " << stab_order 
			<< " with " << S.gens.len << " strong generators" << endl;
		}
}

void sims::point_stabilizer(vector_ge &SG, INT *tl, INT pt, INT verbose_level)
// computes strong generating set for the stabilizer of point pt
{
	sims S;
	
	point_stabilizer_stabchain_with_action(A, S, pt, verbose_level);
	S.extract_strong_generators_in_order(SG, tl, verbose_level - 2);
}

void sims::point_stabilizer_with_action(action *A2, vector_ge &SG, INT *tl, INT pt, INT verbose_level)
// computes strong generating set for the stabilizer of point pt in action A2
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	sims S;
	
	if (f_v) {
		cout << "sims::point_stabilizer_with_action pt=" << pt << endl;
		cout << "sims::point_stabilizer_with_action action = " << A2->label << endl;
		cout << "sims::point_stabilizer_with_action internal action = " << A->label << endl;
		}
	if (f_v) {
		cout << "sims::point_stabilizer_with_action before point_stabilizer_stabchain_with_action" << endl;
		}
	point_stabilizer_stabchain_with_action(A2, S, pt, verbose_level);
	if (f_v) {
		cout << "sims::point_stabilizer_with_action after point_stabilizer_stabchain_with_action" << endl;
		}
	if (f_v) {
		cout << "sims::point_stabilizer_with_action before extract_strong_generators_in_order" << endl;
		}
	S.extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	if (f_v) {
		cout << "sims::point_stabilizer_with_action done" << endl;
		}
}

INT sims::strip_and_add(INT *elt, INT *residue, INT verbose_level)
// returns TRUE if something was added, FALSE if element stripped through
{
	INT drop_out_level, image;
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	//INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "sims::strip_and_add" << endl;
		}
	if (strip(elt, residue, drop_out_level, image, 0 /*verbose_level*/)) {
		if (f_v) {
			cout << "element strips to the identity, finished" << endl;
			}
		return FALSE;
		}
	if (f_v) {
		cout << "after strip, drop_out_level = " << drop_out_level << " image = " << image << endl;
		}
	if (FALSE) {
		cout << "residue = " << endl;
		A->element_print_quick(residue, cout);
		//A->element_print_as_permutation(residue, cout);
		cout << endl;
		}
		
	if (f_v) {
		cout << "sims::strip_and_add calling add_generator_at_level " << drop_out_level << endl;
		}
	add_generator_at_level(residue, drop_out_level, verbose_level);
	//add_generator_at_level_only(residue, drop_out_level, verbose_level);
	// !!! this was add_generator_at_level previously
	
	if (FALSE) {
		cout << "new set of generators:" << endl;
		gens.print(cout);
		gens.print_as_permutation(cout);
		}
	if (f_v) {
		cout << "sims::strip_and_add finished, new group order is ";
		print_group_order(cout);
		cout << endl;
		}
	return TRUE;
}

INT sims::strip(INT *elt, INT *residue, INT &drop_out_level, INT &image, INT verbose_level)
// returns TRUE if the element sifts through
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, bi, j, j_coset;
	
	if (f_v) {
		cout << "sims::strip" << endl;
		cout << "A->base_len=" << A->base_len << endl;
		}
	if (f_vv) {
		A->element_print_quick(elt, cout);
		cout << endl;
		}
	A->element_move(elt, strip1, FALSE);
	for (i = 0; i < A->base_len; i++) {
		if (f_v) {
			cout << "sims::strip i=" << i << endl;
			//A->element_print(strip1, cout);
			//cout << endl;
			}
		bi = A->base[i];
		if (f_vv) {
			cout << "computing image of " << i << "-th base element " << bi << endl;
			}
		j = A->element_image_of(bi, strip1, verbose_level - 2);
		if (f_v) {
			cout << "sims::strip level " << i << " base point " << bi << " gets mapped to " << j << endl;
			}
		j_coset = orbit_inv[i][j];
		if (f_v) {
			cout << "sims::strip j_coset " << j_coset << endl;
			}
		if (j_coset >= orbit_len[i]) {
			if (f_v) {
				cout << "sims::strip not in the orbit, dropping out" << endl;
				}
			image = j;
			drop_out_level = i;
			A->element_move(strip1, residue, FALSE);
			if (f_v) {
				cout << "sims::strip returns FALSE, drop_out_level=" << drop_out_level << endl;
				}
			return FALSE;
			}
		else {
			if (f_v) {
				cout << "sims::strip computing representative of coset " << j_coset << endl;
				}
			coset_rep_inv(i, j_coset, verbose_level);
			if (FALSE) {
				cout << "sims::strip representative of coset " << j_coset << " is " << endl;
				A->element_print(cosetrep, cout);
				cout << endl;
				}
			if (FALSE) {
				cout << "sims::strip before element_mult, strip1=" << endl;
				A->element_print(strip1, cout);
				cout << endl;
				}
			if (FALSE) {
				cout << "sims::strip before element_mult, cosetrep=" << endl;
				A->element_print(cosetrep, cout);
				cout << endl;
				}
			A->element_mult(strip1, cosetrep, strip2, 0 /*verboe_level*/);
			if (FALSE) {
				cout << "sims::strip before element_move" << endl;
				}
			A->element_move(strip2, strip1, FALSE);
			if (FALSE) {
				cout << "sims::strip after dividing off, we have strip1= " << endl;
				A->element_print(strip1, cout);
				cout << endl;
				}
			}
		}
	A->element_move(strip1, residue, FALSE);
	if (f_v) {
		cout << "sims::strip returns TRUE" << endl;
		}
	return TRUE;
}

void sims::add_generator_at_level(INT *elt, INT lvl, INT verbose_level)
// add the generator to the array of generators and then extends the 
// basic orbits 0,..,lvl using extend_base_orbit
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	INT i;
	
	if (f_v) {
		cout << "adding generator at level " << lvl << endl;
		if (FALSE) {
			A->element_print_quick(elt, cout);
			cout << endl;
			}
		}
	add_generator(elt, verbose_level);
	if (f_vvv) {
		print_generator_depth_and_perm();
		}
	for (i = lvl; i >= 0; i--) {
		if (f_v) {
			cout << "sims::add_generator_at_level " << lvl << " calling extend_base_orbit " << i << endl;
			}
		extend_base_orbit(gens.len - 1, i, 0/*verbose_level - 1*/);
		}
}

void sims::add_generator_at_level_only(INT *elt, INT lvl, INT verbose_level)
// add the generator to the array of generators and then extends the 
// basic orbit lvl using extend_base_orbit
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
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

void sims::random_schreier_generator(INT verbose_level)
// computes random Schreier generator into schreier_gen
{
	INT i, r1, r2, pt, pt1, pt1b, pt2, pt2_coset;
	INT *gen, gen_idx, nbg;
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "sims:random_schreier_generator" << endl;
		}
	if (nb_gen[0] == 0) {
		if (f_vv) {
			cout << "sims::random_schreier_generator() nb_gen[0] == 0, choosing the identity" << endl;
			}
		A->element_one(schreier_gen, 0 /* verbose_level */);
		goto finish;
		}
	while (TRUE) {
		if (f_vv) {
			cout << "sims::random_schreier_generator interation" << endl;
			}
		// get a random level:
		i = random_integer(A->base_len);
		if (f_vv) {
			cout << "sims::random_schreier_generator i=" << i << endl;
			}
		pt = orbit[i][0];
		if (f_vv) {
			cout << "sims::random_schreier_generator pt=orbit[i][0]=" << pt << endl;
			}
		if (f_vv) {
			cout << "sims::random_schreier_generator orbit_len[i]=" << orbit_len[i] << endl;
			}
	
		// get a random coset:
		r1 = random_integer(orbit_len[i]);
		if (f_vv) {
			cout << "sims::random_schreier_generator r1=" << r1 << endl;
			}
		pt1 = orbit[i][r1];
		if (f_vv) {
			cout << "sims::random_schreier_generator pt1=" << pt1 << endl;
			}
		if (f_vv) {
			cout << "sims::random_schreier_generator random level " << i << ", base pt " << pt 
				<< ", random coset " << r1 << " of an orbit of length " 
				<< orbit_len[i] << ", image pt " << pt1 << endl;
			}
		if (f_vv) {
			cout << "sims::random_schreier_generator before coset_rep" << endl;
			}
	
		coset_rep(i, r1, 0/*verbose_level*/);
		if (f_vv) {
			cout << "sims::random_schreier_generator after coset_rep" << endl;
			cout << "checking image of pt=" << pt << endl;
			}
		// coset rep now in cosetrep
		pt1b = A->element_image_of(pt, cosetrep, 0/*verbose_level*/);

		if (f_vvv) {
			cout << "sims::random_schreier_generator coset rep maps " << pt << " to " << pt1b << endl;
			}
		if (pt1b != pt1) {
			cout << "sims::random_schreier_generator() fatal: not the same point" << endl;
			cout << "action " << A->label << endl;
			cout << "level i=" << i << endl;
			cout << "coset r1=" << r1 << endl;
			cout << "base pt=" << pt << endl;
			cout << "image pt1=" << pt1 << endl;
			cout << "image  under cosetrep pt1b=" << pt1b << endl;
			cout << "basic orbit " << i << ":" << endl;
			print_basic_orbit(i);
			pt1b = A->element_image_of(pt, cosetrep, FALSE);
			cout << "cosetrep:" << endl;
			A->element_print(cosetrep, cout);
			cout << endl;
			coset_rep(i, r1, 10 /* verbose_level */);
			exit(1);
			}
		
		// get a random generator:
		nbg = nb_gen[i];
		if (nbg == 0) {
			continue;
			}
		r2 = random_integer(nbg);
		gen_idx = gen_perm[r2];
		gen = gens.ith(gen_idx);
		if (f_vvv) {
			cout << "sims::random_schreier_generator random level " << i << ", random coset " << r1 
				<< ", random generator " << r2 << " of " << nb_gen[i] << endl;
			}
		break;
		}
	
	if (f_vv) {
		cout << "sims::random_schreier_generator after the while loop" << endl;
		cout << "cosetrep:" << endl;
		A->element_print(cosetrep, cout);
		cout << "maps " << pt << " to " << pt1 << " : checking: " << pt << " -> ";
		pt1b = A->element_image_of(pt, cosetrep, FALSE);
		cout << pt1b;
		cout << endl;
		}
	
	if (f_vv) {
		cout << "sims::random_schreier_generator after the while loop" << endl;
		cout << "gen=" << endl;
		A->element_print(gen, cout);
		}
	A->element_mult(cosetrep, gen, schreier_gen1, 0);
	if (f_vv) {
		cout << "sims::random_schreier_generator after the while loop" << endl;
		cout << "cosetrep * gen=" << endl;
		A->element_print(schreier_gen1, cout);
		}
	pt2 = A->element_image_of(pt, schreier_gen1, 0);


	if (f_vv) {
		cout << "cosetrep * gen maps " << pt << " to " << pt2 << endl;
		}


	//cout << "maps " << pt << " to " << pt2 << endl;
	pt2_coset = orbit_inv[i][pt2];
	
	if (pt2_coset >= orbit_len[i]) {


		A->element_move(schreier_gen1, schreier_gen, 0);
		cout << "schreier generator is " << endl;
		A->element_print(schreier_gen, cout);
		cout << endl;
		
		if (f_v) {
			cout << "sims::random_schreier_generator done early" << endl;
			}
		return;

#if 0
		cout << "sims::random_schreier_generator() fatal: new pt " << pt2 << " reached from " << pt << " under generator " << i << endl;
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
	
	coset_rep_inv(i, pt2_coset, 0 /*verbose_level - 2*/);
	// coset rep now in cosetrep
	if (f_vv) {
		cout << "cosetrep(" << pt2 << ")^-1=" << endl;
		A->element_print(cosetrep, cout);
		}
	INT pt2b;

	pt2b = A->element_image_of(pt2, cosetrep, FALSE);
	if (f_vv) {
		cout << "cosetrep(" << pt2 << ")^-1 maps " << pt2 << " to " << pt2b << endl;
		}
	if (pt2b != pt) {
		cout << "sims::random_schreier_generator pt2b != pt" << endl;
		cout << "cosetrep(" << pt2 << ")^-1 maps " << pt2 << " to " << pt2b << endl;
		cout << "pt=" << pt << endl;
		exit(1);
		}
	
	A->element_mult(schreier_gen1, cosetrep, schreier_gen, 0);
	if (f_vv) {
		cout << "sims::random_schreier_generator after the while loop" << endl;
		cout << "cosetrep * gen * cosetrep(" << pt2 << ")^-1=" << endl;
		A->element_print(schreier_gen1, cout);
		}

	if (A->element_image_of(pt, schreier_gen, 0) != pt) {
		INT im;
		
		cout << "sims::random_schreier_generator() fatal: schreier generator does not stabilize pt" << endl;
		cout << "pt=" << pt << endl;
		cout << "schreier generator:" << endl;
		A->element_print(schreier_gen, cout);
		im = A->element_image_of(pt, schreier_gen, TRUE);
		cout << "im = " << im << endl;
		exit(1);
		}

finish:
	if (f_vv) {
		cout << "sims::random_schreier_generator random Schreier generator:" << endl;
		A->element_print(schreier_gen, cout);
		cout << endl;
		}
	if (f_v) {
		cout << "sims::random_schreier_generator done" << endl;
		}
}

void sims::build_up_group_random_process_no_kernel(sims *old_G, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go, go1;
	sims K;
	
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel" << endl;
		}
	old_G->group_order(go);
	if (f_v) {
		cout << "target group order = " << go << endl;
		}
	K.init(A);
	K.init_trivial_group(verbose_level - 1);
	K.group_order(go1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel kernel group order " << go1 << endl;
		}
	init_trivial_group(verbose_level - 1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel before build_up_group_random_process" << endl;
		}
	build_up_group_random_process(&K, old_G, go, 
		FALSE /* f_override_chose_next_base_point */,
		NULL /* choose_next_base_point_method */, 
		verbose_level - 1);
	if (f_v) {
		cout << "sims::build_up_group_random_process_no_kernel after build_up_group_random_process" << endl;
		}
}

void sims::extend_group_random_process_no_kernel(sims *extending_by_G, longinteger_object &target_go, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
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
	
	K.init(A);
	K.init_trivial_group(verbose_level - 1);
	build_up_group_random_process(&K, extending_by_G, target_go, 
		FALSE /* f_override_chose_next_base_point */,
		NULL /* choose_next_base_point_method */, 
		verbose_level + 3);
}

void sims::conjugate(action *A, sims *old_G, INT *Elt, 
	INT f_overshooting_OK, INT verbose_level)
// Elt * g * Elt^{-1} where g is in old_G
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 4);
	//INT f_vvv = (verbose_level >= 3);
	longinteger_domain D;
	longinteger_object go, target_go, quo, rem;
	INT *Elt2, *Elt3, *Elt4, *Elt5;
	INT cnt, drop_out_level, image, f_added, c;
	
	if (f_v) {
		cout << "sims::conjugate f_overshooting_OK=" << f_overshooting_OK << endl;
		}
	if (f_v) {
		cout << "action = " << A->label << endl;
		}
	if (FALSE) {
		cout << "transporter = " << endl;
		A->print(cout, Elt);
		}
	
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	Elt4 = NEW_INT(A->elt_size_in_INT);
	Elt5 = NEW_INT(A->elt_size_in_INT);
	init(A);
	init_trivial_group(verbose_level - 1);
	group_order(go);
	old_G->group_order(target_go);
	A->invert(Elt, Elt2);
	cnt = 0;
	while (TRUE) {
	
		if (f_vv) {
			cout << "iteration " << cnt << endl;
			}
		if (cnt > 500) {
			cout << "sims::conjugate() cnt > 1000, something seems to be wrong" << endl;
			exit(1);
			}
		if ((cnt % 2) == 0) {
			if (f_vv) {
				cout << "choosing random schreier generator" << endl;
				}
			random_schreier_generator(verbose_level - 3);
			A->element_move(schreier_gen, A->Elt1, FALSE);
			if (FALSE) {
				cout << "random element chosen:" << endl;
				A->element_print(A->Elt1, cout);
				cout << endl;
				}
			A->move(A->Elt1, Elt4);
			}
		else if ((cnt % 2) == 1){
			if (f_vv) {
				cout << "choosing random element in the group by which we extend" << endl;
				}
			old_G->random_element(A->Elt1, verbose_level - 1);
			if (FALSE) {
				cout << "random element chosen, path = ";
				INT_vec_print(cout, old_G->path, old_G->A->base_len);
				cout << endl;
				}
			if (FALSE) {
				A->element_print(A->Elt1, cout);
				cout << endl;
				}
			A->mult(Elt, A->Elt1, Elt3);
			A->mult(Elt3, Elt2, Elt4);
			if (f_vv) {
				cout << "conjugated" << endl;
				}
			if (FALSE) {
				A->element_print(Elt4, cout);
				cout << endl;
				}
			}
		if (strip(Elt4, A->Elt2, drop_out_level, image, verbose_level - 3)) {
			if (f_vv) {
				cout << "element strips through, residue = " << endl;
				if (FALSE) {
					A->element_print_quick(A->Elt2, cout);
					cout << endl;
					}
				}
			f_added = FALSE;
			}
		else {
			f_added = TRUE;
			if (f_vv) {
				cout << "element needs to be inserted at level = " 
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					A->element_print(A->Elt2, cout);
					cout  << endl;
					}
				}
			add_generator_at_level(A->Elt2, drop_out_level, verbose_level - 3);
			}
		
		group_order(go);
		if ((f_v && f_added) || f_vv) {
			cout << "new group order is " << go << endl;
			}
		if (f_vv) {
			print_transversal_lengths();
			}
		c = D.compare(target_go, go);
		cnt++;
		if (c == 0) {
			if (f_v) {
				cout << "reached the full group after " << cnt << " iterations" << endl;
				}
			break;
			}
		if (c < 0) {
			if (TRUE) {
				cout << "sims::conjugate overshooting the expected group after " << cnt << " iterations" << endl;
				cout << "new group order is " << go << " target_go=" << target_go << endl;
				}
			if (f_overshooting_OK)
				break;
			else
				exit(1);
			}
		}
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(Elt4);
	FREE_INT(Elt5);
	if (f_v) {
		cout << "sims::conjugate done" << endl;
		}
}

INT sims::test_if_in_set_stabilizer(action *A, INT *set, INT size, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go, a;
	INT goi, i, ret;
	INT *Elt1;
	
	if (f_v) {
		cout << "sims::test_if_in_set_stabilizer action = " << A->label << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "testing group of order " << goi << endl;
		}
	ret = TRUE;
	for (i = 0; i < goi; i++) {
		a.create(i);
		element_unrank(a, Elt1);
		if (A->check_if_in_set_stabilizer(Elt1, size, set, verbose_level)) {
			if (f_vv) {
				cout << "element " << i << " strips through, residue = " << endl;
				}
			}
		else {
			cout << "element " << i << " does not stabilize the set" << endl;
			A->element_print(Elt1, cout);
			cout << endl;
			ret = FALSE;
			break;
			}
		}
	FREE_INT(Elt1);
	return ret;
}

INT sims::test_if_subgroup(sims *old_G, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	longinteger_object go, a, b;
	INT goi, i, ret, drop_out_level, image;
	INT *Elt1, *Elt2;
	
	if (f_v) {
		cout << "sims::test_if_subgroup" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	old_G->group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "testing group of order " << goi << endl;
		}
	ret = TRUE;
	for (i = 0; i < goi; i++) {
		a.create(i);
		old_G->element_unrank(a, Elt1);
		if (strip(Elt1, Elt2, drop_out_level, image, verbose_level - 3)) {
			a.create(i);
			old_G->element_unrank(a, Elt1);
			element_rank(b, Elt1);
			if (f_vv) {
				cout << "element " << i << " strips through, rank " << b << endl;
				}
			}
		else {
			cout << "element " << i << " is not contained" << endl;
			old_G->element_unrank(a, Elt1);
			A->element_print(Elt1, cout);
			cout << endl;
			ret = FALSE;
			break;
			}
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	return ret;
}

void sims::build_up_group_random_process(sims *K, 
	sims *old_G, 
	longinteger_object &target_go, 
	INT f_override_choose_next_base_point,
	INT (*choose_next_base_point_method)(action *A, INT *Elt, INT verbose_level), 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 1);
	//INT f_vvv = (verbose_level >= 1);
	//INT f_vvvv = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 6);
	INT f_v4 = (verbose_level >= 7);
	longinteger_domain D;
	longinteger_object go, G_order, K_order, KG_order, quo, rem;
	INT drop_out_level, image, cnt, b, c, /*f_added,*/ old_base_len;
	action *GA;
	action *KA;
	
	if (f_v) {
		cout << "sims::build_up_group_random_process" << endl;
		}
	GA = A;
	KA = K->A;
	
	group_order(G_order);
	K->group_order(K_order);
	D.mult(G_order, K_order, KG_order);
	if (f_v) {
		cout << "sims::build_up_group_random_process: current group order is " << G_order << " target " << target_go << endl;
		cout << "the old_G action " << old_G->A->label << " has base_length = " << old_G->A->base_len 
			<< " and degree " << old_G->A->degree << endl;
		cout << "the kernel action " << KA->label << " has base_length = " << KA->base_len 
			<< " and degree " << KA->degree << endl;
		cout << "the image action has base_length = " << GA->base_len 
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
			cout << "sims::build_up_group_random_process iteration " << cnt << endl;
			}
		if (cnt > 1000) {
			cout << "sims::build_up_group_random_process cnt > 1000, something seems to be wrong" << endl;
			test_if_subgroup(old_G, 2);
			exit(1);
			}
		if (f_v4) {
			old_G->A->print_base();
			old_G->print_orbit_len();
			}
		if ((cnt % 2) == 0) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process: choosing random schreier generator" << endl;
				}
			random_schreier_generator(verbose_level - 5);
			A->element_move(schreier_gen, GA->Elt1, 0);
			if (f_v4) {
				cout << "sims::build_up_group_random_process: random element chosen:" << endl;
				A->element_print_quick(GA->Elt1, cout);
				cout << endl;
				}
			}
		else if ((cnt % 2) == 1){
			if (f_vv) {
				cout << "sims::build_up_group_random_process: choosing random element in the group by which we extend" << endl;
				}
			old_G->random_element(GA->Elt1, verbose_level - 5);
			if (f_vv) {
				cout << "sims::build_up_group_random_process: random element chosen, path = ";
				INT_vec_print(cout, old_G->path, old_G->A->base_len);
				cout << endl;
				}
			if (f_v4) {
				GA->element_print_quick(GA->Elt1, cout);
				cout << endl;
				}
			}
		if (f_v4) {
			cout << "sims::build_up_group_random_process: calling strip:" << endl;
			}
		if (strip(GA->Elt1, GA->Elt2, drop_out_level, image, verbose_level - 5)) {
			if (f_vv) {
				cout << "sims::build_up_group_random_process: element strips through" << endl;
				if (f_v4) {
					cout << "sims::build_up_group_random_process: residue = " << endl;
					GA->element_print_quick(GA->Elt2, cout);
					cout << endl;
					}
				}
			//f_added = FALSE;
			if (!GA->element_is_one(GA->Elt2, 0)) {
				if (f_vvv) {
					cout << "sims::build_up_group_random_process: the residue is not trivial, we need to choose another base point" << endl;
					}
				if (f_override_choose_next_base_point) {
					b = (*choose_next_base_point_method)(GA, GA->Elt2, verbose_level - 5);
					}
				else {
					b = choose_next_base_point_default_method(GA, GA->Elt2, verbose_level - 5);
					}

				if (f_vv) {
					cout << "sims::build_up_group_random_process: next suggested base point is " << b << endl;
					}
				if (b == -1) {
					if (f_vv) {
						cout << "sims::build_up_group_random_process: cannot find next base point" << endl;
						}
					if (K->strip(GA->Elt2, GA->Elt3, drop_out_level, image, 0/*verbose_level - 3*/)) {
						if (f_vv) {
							cout << "sims::build_up_group_random_process: element strips through kernel" << endl;
							if (f_v4) {
								cout << "sims::build_up_group_random_process: residue = " << endl;
								KA->element_print_quick(GA->Elt3, cout);
								cout << endl;
								K->print(FALSE);
								K->print_basic_orbits();
								cout << "sims::build_up_group_random_process: residue" << endl;
								KA->element_print_image_of_set(GA->Elt3, KA->base_len, KA->base);
								cout << "sims::build_up_group_random_process: Elt2" << endl;
								KA->element_print_image_of_set(GA->Elt2, KA->base_len, KA->base);
								}
							}
						if (!KA->element_is_one(GA->Elt3, FALSE)) {
							cout << "sims::build_up_group_random_process: element strips through kernel, residue = " << endl;
							cout << "but the element is not the identity, something is wrong" << endl;
							GA->element_print(GA->Elt3, cout);
							cout << endl;

							cout << "sims::build_up_group_random_process: current group order is " << G_order << " target " << target_go << endl;
							cout << "the old_G action " << old_G->A->label << " has base_length = " << old_G->A->base_len 
								<< " and degree " << old_G->A->degree << endl;
							cout << "the kernel action " << KA->label << " has base_length = " << KA->base_len 
								<< " and degree " << KA->degree << endl;
							cout << "the image action has base_length = " << GA->base_len 
								<< " and degree " << GA->degree << endl;
							cout << "current action " << GA->label << endl;
							cout << "current group order = " << G_order << endl;
							cout << "current kernel order = " << K_order << endl;
							cout << "together = " << KG_order << endl;
							cout << "target_go = " << target_go << endl;

							exit(1);
							}
						}
					K->add_generator_at_level(GA->Elt3, drop_out_level, 0/*verbose_level - 3*/);
					if (f_vvv) {
						cout << "sims::build_up_group_random_process: the residue has been added as kernel generator at level " << drop_out_level << endl;
						}
					//f_added = TRUE;
					}
				else {
					if (f_vvv) {
						cout << "sims::build_up_group_random_process: choosing new base point " << b << endl;
						}
					old_base_len = GA->base_len;
					GA->reallocate_base(b);
					if (f_vvv) {
						//cout << "after reallocate_base 1" << endl;
						}
					reallocate_base(old_base_len, verbose_level - 1);
					if (f_vvv) {
						//cout << "after reallocate_base 2" << endl;
						}
					if (f_vv) {
						cout << "sims::build_up_group_random_process: new base point " << b 
							<< " chosen, new base has length " << GA->base_len << endl;
						cout << "sims::build_up_group_random_process: calling add_generator_at_level" << endl;
						}
					add_generator_at_level(GA->Elt2, GA->base_len - 1, 0/*verbose_level - 3*/);
					if (f_vv) {
						cout << "sims::build_up_group_random_process: the residue has been added at level " << GA->base_len - 1 << endl;
						}
					} // if b
				} // if ! element is one
			else {
				if (f_vv) {
					cout << "sims::build_up_group_random_process: the residue is trivial" << endl;
					}
				}
			if (f_vv) {
				cout << "sims::build_up_group_random_process: before closure_group" << endl;
				}
			//closure_group(10, verbose_level);
			closure_group(10, 0 /*verbose_level - 2*/);
			if (f_vv) {
				cout << "sims::build_up_group_random_process: after closure_group" << endl;
				}
			}
		else {
			//f_added = TRUE;
			if (f_vv) {
				cout << "sims::build_up_group_random_process: element needs to be inserted at level = " 
					<< drop_out_level << " with image " << image << endl;
				if (FALSE) {
					GA->element_print(GA->Elt2, cout);
					cout  << endl;
					}
				}
			add_generator_at_level(GA->Elt2, drop_out_level, 0/*verbose_level - 3*/);
			}
		
		if (f_vv) {
			cout << "sims::build_up_group_random_process: computing group order G" << endl;
			}
		group_order(G_order);
		if (f_vv) {
			cout << "sims::build_up_group_random_process:  G_order=" << G_order << endl;
			}
		K->group_order(K_order);
		if (f_vv) {
			cout << "sims::build_up_group_random_process:  K_order=" << K_order << endl;
			}
		//cout << "K tl: ";
		//INT_vec_print(cout, K->orbit_len, K->A->base_len);
		//cout << endl;
		//cout << "K action " << K->A->label << endl;
		D.mult(G_order, K_order, KG_order);
		if (f_v /* (f_v && f_added) || f_vv */) {
			cout << "sims::build_up_group_random_process: new group order is " << KG_order 
				<< " = " << G_order << " * " << K_order << endl;
			}
		if (f_vv) {
			print_transversal_lengths();
			}
		if (FALSE) {
			cout << "sims::build_up_group_random_process before D.compare" << endl;
			}
		c = D.compare(target_go, KG_order);
		if (FALSE) {
			cout << "sims::build_up_group_random_process after D.compare c=" << c << " cnt=" << cnt << endl;
			}
		cnt++;
		if (c == 0) {
			if (f_v) {
				cout << "sims::build_up_group_random_process: reached the full group after " << cnt << " iterations" << endl;
				}
			break;
			}
		if (c < 0) {
			if (TRUE) {
				cout << "sims::build_up_group_random_process overshooting the expected group after " << cnt << " iterations" << endl;
				cout << "new group order is " << KG_order 
					<< " = " << G_order << " * " << K_order << endl;
				}
			//break;
			exit(1);
			}
		}
	if (f_vv) {
		cout << "sims::build_up_group_random_process finished: found a group of order " << KG_order 
			<< " = " << G_order << " * " << K_order << endl;
		if (f_vvv) {
			cout << "the new action has base_length = " << GA->base_len 
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

void sims::build_up_group_from_generators(sims *K, vector_ge *gens, 
	INT f_target_go, longinteger_object *target_go, 
	INT f_override_choose_next_base_point,
	INT (*choose_next_base_point_method)(action *A, INT *Elt, INT verbose_level), 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_vvv = (verbose_level >= 3);
	//INT f_vvvv = (verbose_level >= 4);
	longinteger_domain D;
	longinteger_object G_order, K_order, KG_order;
	INT drop_out_level, image, f_added, j, level, base_point, b, old_base_len;
	action *GA;
	action *KA;
	vector_ge subset_of_gens;
	
	GA = A;
	KA = K->A;
	
	
	if (f_v) {
		cout << "sims::build_up_group_from_generators base: ";
		INT_vec_print(cout, GA->base, GA->base_len);
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
	for (level = GA->base_len - 1; level >= 0; level--) {
		base_point = GA->base[level];
		if (f_vv) {
			cout << "level " << level << " base point " << base_point << endl;
			}
		GA->find_strong_generators_at_level(GA->base_len, GA->base, level, 
			*gens, subset_of_gens, verbose_level - 3);
		
		{
		schreier O;
	
		if (f_v) {
			cout << "calling O.init()" << endl;
			}
		
		O.init(GA);
		
		if (f_v) {
			cout << "calling O.init_generators()" << endl;
			}
		O.init_generators(subset_of_gens);
		
		if (f_vvv) {
			cout << "generators in schreier" << endl;
			O.print_generators();
			}

		if (f_vv) {
			cout << "computing orbit of point " << base_point << endl;
			}
		O.compute_point_orbit(base_point, 0);
		if (f_vv) {
			cout << "point " << base_point << " lies in an orbit of size " << O.orbit_len[0] << endl;
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
			if (strip(O.cosetrep, GA->Elt2 /* residue */, drop_out_level, image, 0 /*verbose_level - 1*/)) {
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
					cout << "if it is the identity element, that's OK," << endl;
					cout << "otherwise please add another base point," << endl;
					cout << "a point which is moved by the residue" << endl;
					GA->element_print(GA->Elt2, cout);
					}
				if (!GA->element_is_one(GA->Elt2, FALSE)) {
					if (f_vvv) {
						cout << "the residue is not trivial, we need to chose another base point" << endl;
						}
					if (f_override_choose_next_base_point) {
						b = (*choose_next_base_point_method)(GA, GA->Elt2, verbose_level);
						}
					else {
						b = choose_next_base_point_default_method(GA, GA->Elt2, verbose_level);
						}
					if (b == -1) {
						if (f_vv) {
							cout << "sims::build_up_group_from_generators: cannot find next base point" << endl;
							}
						if (K->strip(GA->Elt2, GA->Elt3, drop_out_level, image, verbose_level - 3)) {
							if (f_vv) {
								cout << "element strips through kernel, residue = " << endl;
								if (f_vv) {
									KA->element_print(GA->Elt3, cout);
									cout << endl;
									}
								K->print(FALSE);
								K->print_basic_orbits();
								cout << "residue" << endl;
								KA->element_print_image_of_set(GA->Elt3, KA->base_len, KA->base);
								cout << "Elt2" << endl;
								KA->element_print_image_of_set(GA->Elt2, KA->base_len, KA->base);
								}
							if (!KA->element_is_one(GA->Elt3, FALSE)) {
								cout << "but the element is not the identity, something is wrong" << endl;
								GA->element_print(GA->Elt3, cout);
								cout << endl;
								exit(1);
								}
							}
						K->add_generator_at_level(GA->Elt3, drop_out_level, verbose_level - 3);
						if (f_vv) {
							cout << "the residue has been added as kernel generator at level " << drop_out_level << endl;
							}
						f_added = TRUE;
						}
					else {
						if (f_vv) {
							cout << "action::induced_action: choosing new base point " << b << endl;
							}
						old_base_len = GA->base_len;
						GA->reallocate_base(b);
						if (f_vv) {
							//cout << "after reallocate_base 1" << endl;
							}
						reallocate_base(old_base_len, verbose_level - 1);
						if (f_vv) {
							//cout << "after reallocate_base 2" << endl;
							}
						if (f_v) {
							cout << "new base point " << b 
								<< " chosen, new base has length " << GA->base_len << endl;
							cout << "calling add_generator_at_level" << endl;
							}
						add_generator_at_level(GA->Elt2, GA->base_len - 1, verbose_level - 3);
						if (f_vv) {
							cout << "the residue has been added at level " << GA->base_len - 1 << endl;
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
				add_generator_at_level(GA->Elt2, drop_out_level, 0 /*verbose_level - 1*/);
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
		INT c, cnt;
		
		cnt = 0;
		while (TRUE) {
			group_order(G_order);
			K->group_order(K_order);
			D.mult(G_order, K_order, KG_order);
		
			c = D.compare(*target_go, KG_order);
			cnt++;
			if (c == 0) {
				if (f_v) {
					cout << "reached the full group after " << cnt << " iterations" << endl;
					}
				break;
				}
			if (c < 0) {
				if (TRUE) {
					cout << "sims::build_up_group_from_generators() overshooting the expected group after " << cnt << " iterations" << endl;
					cout << "new group order is " << KG_order 
						<< " = " << G_order << " * " << K_order << endl;
					}
				//break;
				exit(1);
				}
			if (cnt > 10000) {
				cout << "sims::build_up_group_from_generators() after " << cnt << " iterations, we seem to be having problems reaching the target group order" << endl;
				cout << "group order = " << KG_order << endl;
				cout << "target group order = " << *target_go << endl;
				exit(1);
				}

			if (f_vv) {
				cout << "sims::build_up_group_from_generators() calling closure group" << endl;
				}
			closure_group(10, verbose_level - 2);

			}
		}
		
	if (f_v) {
		cout << "sims::build_up_group_from_generators finished: found a group of order " << KG_order 
			<< " = " << G_order << " * " << K_order << endl;
		cout << "the new action has base_length = " << GA->base_len 
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

INT sims::closure_group(INT nb_times, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	//INT f_vv = (verbose_level >= 1);
	//INT f_vvv = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT f_v3 = (verbose_level >= 6);
	INT i, f_extended = FALSE;
	INT *Elt1;
	INT *Elt2;
	longinteger_object old_go, go, go1;
	
	if (f_v) {
		cout << "sims::closure_group" << endl;
		}
	if (f_vv) {
		print_transversal_lengths();
		cout << "verbose_level=" << verbose_level << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	
	group_order(old_go);
	if (f_vv) {
		cout << "sims::closure_group for group of order " << old_go << endl;
		}
	if (old_go.is_one()) {
		FREE_INT(Elt1);
		FREE_INT(Elt2);
		if (f_v) {
			cout << "sims::closure_group finishing with FALSE because the old group order is one" << endl;
			}
		return FALSE;
		}
	group_order(go);
	for (i = 0; i < nb_times; i++) {
		if (f_vv) {
			cout << "sims::closure_group loop " << i << " / " << nb_times << " go=" << go << endl;
			}
		if (f_v3) {
			cout << "sims::closure_group before random_schreier_generator" << endl;
			}
		random_schreier_generator(verbose_level /*- 4*/);
		if (f_v3) {
			cout << "sims::closure_group after random_schreier_generator" << endl;
			}
		// now in S->schreier_gen
		group_order(go);
		A->element_move(schreier_gen, Elt2, 0);
		if (strip_and_add(Elt2, Elt1 /* residue */, verbose_level - 3)) {
			group_order(go1);
			if (f_vv) {
				cout << "closure_group: iteration " << i << " the group has been extended, old order " 
					<< go << " new group order " << go1 << endl;
				print_transversal_lengths();
				}
			if (f_v3) {
				cout << "original element:" << endl;
				A->element_print_quick(schreier_gen, cout);
				cout << "new generator:" << endl;
				A->element_print_quick(Elt2, cout);
				}
			f_extended = TRUE;
			}
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_extended) {
		if (f_v) {
			cout << "sims::closure_group group order extended from " << old_go << " to " << go1 << endl;
			if (f_vv) {
				print_transversal_lengths();
				}
			}
		}
	else {
		if (f_vv) {
			cout << "sims::closure_group group order stays at " << old_go << endl;
			}
		}
	return f_extended;
}

void sims::write_all_group_elements(BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt;
	//BYTE *elt;
	longinteger_object go;
	FILE *fp;
	INT i;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	//elt = NEW_BYTE(A->coded_elt_size_in_char);
	group_order(go);
	
	fp = fopen(fname, "wb");
	for (i = 0; i < go.as_INT(); i++) {
		element_unrank_INT(i, Elt);
		A->element_write_file_fp(Elt, fp, 0/* verbose_level*/);
		}
	fclose(fp);
	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
	FREE_INT(Elt);
	//FREE_BYTE(elt);
}

void sims::print_all_group_elements_to_file(BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt;
	longinteger_object go;
	INT i;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	
	{
	ofstream fp(fname);
	for (i = 0; i < go.as_INT(); i++) {
		element_unrank_INT(i, Elt);
		fp << "Element " << setw(5) << i << " / " << go.as_INT() << endl;
		A->element_print(Elt, fp);
		fp << endl;
		}
	}
	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
	FREE_INT(Elt);
}

void sims::print_all_group_elements()
{
	INT *Elt;
	longinteger_object go;
	INT i;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	
	for (i = 0; i < go.as_INT(); i++) {
		element_unrank_INT(i, Elt);
		cout << "Element " << setw(5) << i << " / " << go.as_INT() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	FREE_INT(Elt);
}

void sims::print_all_group_elements_as_permutations()
{
	INT *Elt;
	longinteger_object go;
	INT i;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	
	for (i = 0; i < go.as_INT(); i++) {
		element_unrank_INT(i, Elt);
		cout << "Element " << setw(5) << i << " / " << go.as_INT() << ":" << endl;
		//A->element_print(Elt, cout);
		//cout << endl;
		A->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	FREE_INT(Elt);
}

void sims::print_all_group_elements_as_permutations_in_special_action(action *A_special)
{
	INT *Elt;
	longinteger_object go;
	INT i;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	
	for (i = 0; i < go.as_INT(); i++) {
		element_unrank_INT(i, Elt);
		cout << "Element " << setw(5) << i << " / " << go.as_INT() << ":" << endl;
		A->element_print(Elt, cout);
		cout << endl;
		A_special->element_print_as_permutation(Elt, cout);
		cout << endl;
		}
	FREE_INT(Elt);
}

void sims::print_all_transversal_elements()
{
	INT *Elt;
	longinteger_object go;
	INT i, j, ii;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	
	for (i = A->base_len - 1; i >= 0; i--) {
		for (j = 0; j < A->transversal_length[i]; j++) {
			if (j == 0 && i < A->base_len - 1) {
				// skip the identity in the upper transversals
				continue;
				}
			for (ii = 0; ii < A->base_len; ii++) {
				path[ii] = 0;
				}
			path[i] = j;
			element_from_path(Elt, 0 /* verbose_level */);
			for (ii = 0; ii < A->base_len; ii++) {
				cout << setw(5) << path[ii] << " ";
				}
			cout << endl;
			A->element_print(Elt, cout);
			cout << endl;
			A->element_print_as_permutation(Elt, cout);
			cout << endl;
			}
		}
	FREE_INT(Elt);
}

void sims::element_as_permutation(action *A_special, INT elt_rk, INT *perm, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "sims::element_as_permutation" << endl;
		}
	INT *Elt;
	
	Elt = NEW_INT(A->elt_size_in_INT);
	
	element_unrank_INT(elt_rk, Elt);

	A_special->element_as_permutation(Elt, perm, 0);
	

	FREE_INT(Elt);
	if (f_v) {
		cout << "sims::element_as_permutation done" << endl;
		}
}

void sims::table_of_group_elements_in_data_form(INT *&Table, INT &len, INT &sz, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt;
	longinteger_object go;
	INT i;
	
	if (f_v) {
		cout << "sims::table_of_group_elements_in_data_form" << endl;
		}
	Elt = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	len = go.as_INT();
	sz = A->make_element_size;
	Table = NEW_INT(len * sz);
	for (i = 0; i < len; i++) {
		element_unrank_INT(i, Elt);
		INT_vec_copy(Elt, Table + i * sz, sz);
		}
	FREE_INT(Elt);
	if (f_v) {
		cout << "sims::table_of_group_elements_in_data_form done" << endl;
		}
}

void sims::regular_representation(INT *Elt, INT *perm, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	INT goi, i, j;
	INT *Elt1;
	INT *Elt2;
	
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	goi = go.as_INT();
	for (i = 0; i < goi; i++) {
		element_unrank_INT(i, Elt1);
		A->mult(Elt1, Elt, Elt2);
		j = element_rank_INT(Elt2);
		perm[i] = j;
		}
	if (f_v) {
		cout << "sims::regular_representation of" << endl;
		A->print(cout, Elt);
		cout << endl;
		cout << "is:" << endl;
		perm_print(cout, perm, goi);
		cout << endl;
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
}

void sims::element_ranks_subgroup(sims *subgroup, INT *element_ranks, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	INT goi;
	INT i, j;
	INT *Elt1;

	subgroup->group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "sims::element_ranks_subgroup subgroup of order " << goi << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	for (i = 0; i < goi; i++) {
		subgroup->element_unrank_INT(i, Elt1);
		j = element_rank_INT(Elt1);
		element_ranks[i] = j;
		}
	FREE_INT(Elt1);
}

void sims::center(vector_ge &gens, INT *center_element_ranks, INT &nb_elements, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	vector_ge gens_inv;
	INT goi, i, j, k, len;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	
	len = gens.len;
	gens_inv.init(A);
	gens_inv.allocate(len);
	for (i = 0; i < len; i++) {
		A->invert(gens.ith(i), gens_inv.ith(i));
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	nb_elements = 0;
	group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "sims::center computing the center of a group of order " << goi << endl;
		}
	for (i = 0; i < goi; i++) {
		element_unrank_INT(i, Elt1);
		for (j = 0; j < len; j++) {
			A->mult(gens_inv.ith(j), Elt1, Elt2);
			A->mult(Elt2, gens.ith(j), Elt3);
			k = element_rank_INT(Elt3);
			if (k != i)
				break;
			}
		if (j == len) {
			center_element_ranks[nb_elements++] = i;
			}
		}
	if (f_v) {
		cout << "sims::center center is of order " << nb_elements << ":" << endl;
		INT_vec_print(cout, center_element_ranks, nb_elements);
		cout << endl;
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
}

void sims::all_cosets(INT *subset, INT size, INT *all_cosets, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	INT goi, i, j, k, nb_cosets, cnt;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *f_taken;
	
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "sims::all_cosets" << endl;
		cout << "action " << A->label << endl;
		cout << "subset of order " << size << endl;
		cout << "group of order " << goi << endl;
		}
	nb_cosets = goi / size;
	if (size * nb_cosets != goi) {
		cout << "sims::all_cosets size * nb_cosets != goi" << endl;
		}
	f_taken = NEW_INT(goi);
	for (i = 0; i < goi; i++) {
		f_taken[i] = FALSE;
		}
	cnt = 0;
	for (i = 0; i < goi; i++) {
		if (f_taken[i])
			continue;
		element_unrank_INT(i, Elt1);
		for (j = 0; j < size; j++) {
			element_unrank_INT(subset[j], Elt2);
			A->mult(Elt2, Elt1, Elt3); // we need right cosets!!!
			k = element_rank_INT(Elt3);
			if (f_taken[k]) {
				cout << "sims::all_cosets error: f_taken[k]" << endl;
				exit(1);
				}
			all_cosets[cnt * size + j] = k;
			f_taken[k] = TRUE;
			}
		cnt++;
		}
	if (cnt != nb_cosets) {
		cout << "sims::all_cosets cnt != nb_cosets" << endl;
		exit(1);
		}
	if (f_v) {
		cout << "sims::all_cosets finished" << endl;
		print_integer_matrix_width(cout, all_cosets, nb_cosets, size, size, 2);
		cout << endl;
		}
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(f_taken);
}

void sims::find_standard_generators_INT(INT ord_a, INT ord_b, INT ord_ab, INT &a, INT &b, INT &nb_trials, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT nb_trials1, o;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	
	if (f_v) {
		cout << "sims::find_standard_generators_INT ord_a=" << ord_a << " ord_b=" << ord_b << " ord_ab=" << ord_ab << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	while (TRUE) {
		a = find_element_of_given_order_INT(ord_a, nb_trials1, verbose_level - 1);
		nb_trials += nb_trials1;
		b = find_element_of_given_order_INT(ord_b, nb_trials1, verbose_level - 1);
		nb_trials += nb_trials1;
		element_unrank_INT(a, Elt1);
		element_unrank_INT(b, Elt2);
		A->mult(Elt1, Elt2, Elt3);
		o = A->element_order(Elt3);
		if (o == ord_ab)
			break;
		}

	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	if (f_v) {
		cout << "found a=" << a << " b=" << b << " nb_trials=" << nb_trials << endl;
		}
}

INT sims::find_element_of_given_order_INT(INT ord, INT &nb_trials, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	INT o, d, a, goi;
	INT *Elt1;

	nb_trials = 0;
	group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT" << endl;
		cout << "action " << A->label << endl;
		cout << "group of order " << goi << endl;
		cout << "looking for an element of order " << ord << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	while (TRUE) {
		nb_trials++;
		if (f_v) {
			cout << "sims::find_element_of_given_order_INT before random_element" << endl;
			}
		random_element(Elt1, 0 /*verbose_level - 1*/);
		if (f_v) {
			cout << "sims::find_element_of_given_order_INT after random_element" << endl;
			}
		o = A->element_order(Elt1);
		if (f_v) {
			cout << "sims::find_element_of_given_order_INT random_element has order " << o << endl;
			}
		if (o % ord == 0) {
			if (f_v) {
				cout << "sims::find_element_of_given_order_INT the order is divisible by " << ord << " which is good" << endl;
				}
			break;
			}
		}
	d = o / ord;
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT raising to the power " << d << endl;
		}
	A->element_power_INT_in_place(Elt1, d, verbose_level - 1);
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT after raising to the power " << d << endl;
		}
	a = element_rank_INT(Elt1);
	FREE_INT(Elt1);
	return a;
}

INT sims::find_element_of_given_order_INT(INT *Elt, INT ord, INT &nb_trials, INT max_trials, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 4);
	longinteger_object go;
	INT o, d, goi;
	INT *Elt1;

	nb_trials = 0;
	group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT" << endl;
		cout << "action " << A->label << endl;
		cout << "group of order " << goi << endl;
		cout << "looking for an element of order " << ord << endl;
		cout << "max_trials = " << max_trials << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	o = 0;
	while (nb_trials < max_trials) {
		nb_trials++;
		if (f_vv) {
			cout << "sims::find_element_of_given_order_INT before random_element" << endl;
			}
		random_element(Elt1, 0 /*verbose_level - 1*/);
		if (f_vv) {
			cout << "sims::find_element_of_given_order_INT after random_element" << endl;
			}
		o = A->element_order(Elt1);
		if (f_vv) {
			cout << "sims::find_element_of_given_order_INT random_element has order " << o << endl;
			}
		if (o % ord == 0) {
			if (f_vv) {
				cout << "sims::find_element_of_given_order_INT the order is divisible by " << ord << " which is good" << endl;
				}
			break;
			}
		}
	if (nb_trials == max_trials) {
		FREE_INT(Elt1);
		if (f_v) {
			cout << "sims::find_element_of_given_order_INT unsuccessful" << endl;
			}
		return FALSE;
		}
	d = o / ord;
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT raising to the power " << d << endl;
		}
	A->element_power_INT_in_place(Elt1, d, verbose_level - 1);
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT after raising to the power " << d << endl;
		}
	A->element_move(Elt1, Elt, 0);
	FREE_INT(Elt1);
	if (f_v) {
		cout << "sims::find_element_of_given_order_INT done" << endl;
		}
	return TRUE;
}

void sims::find_element_of_prime_power_order(INT p, INT *Elt, INT &e, INT &nb_trials, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	longinteger_object go;
	INT o;

	nb_trials = 0;
	group_order(go);
	if (f_v) {
		cout << "sims::find_element_of_prime_power_order" << endl;
		cout << "action " << A->label << endl;
		cout << "group of order " << go << endl;
		cout << "prime " << p << endl;
		}
	while (TRUE) {
		nb_trials++;
		random_element(Elt, 0 /*verbose_level - 1*/);
		o = A->element_order(Elt);
		e = 0;
		while (o % p == 0) {
			e++;
			o = o / p;
			}
		if (e) {
			break;
			}
		}
	A->element_power_INT_in_place(Elt, o, verbose_level - 1);
	if (f_v) {
		cout << "sims::find_element_of_prime_power_order done, e=" << e << " nb_trials=" << nb_trials << endl;
		}
}

void sims::save_list_of_elements(BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1;
	FILE *f2;
	INT goi, i;
	longinteger_object go;

	group_order(go);
	goi = go.as_INT();
	if (f_v) {
		cout << "sims::save_list_of_elements(): saving " << goi << " elements to file " << fname << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	f2 = fopen(fname, "wb");
	for (i = 0; i < goi; i++) {
		element_unrank_INT(i, Elt1);
		//cout << "element " << i << ":" << endl;
		//A->element_print(Elt1, cout);
		A->element_write_file_fp(Elt1, f2, 0/* verbose_level*/);
		//A->element_print_as_permutation(Elt1, cout);
		//AA.print_as_permutation(cout, Elt1);
		//cout << endl;
		}
	fclose(f2);
	FREE_INT(Elt1);
	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
}

void sims::read_list_of_elements(action *A, BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1, *Elt2;
	FILE *f2;
	INT goi, i;

	goi = file_size(fname) / A->coded_elt_size_in_char;
	if (f_v) {
		cout << "sims::read_list_of_elements(): reading " << goi << " elements from file " << fname << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	f2 = fopen(fname, "rb");
	
	init(A);
	init_trivial_group(verbose_level - 1);
	for (i = 0; i < goi; i++) {
		A->element_read_file_fp(Elt1, f2, 0/* verbose_level*/);
		//cout << "element " << i << ":" << endl;
		//A->element_print(Elt1, cout);
		strip_and_add(Elt1, Elt2, verbose_level - 1);
		}
	fclose(f2);
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_v) {
		cout << "read file " << fname << " of size " << file_size(fname) << endl;
		}
}

void sims::evaluate_word_INT(INT word_len, INT *word, INT *Elt, INT verbose_level)
{
	INT *Elt1;
	INT *Elt2;
	INT i, j;
	

	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	A->one(Elt);
	for (i = 0; i < word_len; i++) {
		j = word[i];
		element_unrank_INT(j, Elt1);
		A->mult(Elt1, Elt, Elt2);
		A->move(Elt2, Elt);
		}
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
}

void sims::write_sgs(const BYTE *fname, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt;
	BYTE *elt;
	longinteger_object go;
	INT i;
	vector_ge SG;
	INT *tl;
	
	if (f_v) {
		cout << "sims::write_sgs fname=" << fname << endl;
		}
	Elt = NEW_INT(A->elt_size_in_INT);
	elt = NEW_BYTE(A->coded_elt_size_in_char);
	group_order(go);

	tl = NEW_INT(A->base_len);
	extract_strong_generators_in_order(SG, tl, 0 /*verbose_level*/);


	{
	ofstream fp(fname);
	fp << "# group of order " << go << endl;
	fp << "# action: " << A->label << endl;
	fp << "# action: " << A->label_tex << endl;
	fp << "# base length: " << endl;
	fp << A->base_len << endl;
	fp << "# base: " << endl;
	for (i = 0; i < A->base_len; i++) {
		fp << setw(5) << A->base[i] << ", ";
		}
	fp << endl;
	fp << "# transversal lengths: " << endl;
	for (i = 0; i < A->base_len; i++) {
		fp << setw(5) << tl[i] << ", ";
		}
	fp << endl;
	fp << "# number of generators: " << endl;
	fp << SG.len << endl;
	fp << "# make_element_size: " << endl;
	fp << A->make_element_size << endl;
	fp << "# generators are:" << endl;
	for (i = 0; i < SG.len; i++) {
		A->element_print_for_make_element(SG.ith(i), fp);
		fp << endl;
		}
	}
	if (f_v) {
		cout << "written file " << fname << " of size " << file_size(fname) << endl;
		}
	FREE_INT(Elt);
	FREE_BYTE(elt);
}



#define READ_SGS_BUF_SIZE 50000


void sims::read_sgs(const BYTE *fname, vector_ge *SG, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT *Elt;
	BYTE *elt;
	longinteger_object go;
	INT i, j;
	BYTE label[1000];
	BYTE *buf;
	INT base_length;
	INT *base1; // [base_length]
	INT *tl; //[base_length]
	INT nb_gens;
	INT make_element_size1;
	INT *data; // [nb_gens * make_element_size1]
	//BYTE str[1000];

	if (f_v) {
		cout << "sims::read_sgs fname=" << fname << endl;
		}

	buf = NEW_BYTE(READ_SGS_BUF_SIZE);
	Elt = NEW_INT(A->elt_size_in_INT);
	elt = NEW_BYTE(A->coded_elt_size_in_char);
	//group_order(go);
	
	if (f_v) {
		cout << "reading file " << fname << " of size " << file_size(fname) << endl;
		}
	{
		ifstream fp(fname);
		//string str;
		BYTE *p_buf;

		// reading a comment line: "group of order ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading action
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		strcpy(label, buf + 10);
		cout << "action: " << label << endl;
		if (strcmp(label, A->label)) {
			cout << "actions do not match" << endl;
			cout << "action on file: " << label << endl;
			cout << "current action: " << A->label << endl;
			exit(1);
			}
		// reading action
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading comment "base_len ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading base_len
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		base_length = atoi(buf);
		if (f_vv) {
			cout << "base_length=" << base_length << endl;
			}

		base1 = NEW_INT(base_length);
		// reading comment
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading base
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		p_buf = buf;
		for (i = 0; i < base_length; i++) {
			if (!s_scan_int(&p_buf, &base1[i])) {
				cout << "error reading base" << endl;
				exit(1);
				}
			}
		if (f_vv) {
			cout << "read base: ";
			INT_vec_print(cout, base1, base_length);
			cout << endl;
			}

		tl = NEW_INT(base_length);

		// reading comment "transversal lengths ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading transversal lengths
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		p_buf = buf;
		for (i = 0; i < base_length; i++) {
			if (!s_scan_int(&p_buf, &tl[i])) {
				cout << "error reading tl[" << i << "]" << endl;
				exit(1);
				}
			if (f_vv) {
				cout << "read tl[" << i << "]=" << tl[i] << endl;
				//cout << "remaining string: '" << p_buf << "'" << endl;
				}
			}
		if (f_vv) {
			cout << "read tl: ";
			INT_vec_print(cout, tl, base_length);
			cout << endl;
			}

		// reading comment line "number of generators ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading number of generators
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		nb_gens = atoi(buf);
		if (f_vv) {
			cout << "nb_gens=" << nb_gens << endl;
			}
		// reading comment "make element size ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		// reading make_element_size1
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		make_element_size1 = atoi(buf);
		if (f_vv) {
			cout << "make_element_size=" << make_element_size1 << endl;
			}
		if (make_element_size1 != A->make_element_size) {
			cout << "make element size does not match" << endl;
			exit(1);
			}

		// reading comment "generators are ..."
		fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
		
		data = NEW_INT(nb_gens * make_element_size1);
		for (i = 0; i < nb_gens; i++) {
			// reading generator i:
			if (f_vv) {
				cout << "reading generator " << i << ":" << endl;
				}
			fp.getline(buf, READ_SGS_BUF_SIZE, '\n');
			p_buf = buf;
			for (j = 0; j < make_element_size1; j++) {
				if (!s_scan_int(&p_buf, &data[i * make_element_size1 + j])) {
					cout << "error reading generator " << i << " integer " << j << endl;
					exit(1);
					}
				}
			}
	}

	if (f_v) {
		cout << "parsing generators:" << endl;
		}
	SG->init(A);
	SG->allocate(nb_gens);
	for (i = 0; i < nb_gens; i++) {
		A->make_element(Elt, data + i * make_element_size1, 0);
		A->element_move(Elt, SG->ith(i), 0);
		if (f_vv) {
			cout << "generator " << i << ":" << endl;
			A->element_print_quick(SG->ith(i), cout);
			cout << endl;
			}
		}

	if (A->f_has_base) {
		if (base_length != A->base_len) {
			cout << "base_len does not match" << endl;
			cout << "A->base_len=" << A->base_len << endl;
			cout << "read " << base_length << endl;
			exit(1);
			}
		for (i = 0; i < base_length; i++) {
			if (base1[i] != A->base[i]) {
				cout << "base does not match" << endl;
				cout << "A->base[" << i << "]=" << A->base[i] << endl;
				cout << "base1[" << i << "]=" << base1[i] << endl;
				exit(1);
				}
			}
		}
	else {
		if (f_v) {
			cout << "action does not have a base, we will initialize with the base from file" << endl;
			}
		A->allocate_base_data(base_length);
		for (i = 0; i < base_length; i++) {
			A->base[i] = base1[i];
			}
		A->base_len = base_length;
		if (f_vv) {
			cout << "the base is: ";
			INT_vec_print(cout, A->base, A->base_len);
			cout << endl;

			cout << "rallocating base:" << endl;
			}
		reallocate_base(0, verbose_level - 1);
		}

	if (f_v) {
		cout << "before init_trivial_group" << endl;
		}
	init_trivial_group(verbose_level - 2);
	if (f_v) {
		cout << "before init_generators" << endl;
		}
	init_generators(*SG, verbose_level - 2);
	if (f_v) {
		cout << "before compute_base_orbits_known_length" << endl;
		}
	compute_base_orbits_known_length(tl, verbose_level - 2);
	group_order(go);
	if (f_v) {
		cout << "sims::read_sgs created a group of order " << go << endl;
		}
	
	FREE_INT(data);
	FREE_INT(tl);
	FREE_INT(base1);
	FREE_INT(Elt);
	FREE_BYTE(elt);
	FREE_BYTE(buf);
}




INT sims::least_moved_point_at_level(INT lvl, INT verbose_level)
// returns -1 if there are no generators
{
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	INT i, j, h, nbg, gen_idx, least_moved_point = -1;
	
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
		cout << "sims::least_moved_point_at_level lvl = " << lvl << ", least moved point = " << least_moved_point << endl;
		}
	return least_moved_point;
}

#if 0
INT sims::identify_group(BYTE *path_t144, BYTE *discreta_home, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT group_idx;
	INT h, i, j, *Elt;
	longinteger_object go;
	const BYTE *fname = "group_generators.txt";

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
		cout << "sims::identify_group written file " << fname << " of size " << file_size(fname) << endl;
		}
	BYTE cmd[2000];

	sprintf(cmd, "%s/t144.out -discreta_home %s group_generators.txt >log.tmp", path_t144, discreta_home);
	
	if (f_v) {
		cout << "sims::identify_group calling '" << cmd << "'" << endl;
		}

	system(cmd);
		
	{
	ifstream f("result.txt");
	f >> group_idx;
	}
	if (f_v) {
		cout << "sims::identify_group: the group is isomorphic to group " << go << "#" << group_idx << endl;
		}
	return group_idx;
}
#endif

INT sims::mult_by_rank(INT rk_a, INT rk_b, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk_c;

	if (f_v) {
		cout << "sims::mult_by_rank" << endl;
		}
	element_unrank_INT(rk_a, Elt1);
	element_unrank_INT(rk_b, Elt2);
	A->element_mult(Elt1, Elt2, Elt3, 0);
	rk_c = element_rank_INT(Elt3);
	return rk_c;
}

INT sims::mult_by_rank(INT rk_a, INT rk_b)
{
	INT rk_c;

	rk_c = mult_by_rank(rk_a, rk_b, 0);
	return rk_c;
}

INT sims::invert_by_rank(INT rk_a, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT rk_b;

	if (f_v) {
		cout << "sims::invert_by_rank" << endl;
		}
	element_unrank_INT(rk_a, Elt1);
	A->element_invert(Elt1, Elt2, 0);
	rk_b = element_rank_INT(Elt2);
	return rk_b;
}

INT sims::conjugate_by_rank(INT rk_a, INT rk_b, INT verbose_level)
// comutes b^{-1} * a * b
{
	INT f_v = (verbose_level >= 1);
	INT rk_c;

	if (f_v) {
		cout << "sims::conjugate_by_rank" << endl;
		}
	element_unrank_INT(rk_a, Elt1); // Elt1 = a
	element_unrank_INT(rk_b, Elt2); // Elt2 = b
	A->element_invert(Elt2, Elt3, 0); // Elt3 = b^{-1}
	A->element_mult(Elt3, Elt1, Elt4, 0);
	A->element_mult(Elt4, Elt2, Elt3, 0);
	rk_c = element_rank_INT(Elt3);
	return rk_c;
}

INT sims::conjugate_by_rank_b_bv_given(INT rk_a, INT *Elt_b, INT *Elt_bv, INT verbose_level)
// comutes b^{-1} * a * b
{
	INT f_v = (verbose_level >= 1);
	INT rk_c;

	if (f_v) {
		cout << "sims::conjugate_by_rank_b_bv_given" << endl;
		}
	element_unrank_INT(rk_a, Elt1); // Elt1 = a
	A->element_mult(Elt_bv, Elt1, Elt4, 0);
	A->element_mult(Elt4, Elt_b, Elt3, 0);
	rk_c = element_rank_INT(Elt3);
	return rk_c;
}

void sims::sylow_subgroup(INT p, sims *P, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT *Elt1, *Elt2;
	longinteger_domain D;
	longinteger_object go, go1, go_P, go_P1;
	INT i, e, e1, c, nb_trials;

	if (f_v) {
		cout << "sims::sylow_subgroup" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);

	group_order(go);
	if (f_v) {
		cout << "sims::sylow_subgroup the group has order " << go << endl;
		}
	e = D.multiplicity_of_p(go, go1, p);
	if (f_v) {
		cout << "sims::sylow_subgroup the prime " << p << " divides exactly " << e << " times" << endl;
		}
	go_P.create_power(p, e);
	if (f_v) {
		cout << "sims::sylow_subgroup trying to find a subgroup of order " << go_P << endl;
		}
	
	P->init(A);
	P->init_trivial_group(verbose_level - 1);

	P->group_order(go_P1);
	while (TRUE) {
		c = D.compare(go_P1, go_P);
		if (c == 0) {
			break;
			}
		if (c > 0) {
			cout << "sims::sylow_subgroup overshooting the group order" << endl;
			exit(1);
			}


		find_element_of_prime_power_order(p, Elt1, e1, nb_trials, 0 /* verbose_level */);
		for (i = 0; i < e1; i++) {
			if (P->is_normalizing(Elt1, 0 /* verbose_level */)) {
				if (P->strip_and_add(Elt1, Elt2 /* residue */, 0 /* verbose_level */)) {
					P->group_order(go_P1);
					if (f_v) {
						cout << "sims::sylow_subgroup increased the order of the subgroup to " << go_P1 << endl;
						}
					}
				break;
				}
			A->element_power_INT_in_place(Elt1, p, 0 /* verbose_level */);
			}
		}
	if (f_v) {
		cout << "sims::sylow_subgroup found a " << p << "-Sylow subgroup of order " << go_P1 << endl;
		}

	FREE_INT(Elt1);
	FREE_INT(Elt2);
	if (f_v) {
		cout << "sims::sylow_subgroup done" << endl;
		}
}

INT sims::is_normalizing(INT *Elt, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT ret = FALSE;
	INT i;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;
	INT *Elt4;
	INT drop_out_level, image;

	if (f_v) {
		cout << "sims::is_normalizing" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	Elt4 = NEW_INT(A->elt_size_in_INT);

	for (i = 0; i < nb_gen[0]; i++) {
		A->element_invert(Elt, Elt1, 0);
		A->element_move(gens.ith(i), Elt2, 0);
		A->element_mult(Elt1, Elt2, Elt3, 0);
		A->element_mult(Elt3, Elt, Elt4, 0);
		if (!strip(Elt4, Elt3 /* residue */, drop_out_level, image, 0 /* verbose_level */)) {
			if (f_v) {
				cout << "sims::is_normalizing the element does not normalize generator " << i << " / " << nb_gen[0] << endl;
				}
			break;
			}
		}
	if (i == nb_gen[0]) {
		if (f_v) {
			cout << "sims::is_normalizing the element normalizes all " << nb_gen[0] << " generators" << endl;
			}
		ret = TRUE;
		}
	else {
		ret = FALSE;
		}
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);
	FREE_INT(Elt4);
	if (f_v) {
		cout << "sims::is_normalizing done" << endl;
		}
	return ret;
}

void sims::create_Cayley_graph(vector_ge *gens, INT *&Adj, INT &n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, h, j;
	longinteger_object go;
	INT *Elt1;
	INT *Elt2;

	if (f_v) {
		cout << "sims::create_Cayley_graph" << endl;
		}
	group_order(go);
	n = go.as_INT();
	if (f_v) {
		cout << "sims::create_Cayley_graph Computing the adjacency matrix of a graph with " << n << " vertices" << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Adj = NEW_INT(n * n);
	INT_vec_zero(Adj, n * n);
	for (i = 0; i < n; i++) {
		element_unrank_INT(i, Elt1);
		//cout << "i=" << i << endl;
		for (h = 0; h < gens->len; h++) {
			A->element_mult(Elt1, gens->ith(h), Elt2, 0);
#if 0
			cout << "i=" << i << " h=" << h << endl;
			cout << "Elt1=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "g_h=" << endl;
			A->element_print_quick(gens->ith(h), cout);
			cout << "Elt2=" << endl;
			A->element_print_quick(Elt2, cout);
#endif
			j = element_rank_INT(Elt2);
			Adj[i * n + j] = Adj[j * n + i] = 1;
#if 0
			if (i == 0) {
				cout << "edge " << i << " " << j << endl;
				}
#endif
			}
		}

#if 0
	cout << "The adjacency matrix of a graph with " << n << " vertices has been computed" << endl;
	//INT_matrix_print(Adj, goi, goi);
#endif

	FREE_INT(Elt1);
	FREE_INT(Elt2);


	if (f_v) {
		cout << "sims::create_Cayley_graph done" << endl;
		}
}

void sims::create_group_table(INT *&Table, INT &n, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, k;
	longinteger_object go;
	INT *Elt1;
	INT *Elt2;
	INT *Elt3;

	if (f_v) {
		cout << "sims::create_group_table" << endl;
		}
	group_order(go);
	n = go.as_INT();
	if (f_v) {
		cout << "sims::create_group_table Computing the table of a group of order " << n << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Elt3 = NEW_INT(A->elt_size_in_INT);
	Table = NEW_INT(n * n);
	INT_vec_zero(Table, n * n);
	for (i = 0; i < n; i++) {
		element_unrank_INT(i, Elt1);
		//cout << "i=" << i << endl;
		for (j = 0; j < n; j++) {
			element_unrank_INT(j, Elt2);
			A->element_mult(Elt1, Elt2, Elt3, 0);
#if 0
			cout << "i=" << i << " j=" << j << endl;
			cout << "Elt_i=" << endl;
			A->element_print_quick(Elt1, cout);
			cout << "Elt_j=" << endl;
			A->element_print_quick(Elt2, cout);
			cout << "Elt3=" << endl;
			A->element_print_quick(Elt3, cout);
#endif
			k = element_rank_INT(Elt3);
			Table[i * n + j] = k;
			}
		}

	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Elt3);

	if (f_v) {
		cout << "sims::create_group_table done" << endl;
		}
}

void sims::write_as_magma_permutation_group(const BYTE *fname_base, vector_ge *gens, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, k, n, l, h;
	longinteger_object go;
	INT *Elt1;
	INT *Elt2;
	INT *Table;

	if (f_v) {
		cout << "sims::write_as_magma_permutation_group" << endl;
		}
	group_order(go);
	n = go.as_INT();
	l = gens->len;
	if (f_v) {
		cout << "sims::write_as_magma_permutation_group Computing the Table, go=" << go << endl;
		}
	Elt1 = NEW_INT(A->elt_size_in_INT);
	Elt2 = NEW_INT(A->elt_size_in_INT);
	Table = NEW_INT(l * n);
	INT_vec_zero(Table, l * n);
	for (h = 0; h < l; h++) {
		if (f_v) {
			cout << "sims::write_as_magma_permutation_group h = " << h <<  " / " << l << endl;
			}
		for (i = 0; i < n; i++) {
			if (f_v) {
				cout << "sims::write_as_magma_permutation_group i = " << i <<  " / " << n << endl;
				}
			element_unrank_INT(i, Elt1);

			A->element_mult(Elt1, gens->ith(h), Elt2, 0);

			if (f_v) {
				cout << "Elt2=" << endl;
				A->element_print(Elt2, cout);
				}
			k = element_rank_INT(Elt2);
			if (f_v) {
				cout << "has rank k=" << k << endl;
				}
			Table[h * n + i] = k;
			}
		}
#if 0
> G := PermutationGroup< 12 | (1,6,7)(2,5,8,3,4,9)(11,12),
>                             (1,3)(4,9,12)(5,8,10,6,7,11) >;
#endif
	BYTE fname[1000];

	sprintf(fname, "%s.magma", fname_base);
	{
	ofstream fp(fname);
	
	fp << "G := PermutationGroup< " << n << " | " << endl;
	for (i = 0; i < l; i++) {
		perm_print_counting_from_one(fp, Table + i * n, n);
		if (i < l - 1) {
			fp << ", " << endl;
			}
		}
	fp << " >;" << endl;
	}
	cout << "Written file " << fname << " of size " << file_size(fname) << endl;
	
	FREE_INT(Elt1);
	FREE_INT(Elt2);
	FREE_INT(Table);

	if (f_v) {
		cout << "sims::write_as_magma_permutation_group done" << endl;
		}
}


