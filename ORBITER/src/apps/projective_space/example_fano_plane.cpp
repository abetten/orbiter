// example_fano_plane.C
//
//
// Anton Betten
// September 25, 2017

#include "orbiter.h"


using namespace orbiter;


int main()
{
	int verbose_level = 0;
	int q = 2;
	int d = 3;
	int k = 4;
	finite_field *F;
	int f_projective = TRUE;
	int f_general = FALSE;
	int f_affine = FALSE;
	int f_semilinear = FALSE;
	int f_special = FALSE;
	sims *S;
	action *A;
	longinteger_object go;
	int *Elt;
	int i;
	int *v;
	schreier *Sch;
	vector_ge *nice_gens;

		
	v = NEW_int(d);

	F = NEW_OBJECT(finite_field);
	F->init(q, 0);

	create_linear_group(S, A, 
		F, d, 
		f_projective, f_general, f_affine, 
		f_semilinear, f_special, 
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);
	
	A->group_order(go);
	cout << "created a group of order " << go << endl;
	
	Elt = NEW_int(A->elt_size_in_int);

	for (i = 0; i < go.as_int(); i++) {
		S->element_unrank_int(i, Elt, 0 /* verbose_level */);
		cout << "element " << i << " / " << go << ":" << endl;
		A->element_print_quick(Elt, cout);
		}

	for (i = 0; i < A->degree; i++) {
		F->PG_element_unrank_modified(v, 1, d, i);
		cout << "point " << i << " / " << A->degree << " is ";
		int_vec_print(cout, v, d);
		cout << endl;
		}

	cout << "generating set: " << endl;
	A->Strong_gens->print_generators();

	Sch = A->Strong_gens->orbits_on_points_schreier(A, verbose_level);
	
	cout << "We have " << Sch->nb_orbits << " orbits on points" << endl;

	Sch->print_and_list_orbits(cout);
	
	poset *Poset;
	poset_classification *Gen;

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);

	Gen = orbits_on_k_sets_compute(Poset,
		k, verbose_level);
	Gen->print_orbit_numbers(k);

	int nb_orbits;

	nb_orbits = Gen->nb_orbits_at_level(k);
	cout << "We found " << nb_orbits
			<< " orbits of subsets of size " << k << endl;

	for (i = 0; i < nb_orbits; i++) {
	
		set_and_stabilizer *SaS;

		SaS = Gen->get_set_and_stabilizer(
				k /* level */,
				i /* orbit_at_level */,
				0 /* verbose_level */);
		cout << "orbit " << i << " / " << nb_orbits << " : ";
		SaS->print_set_tex(cout);
		cout << endl;

		SaS->print_generators_tex(cout);
		}

	
	FREE_OBJECT(Gen);
	FREE_OBJECT(Poset);
	FREE_OBJECT(F);
}

