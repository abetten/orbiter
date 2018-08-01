// example_fano_plane.C
//
//
// Anton Betten
// September 25, 2017

#include "orbiter.h"

int main()
{
	INT verbose_level = 0;
	INT q = 2;
	INT d = 3;
	INT k = 4;
	finite_field *F;
	INT f_projective = TRUE;
	INT f_general = FALSE;
	INT f_affine = FALSE;
	INT f_semilinear = FALSE;
	INT f_special = FALSE;
	sims *S;
	action *A;
	longinteger_object go;
	INT *Elt;
	INT i;
	INT *v;
	schreier *Sch;

		
	v = NEW_INT(d);

	F = new finite_field;
	F->init(q, 0);

	create_linear_group(S, A, 
		F, d, 
		f_projective, f_general, f_affine, 
		f_semilinear, f_special, 
		verbose_level);
	
	A->group_order(go);
	cout << "created a group of order " << go << endl;
	
	Elt = NEW_INT(A->elt_size_in_INT);

	for (i = 0; i < go.as_INT(); i++) {
		S->element_unrank_INT(i, Elt, 0 /* verbose_level */);
		cout << "element " << i << " / " << go << ":" << endl;
		A->element_print_quick(Elt, cout);
		}

	for (i = 0; i < A->degree; i++) {
		PG_element_unrank_modified(*F, v, 1, d, i);
		cout << "point " << i << " / " << A->degree << " is ";
		INT_vec_print(cout, v, d);
		cout << endl;
		}

	cout << "generating set: " << endl;
	A->Strong_gens->print_generators();

	Sch = A->Strong_gens->orbits_on_points_schreier(A, verbose_level);
	
	cout << "We have " << Sch->nb_orbits << " orbits on points" << endl;

	Sch->print_and_list_orbits(cout);
	
	generator *Gen;

	Gen = orbits_on_k_sets_compute(A, A, 
		A->Strong_gens, 
		k, verbose_level);
	Gen->print_orbit_numbers(k);

	INT nb_orbits;

	nb_orbits = Gen->nb_orbits_at_level(k);
	cout << "We found " << nb_orbits << " orbits of subsets of size " << k << endl;

	for (i = 0; i < nb_orbits; i++) {
	
		set_and_stabilizer *SaS;

		SaS = Gen->get_set_and_stabilizer(k /* level */, i /* orbit_at_level */, 0 /* verbose_level */);
		cout << "orbit " << i << " / " << nb_orbits << " : ";
		SaS->print_set_tex(cout);
		cout << endl;

		SaS->print_generators_tex(cout);
		}

	
	
}

