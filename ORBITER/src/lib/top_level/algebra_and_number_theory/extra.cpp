// extra.C
// 
// Anton Betten
//
// started 9/23/2010
//
//
// 
//
//

#include "orbiter.h"

namespace orbiter {
namespace top_level {

void isomorph_print_set(ostream &ost, int len, int *S, void *data)
{
	//isomorph *G = (isomorph *) data;
	
	print_vector(ost, S, (int) len);
	//G->print(ost, S, len);
}


sims *create_sims_for_stabilizer(action *A, 
	int *set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_stabilizer_compute STAB;
	sims *Stab;
	int nb_backtrack_nodes;
	int t0 = os_ticks();
	
	if (f_v) {
		cout << "create_sims_for_stabilizer" << endl;
		}
	strong_generators *Aut_gens;

	poset *Poset;

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A, A, A->Strong_gens, verbose_level);
	STAB.init(Poset, set, set_size, verbose_level);
	STAB.compute_set_stabilizer(t0,
			nb_backtrack_nodes, Aut_gens,
			verbose_level - 1);
	
	Stab = Aut_gens->create_sims(verbose_level - 1);
	

	FREE_OBJECT(Aut_gens);
	if (f_v) {
		longinteger_object go;
		Stab->group_order(go);
		cout << "create_sims_for_stabilizer, "
				"found a group of order " << go << endl;
		}
	FREE_OBJECT(Poset);
	return Stab;
}

sims *create_sims_for_stabilizer_with_input_group(action *A, 
	action *A0, strong_generators *Strong_gens, 
	int *set, int set_size, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	set_stabilizer_compute STAB;
	strong_generators *Aut_gens;
	sims *Stab;
	int nb_backtrack_nodes;
	int t0 = os_ticks();
	
	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group" << endl;
		}

	poset *Poset;

	Poset = NEW_OBJECT(poset);
	Poset->init_subset_lattice(A0, A, A0->Strong_gens, verbose_level);
	STAB.init_with_strong_generators(Poset,
			set, set_size, verbose_level);
	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group "
				"after STAB.init_with_strong_generators" << endl;
		}

	
	STAB.compute_set_stabilizer(t0, nb_backtrack_nodes,
			Aut_gens, verbose_level - 1);

	Stab = Aut_gens->create_sims(verbose_level - 1);
	

	FREE_OBJECT(Poset);
	FREE_OBJECT(Aut_gens);

	if (f_v) {
		cout << "create_sims_for_stabilizer_with_input_group "
				"after STAB.compute_set_stabilizer" << endl;
		}
	
	if (f_v) {
		longinteger_object go;
		Stab->group_order(go);
		cout << "create_sims_for_stabilizer_with_input_group, "
				"found a group of order " << go << endl;
		}
	return Stab;
}



}}



