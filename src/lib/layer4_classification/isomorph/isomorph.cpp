// isomorph.cpp
// 
// Anton Betten
// started 2007
// moved here from reader2.cpp: 3/22/09
// renamed isomorph.cpp from global.cpp: 7/14/11
//
// 
//
//

#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "layer3_group_actions/group_actions.h"
#include "classification.h"

using namespace std;

namespace orbiter {
namespace layer4_classification {
namespace isomorph {


isomorph::isomorph()
{
	size = 0;
	level = 0;

	//std::string prefix;
	//std::string prefix_invariants;
	//std::string prefix_tex;

	A_base = NULL;
	A = NULL;
	
	Sub = NULL;

	Lifting = NULL;
	
	Folding = NULL;
	

	//stabilizer_recreated = NULL;
	print_set_function = NULL;
	print_set_data = NULL;
	
	nb_times_make_set_smaller_called = 0;
}



isomorph::~isomorph()
{
	int f_v = false;

	if (f_v) {
		cout << "isomorph::~isomorph" << endl;
	}

	if (Sub) {
		FREE_OBJECT(Sub);
	}
	if (Lifting) {
		FREE_OBJECT(Lifting);
	}
	if (Folding) {
		FREE_OBJECT(Folding);
	}

	if (f_v) {
		cout << "isomorph::~isomorph done" << endl;
	}
}

void isomorph::init(
		std::string &prefix,
		actions::action *A_base,
		actions::action *A,
		poset_classification::poset_classification *gen,
	int size, int level, 
	int f_use_database_for_starter, 
	int f_implicit_fusion,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph::init" << endl;
		cout << "prefix=" << prefix << endl;
		cout << "A_base=" << A_base->label << endl;
		cout << "A=" << A->label << endl;
		cout << "size=" << size << endl;
		cout << "level=" << level << endl;
		cout << "f_use_database_for_starter="
				<< f_use_database_for_starter << endl;
		cout << "f_implicit_fusion=" << f_implicit_fusion << endl;
	}

	isomorph::size = size;
	isomorph::level = level;


	isomorph::prefix = prefix;

	prefix_invariants = prefix + "INVARIANTS/";


	prefix_tex = prefix + "TEX/";


	isomorph::A_base = A_base;
	isomorph::A = A;


	Sub = NEW_OBJECT(substructure_classification);

	if (f_v) {
		cout << "isomorph::init "
				"before Sub->init" << endl;
	}

	Sub->init(this,
			gen,
			f_use_database_for_starter,
			f_implicit_fusion,
			verbose_level - 1);

	if (f_v) {
		cout << "isomorph::init "
				"after Sub->init" << endl;
	}


	Lifting = NEW_OBJECT(substructure_lifting_data);

	if (f_v) {
		cout << "isomorph::init "
				"before Lifting->init" << endl;
	}

	Lifting->init(this, verbose_level - 1);

	if (f_v) {
		cout << "isomorph::init "
				"after Lifting->init" << endl;
	}

	Folding = NEW_OBJECT(flag_orbit_folding);

	if (f_v) {
		cout << "isomorph::init "
				"before Folding->init" << endl;
	}

	Folding->init(this, verbose_level - 1);

	if (f_v) {
		cout << "isomorph::init "
				"after Folding->init" << endl;
	}


	if (f_v) {
		cout << "isomorph::init done" << endl;
	}
}

void isomorph::print_node_local(
		int level, int node_local)
{
	Sub->print_node_local(level, node_local);
}

void isomorph::print_node_global(
		int level, int node_global)
{
	Sub->print_node_global(level, node_global);
}


void isomorph::init_high_level(
		actions::action *A,
		poset_classification::poset_classification *gen,
	int size,
	std::string &prefix_classify,
	std::string &prefix, int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph::init_high_level" << endl;
	}

	
	layer2_discreta::typed_objects::discreta_init();

	int f_use_database_for_starter = false;
	int f_implicit_fusion = false;
	
	if (f_v) {
		cout << "isomorph::init_high_level "
				"before init" << endl;
	}
	init(prefix, A, A, gen, 
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level);
		// sets q, level and initializes file names
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after init" << endl;
	}


	

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before Sub->read_data_files_for_starter" << endl;
	}

	Sub->read_data_files_for_starter(level,
			prefix_classify, verbose_level);

	if (f_v) {
		cout << "isomorph::init_high_level "
				"after Sub->read_data_files_for_starter" << endl;
	}

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before init_solution" << endl;
	}

	Lifting->init_solution(verbose_level);
	
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after init_solution" << endl;
	}


	if (f_v) {
		cout << "isomorph::init_high_level "
				"before Lifting->read_orbit_data" << endl;
	}

	Lifting->read_orbit_data(verbose_level);

	if (f_v) {
		cout << "isomorph::init_high_level "
				"after Lifting->read_orbit_data" << endl;
	}


	Sub->depth_completed = level /*- 2*/;

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before Folding->iso_test_init" << endl;
	}
	Folding->iso_test_init(verbose_level);
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after Folding->iso_test_init" << endl;
	}

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before Reps->load" << endl;
	}
	Folding->Reps->load(verbose_level);
	if (f_v) {
		cout << "isomorph::init_high_level "
				"after Reps->load" << endl;
	}

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before setup_and_open_solution_database" << endl;
	}
	Lifting->setup_and_open_solution_database(verbose_level - 1);

	if (f_v) {
		cout << "isomorph::init_high_level "
				"before Sub->setup_and_open_level_database" << endl;
	}
	Sub->setup_and_open_level_database(verbose_level - 1);


	if (f_v) {
		cout << "isomorph::init_high_level done" << endl;
	}
}

void isomorph::induced_action_on_set_and_kernel(
	ostream &file,
	actions::action *A,
	groups::sims *Stab, int size, long int *set,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	actions::action *AAA;
	//sims K;
	ring_theory::longinteger_object go, ko;
	int i;
	int *Elt1;

	Elt1 = NEW_int(A->elt_size_in_int);

	if (f_v) {
		cout << "isomorph::induced_action_on_set_and_kernel "
				"calling induced_action_by_restriction" << endl;
		cout << "set: ";
		Lint_vec_print(cout, set, size);
		cout << endl;
	}

	int f_induce_action = true;
	std::string label_of_set;
	std::string label_of_set_tex;

	label_of_set.assign("_isomorph::induced_action_on_set_and_kernel");
	label_of_set_tex.assign("\\_isomorph::induced\\_action\\_on\\_set\\_and\\_kernel");

	AAA = Sub->gen->get_A2()->Induced_action->create_induced_action_by_restriction(
		Stab,
		size, set, label_of_set, label_of_set_tex,
		f_induce_action,
		verbose_level - 1);

	if (f_v) {
		cout << "isomorph::induced_action_on_set_and_kernel: "
				"after induced_action_by_restriction" << endl;
	}

	AAA->group_order(go);

	file << endl << "\\bigskip" << endl
			<< "The induced group has order "
			<< go << " and is generated by:" << endl << endl;
	AAA->group_order(go);
	for (i = 0; i < Stab->gens.len; i++) {
		int f_do_it_anyway_even_for_big_degree= true;
		int f_print_cycles_of_length_one = true;

		file << "$g_{" << setw(2) << i + 1 << "} = $";
		AAA->Group_element->element_print_as_permutation_with_offset(
			Stab->gens.ith(i), file, 1,
			f_do_it_anyway_even_for_big_degree,
			f_print_cycles_of_length_one,
			0 /* verbose_level */);
		file << "\\\\" << endl;
	}
	if (go.compare_with_int(10) < 0) {
		file << "group order is small, so we list "
				"all elements\\\\" << endl;
		for (i = 0; i < go.as_lint(); i++) {
			int f_do_it_anyway_even_for_big_degree = true;
			int f_print_cycles_of_length_one = true;

			file << "$a_{" << setw(2) << i + 1 << "} = $";
			Stab->element_unrank_lint(i, Elt1);
			AAA->Group_element->element_print_as_permutation_with_offset(
					Elt1, file, 1,
				f_do_it_anyway_even_for_big_degree,
				f_print_cycles_of_length_one,
				0 /* verbose_level */);
			file << "\\\\" << endl;
		}
		file << "and now the elements themselves:" << endl;
		for (i = 0; i < go.as_lint(); i++) {

			Stab->element_unrank_lint(i, Elt1);

			//int n;
			vector<long int> fp;

			//fp = NEW_int(A->degree);
			A->Group_element->compute_fixed_points(Elt1, fp, 0);
			//cout << "with " << n << " fixed points" << endl;
			//FREE_int(fp);


			file << "$a_{" << setw(2) << i + 1 << "} = $" << endl;
			file << "$";
			AAA->Group_element->element_print_latex(Elt1, file);
			file << "$ with " << fp.size() << " fixed points\\\\" << endl;
		}
	}

	if (AAA->Kernel) {
		if (f_v) {
			cout << "isomorph::induced_action_on_set_and_kernel: "
					"printing kernel generators" << endl;
		}
		AAA->Kernel->group_order(ko);
		file << "Kernel has order " << ko
				<< " and is generated by:\\\\" << endl;
		for (i = 0; i < AAA->Kernel->gens.len; i++) {
			file << "$$ b_{" << setw(2) << i + 1 << "} = " << endl;
			A->Group_element->element_print_latex(
					AAA->Kernel->gens.ith(i), file);
			file << "$$" << endl;
			//file << "$b_{" << setw(2) << i + 1 << "} = $" << endl;
			//A->element_print_as_permutation_with_offset(K.gens.ith(i), file, 1);
			file << "\\\\" << endl;
		}

		if (!ko.is_one()) {
			groups::schreier Orb;
			isomorph::A->compute_all_point_orbits(Orb,
					AAA->Kernel->gens, verbose_level - 2);
			int *val, *mult, len;

			file << "The kernel has $" << Orb.nb_orbits
					<< "$ orbits on the quadric.\\\\" << endl;
			Int_vec_distribution(
					Orb.orbit_len, Orb.nb_orbits,
					val, mult, len);
			file << "The orbit length are $[";
			for (i = len - 1; i >= 0; i--) {
				file << val[i];
				if (mult[i] > 1) {
					file << "^{" << mult[i] << "}";
				}
				if (i) {
					file << ", ";
				}
			}
			file << "]$\\\\" << endl;

#if 0
			int min_length, min_idx;

			min_idx = -1;
			for (i = 0; i < Orb.nb_orbits; i++) {
				if (Orb.orbit_len[i] == 1) {
					continue;
				}
				if (min_idx == -1) {
					min_idx = i;
					min_length = Orb.orbit_len[i];
					continue;
				}
				if (Orb.orbit_len[i] < min_length) {
					min_idx = i;
					min_length = Orb.orbit_len[i];
				}
			}
			if (min_idx >= 0) {
				induced_action_on_orbit(file, AAA.Kernel->A,
						AAA.Kernel, Orb, min_idx, verbose_level);
			}
#endif

			FREE_int(val);
			FREE_int(mult);
		}
	} // if (AAA.Kernel)


	file << "\\bigskip" << endl << endl;
	FREE_int(Elt1);
	FREE_OBJECT(AAA);

}


void isomorph::read_everything_including_classification(
		std::string &prefix_classify, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "isomorph::read_everything_including_classification" << endl;
	}


	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before Sub->read_data_files_for_starter" << endl;
	}
	Sub->read_data_files_for_starter(level,
			prefix_classify, verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after Sub->read_data_files_for_starter" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before Lifting->init_solution" << endl;
	}
	Lifting->init_solution(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after Lifting->init_solution" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before Lifting->load_table_of_solutions" << endl;
	}
	Lifting->load_table_of_solutions(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after Lifting->load_table_of_solutions" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before Lifting->read_orbit_data" << endl;
	}
	Lifting->read_orbit_data(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after Lifting->read_orbit_data" << endl;
	}

	Sub->depth_completed = level /*- 2*/;

	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before gen->recreate_schreier_vectors_up_to_level" << endl;
	}
	Sub->gen->recreate_schreier_vectors_up_to_level(level - 1,
			verbose_level);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after gen->recreate_schreier_vectors_up_to_level" << endl;
	}


	if (f_v) {
		for (i = 0; i <= level + 1; i++) {
			cout << "gen->first_node_at_level[" << i << "]"
					"=" << Sub->gen->first_node_at_level(i) << endl;
		}
		cout << "depth_completed=" << Sub->depth_completed << endl;
	}


	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before Folding->iso_test_init" << endl;
	}
	Folding->iso_test_init(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after Folding->iso_test_init" << endl;
	}

	//int f_implicit_fusion = false;

	//gen->f_allowed_to_show_group_elements = false;

	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before read_starter_nb_orbits" << endl;
	}
	Lifting->read_starter_nb_orbits(verbose_level); // added Oct 30, 2014
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after read_starter_nb_orbits" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"before Folding->Reps->load" << endl;
	}
	Folding->Reps->load(verbose_level - 1);
	if (f_v) {
		cout << "isomorph::read_everything_including_classification "
				"after Folding->Reps->load" << endl;
	}

	if (f_v) {
		cout << "isomorph::read_everything_including_classification done" << endl;
	}
}



}}}

