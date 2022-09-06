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
	int f_v = FALSE;

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

void isomorph::init(std::string &prefix,
		actions::action *A_base, actions::action *A,
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


	isomorph::prefix.assign(prefix);

	prefix_invariants.assign(prefix);
	prefix_invariants.append("INVARIANTS/");


	prefix_tex.assign(prefix);
	prefix_tex.append("TEX/");


	isomorph::A_base = A_base;
	isomorph::A = A;


	Sub = NEW_OBJECT(substructure_classification);

	if (f_v) {
		cout << "isomorph::init before Sub->init" << endl;
	}

	Sub->init(this,
			gen,
			f_use_database_for_starter,
			f_implicit_fusion,
			verbose_level);

	if (f_v) {
		cout << "isomorph::init after Sub->init" << endl;
	}


	Lifting = NEW_OBJECT(substructure_lifting_data);

	if (f_v) {
		cout << "isomorph::init before Lifting->init" << endl;
	}

	Lifting->init(this, verbose_level);

	if (f_v) {
		cout << "isomorph::init after Lifting->init" << endl;
	}

	Folding = NEW_OBJECT(flag_orbit_folding);

	if (f_v) {
		cout << "isomorph::init before Folding->init" << endl;
	}

	Folding->init(this, verbose_level);

	if (f_v) {
		cout << "isomorph::init after Folding->init" << endl;
	}


#if 0
	char cmd[1000];

	sprintf(cmd, "mkdir %s", prefix.c_str());
	system(cmd);
	sprintf(cmd, "mkdir %sINVARIANTS/", prefix.c_str());
	system(cmd);
	sprintf(cmd, "mkdir %sTEX/", prefix.c_str());
	system(cmd);
#endif

	if (f_v) {
		cout << "isomorph::init done" << endl;
	}
}

void isomorph::print_node_local(int level, int node_local)
{
	Sub->print_node_local(level, node_local);
}

void isomorph::print_node_global(int level, int node_global)
{
	Sub->print_node_global(level, node_global);
}


void isomorph::init_high_level(actions::action *A,
		poset_classification::poset_classification *gen,
	int size, std::string &prefix_classify, std::string &prefix, int level,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "isomorph::init_high_level" << endl;
	}

	
	layer2_discreta::discreta_init();

	int f_use_database_for_starter = FALSE;
	int f_implicit_fusion = FALSE;
	
	if (f_v) {
		cout << "isomorph::init_high_level before init" << endl;
	}
	init(prefix, A, A, gen, 
		size, level, 
		f_use_database_for_starter, 
		f_implicit_fusion, 
		verbose_level);
		// sets q, level and initializes file names
	if (f_v) {
		cout << "isomorph::init_high_level after init" << endl;
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




}}

