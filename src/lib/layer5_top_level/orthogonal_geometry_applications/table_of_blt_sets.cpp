/*
 * table_of_blt_sets.cpp
 *
 *  Created on: Feb 4, 2023
 *      Author: betten
 */




#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace orthogonal_geometry_applications {


table_of_blt_sets::table_of_blt_sets()
{
	Record_birth();
	Space = NULL;

	nb_objects = 0;

	Object_create_description = NULL;

	Object_create = NULL;

	//Object_with_action = NULL;


}

table_of_blt_sets::~table_of_blt_sets()
{
	Record_death();
	int verbose_level = 1;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_blt_sets::~table_of_blt_sets 1" << endl;
	}
	if (Object_create_description) {
		FREE_OBJECTS(Object_create_description);
	}
	if (f_v) {
		cout << "table_of_blt_sets::~table_of_blt_sets 2" << endl;
	}
	if (Object_create) {
		FREE_OBJECTS(Object_create);
	}
	if (f_v) {
		cout << "table_of_blt_sets::~table_of_blt_sets 3" << endl;
	}
#if 0
	if (Object_with_action) {
		FREE_OBJECTS(Object_with_action);
	}
#endif
	if (f_v) {
		cout << "table_of_blt_sets::~table_of_blt_sets 4" << endl;
	}
}

void table_of_blt_sets::init(
		orthogonal_geometry_applications::orthogonal_space_with_action *Space,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_blt_sets::init" << endl;
	}

	table_of_blt_sets::Space = Space;

	combinatorics::knowledge_base::knowledge_base K;

	int h;


	poset_classification::poset_classification_control Control_six_arcs;


	nb_objects = K.BLT_nb_reps(Space->O->F->q);

	Object_create_description = NEW_OBJECTS(BLT_set_create_description, nb_objects);

	Object_create = NEW_OBJECTS(BLT_set_create, nb_objects);


#if 0
	Object_with_action = NEW_OBJECTS(blt_set_with_action, nb_objects);
#endif


	for (h = 0; h < nb_objects; h++) {

		if (f_v) {
			cout << "table_of_blt_sets::init "
					<< h << " / " << nb_objects << endl;
		}

		Object_create_description[h].f_catalogue = true;
		Object_create_description[h].iso = h;
		Object_create_description[h].f_invariants = false; //true;


		if (f_v) {
			cout << "table_of_blt_sets::init "
					"before Object_create->init" << endl;
		}
		Object_create[h].init(
				&Object_create_description[h],
				Space,
				verbose_level);
		if (f_v) {
			cout << "table_of_blt_sets::init "
					"after Object_create->init" << endl;
		}






		if (!Object_create[h].f_has_group) {
			cout << "!Object_create[h]->f_has_group" << endl;
			exit(1);
		}

#if 0
		string label_txt;
		string label_tex;

		Object_create[h].Blt_set->label_txt = "BLT_set_" + std::to_string(Space->P->Subspaces->F->q) + "_iso" + std::to_string(h);
		Object_create[h].Blt_set->label_tex = "BLT\\_set\\_" + std::to_string(Space->P->Subspaces->F->q) + "\\_iso" + std::to_string(h);
#endif



#if 0
		if (f_v) {
			cout << "table_of_blt_sets::init "
					"before Object_with_action[h].init_with_surface_object" << endl;
		}
		Object_with_action[h].init_set(
				Space->A,
				Space->Blt_set_domain_with_action,
				Object_create[h].Blt_set,
				//label_txt,
				//label_tex,
				Object_create[h].Sg,
				//true /* f_invariants */,
				verbose_level);

		if (f_v) {
			cout << "table_of_blt_sets::init "
					"after Object_with_action[h].init_with_surface_object" << endl;
		}
#endif


	}




	if (f_v) {
		cout << "table_of_blt_sets::init done" << endl;
	}

}


void table_of_blt_sets::do_export(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_blt_sets::do_export" << endl;
	}

	int h;
	std::string *Table;
	int nb_rows;
	int nb_cols;
	std::string *ColHeadings;

	nb_rows = nb_objects;
	nb_cols = 6;

	Table = new string[nb_rows * nb_cols];
	ColHeadings = new string[nb_cols];


	for (h = 0; h < nb_objects; h++) {

		if (f_v) {
			cout << "table_of_blt_sets::do_export "
					<< h << " / " << nb_objects << endl;
		}


		Table[h * nb_cols + 0] = std::to_string(h);



		Table[h * nb_cols + 1] = std::to_string(Space->O->F->q);


		Table[h * nb_cols + 2] = std::to_string(h);


		if (f_v) {
			cout << "collineation stabilizer order" << endl;
		}
		if (Object_create[h].f_has_group) {

			Table[h * nb_cols + 3] = Object_create[h].Sg->group_order_stringify();

		}


		Table[h * nb_cols + 4] = "\"" + Lint_vec_stringify(Object_create[h].BA->Blt_set->set, Object_create[h].OA->Blt_set_domain_with_action->Blt_set_domain->target_size) + "\"";


		if (f_v) {
			cout << "table_of_blt_sets::do_export before Row[3]" << endl;
		}

		Table[h * nb_cols + 5] = std::to_string(Object_create[h].BA->Blt_set_group_properties->Orbits_on_points->Sch->Forest->nb_orbits);



	}

	ColHeadings[0] = "Line";
	ColHeadings[1] = "Q";
	ColHeadings[2] = "OCN";
	ColHeadings[3] = "AGO";
	ColHeadings[4] = "Rep";
	ColHeadings[5] = "NbOrbPoints";







	other::orbiter_kernel_system::file_io Fio;

	string fname;
	fname = "table_of_blt_sets_q" + std::to_string(Space->O->F->q) + ".csv";


	if (f_v) {
		cout << "table_of_blt_sets::do_export "
				"before Fio.Csv_file_support->write_table_of_strings_with_col_headings" << endl;
	}
	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			ColHeadings,
			verbose_level);
	if (f_v) {
		cout << "table_of_blt_sets::do_export "
				"after Fio.Csv_file_support->write_table_of_strings_with_col_headings" << endl;
	}

	if (f_v) {
		cout << "table_of_blt_sets::do_export "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}



	delete [] Table;
	delete [] ColHeadings;

	if (f_v) {
		cout << "table_of_blt_sets::do_export done" << endl;
	}
}




}}}

