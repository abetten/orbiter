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
	Space = NULL;

	nb_objects = 0;

	Object_create_description = NULL;

	Object_create = NULL;

	Object_with_action = NULL;


}

table_of_blt_sets::~table_of_blt_sets()
{
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
	if (Object_with_action) {
		FREE_OBJECTS(Object_with_action);
	}
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

	knowledge_base::knowledge_base K;

	int h;


	poset_classification::poset_classification_control Control_six_arcs;


	nb_objects = K.BLT_nb_reps(Space->O->F->q);

	Object_create_description = NEW_OBJECTS(BLT_set_create_description, nb_objects);

	Object_create = NEW_OBJECTS(BLT_set_create, nb_objects);



	Object_with_action = NEW_OBJECTS(blt_set_with_action, nb_objects);


	for (h = 0; h < nb_objects; h++) {

		if (f_v) {
			cout << "table_of_blt_sets::init "
					<< h << " / " << nb_objects << endl;
		}

		Object_create_description[h].f_catalogue = true;
		Object_create_description[h].iso = h;


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

		string label_txt;
		string label_tex;


		label_txt = "BLT_set_" + std::to_string(Space->P->Subspaces->F->q) + "_iso" + std::to_string(h);
		label_tex = "BLT\\_set\\_" + std::to_string(Space->P->Subspaces->F->q) + "\\_iso" + std::to_string(h);




		if (f_v) {
			cout << "table_of_blt_sets::init "
					"before Object_with_action[h].init_with_surface_object" << endl;
		}
		Object_with_action[h].init_set(
				Space->A,
				Space->Blt_set_domain_with_action,
				Object_create[h].set,
				label_txt,
				label_tex,
				Object_create[h].Sg,
				true /* f_invariants */,
				verbose_level);

		if (f_v) {
			cout << "table_of_blt_sets::init "
					"after Object_with_action[h].init_with_surface_object" << endl;
		}
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
	int nb_cols;

	nb_cols = 4;

	Table = new string[nb_objects * nb_cols];

	long int *Row;


	Row = NEW_lint(nb_cols);

	for (h = 0; h < nb_objects; h++) {

		if (f_v) {
			cout << "table_of_blt_sets::do_export "
					<< h << " / " << nb_objects << endl;
		}

		Row[0] = h;


		if (f_v) {
			cout << "collineation stabilizer order" << endl;
		}
		if (Object_create[h].f_has_group) {
			Row[1] = Object_create[h].Sg->group_order_as_lint();
		}
		else {
			Row[1] = 0;
		}
#if 0
		if (f_v) {
			cout << "projectivity stabilizer order" << endl;
		}
		if (Space->A->is_semilinear_matrix_group()) {


			Row[2] = Object_with_action[h].projectivity_group_gens->group_order_as_lint();

		}
		else {
			Row[2] = Object_create[h].Sg->group_order_as_lint();
		}
#endif

		Row[2] = Object_create[h].OA->Blt_set_domain_with_action->Blt_set_domain->target_size;
		if (f_v) {
			cout << "table_of_blt_sets::do_export before Row[3]" << endl;
		}
		Row[3] = Object_with_action[h].Blt_set_group_properties->Orbits_on_points->Sch->nb_orbits;



		int j;

		for (j = 0; j < nb_cols; j++) {
			Table[h * nb_cols + j] = std::to_string(Row[j]);
		}


	}

	FREE_lint(Row);

	if (f_v) {
		cout << "table_of_blt_sets::do_export "
				"before export_csv" << endl;
	}

	export_csv(Table,
			nb_cols,
			verbose_level);

	if (f_v) {
		cout << "table_of_blt_sets::do_export "
				"after export_csv" << endl;
	}


	delete [] Table;

	if (f_v) {
		cout << "table_of_blt_sets::do_export done" << endl;
	}
}


void table_of_blt_sets::export_csv(
		std::string *Table,
		int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_blt_sets::export_csv" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	string fname;
	fname = "table_of_blt_sets_q" + std::to_string(Space->O->F->q) + ".csv";

	//Fio.lint_matrix_write_csv(fname, Table, nb_quartic_curves, nb_cols);

	{
		ofstream f(fname);
		int i, j;

		f << "Row,OCN,Ago,NbOrbPts";



		f << endl;
		for (i = 0; i < nb_objects; i++) {
			f << i;
			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}


	cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

	if (f_v) {
		cout << "table_of_blt_sets::export_csv done" << endl;
	}
}



}}}

