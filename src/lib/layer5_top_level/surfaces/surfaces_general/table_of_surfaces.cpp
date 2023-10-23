/*
 * table_of_surfaces.cpp
 *
 *  Created on: Feb 4, 2023
 *      Author: betten
 */

#include "orbiter.h"

using namespace std;


namespace orbiter {
namespace layer5_applications {
namespace applications_in_algebraic_geometry {
namespace cubic_surfaces_in_general {


table_of_surfaces::table_of_surfaces()
{
	PA = NULL;

	nb_cubic_surfaces = 0;

	Surface_create_description = NULL;

	SC = NULL;

	SOG = NULL;

}

table_of_surfaces::~table_of_surfaces()
{
	if (Surface_create_description) {
		FREE_OBJECTS(Surface_create_description);
	}
	if (SC) {
		FREE_OBJECTS(SC);
	}
	if (SOG) {
		FREE_OBJECTS(SOG);
	}
}



void table_of_surfaces::init(
		projective_geometry::projective_space_with_action *PA,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "table_of_surfaces::init" << endl;
	}

	table_of_surfaces::PA = PA;

	knowledge_base::knowledge_base K;

	int h;




	poset_classification::poset_classification_control Control_six_arcs;



	nb_cubic_surfaces = K.cubic_surface_nb_reps(PA->F->q);

	Surface_create_description = NEW_OBJECTS(surface_create_description, nb_cubic_surfaces);

	SC = NEW_OBJECTS(surface_create, nb_cubic_surfaces);



	SOG = NEW_OBJECTS(surface_object_with_group, nb_cubic_surfaces);

	if (f_v) {
		cout << "table_of_surfaces::init creating surfaces" << endl;
	}

	for (h = 0; h < nb_cubic_surfaces; h++) {

		if (f_vv) {
			cout << "table_of_surfaces::init "
					<< h << " / " << nb_cubic_surfaces << endl;
		}

		Surface_create_description[h].f_catalogue = true;
		Surface_create_description[h].iso = h;


		Surface_create_description[h].f_space_pointer = true;
		Surface_create_description[h].space_pointer = PA;


		if (f_vv) {
			cout << "table_of_surfaces::init before SC->init" << endl;
		}
		SC[h].init(&Surface_create_description[h], verbose_level - 2);
		if (f_v) {
			cout << "table_of_surfaces::init after SC->init" << endl;
		}






		if (!SC[h].f_has_group) {
			cout << "!SC[h]->f_has_group" << endl;
			exit(1);
		}

		if (f_vv) {
			cout << "table_of_surfaces::init "
					"before SOG[h].init_with_surface_object" << endl;
		}
		SOG[h].init_with_surface_object(PA->Surf_A,
				SC[h].SO,
				SC[h].Sg,
				false /* f_has_nice_gens */,
				NULL /* vector_ge *nice_gens */,
				verbose_level - 2);
		if (f_vv) {
			cout << "table_of_surfaces::init "
					"after SOG[h].init_with_surface_object" << endl;
		}
	}




	if (f_v) {
		cout << "table_of_surfaces::init done" << endl;
	}

}


void table_of_surfaces::do_export(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_surfaces::do_export" << endl;
	}

	std::string *Table;
	int nb_cols;


	if (f_v) {
		cout << "table_of_surfaces::do_export "
				"before create_table" << endl;
	}
	create_table(Table, nb_cols, verbose_level);
	if (f_v) {
		cout << "table_of_surfaces::do_export "
				"after create_table" << endl;
	}


	if (f_v) {
		cout << "table_of_surfaces::do_export "
				"before export_csv" << endl;
	}

	export_csv(Table,
			nb_cols,
			verbose_level);

	if (f_v) {
		cout << "table_of_surfaces::do_export "
				"after export_csv" << endl;
	}


	if (f_v) {
		cout << "table_of_surfaces::do_export "
				"before export_sql" << endl;
	}

	export_sql(Table,
				nb_cols,
				verbose_level);

	if (f_v) {
		cout << "table_of_surfaces::do_export "
				"after export_sql" << endl;
	}

	delete [] Table;

	if (f_v) {
		cout << "table_of_surfaces::do_export done" << endl;
	}
}


void table_of_surfaces::create_table(std::string *&Table, int &nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_surfaces::create_table" << endl;
	}

	nb_cols = 22;

	Table = new string[nb_cubic_surfaces * nb_cols];

	int h;

	for (h = 0; h < nb_cubic_surfaces; h++) {

		if (f_v) {
			cout << "table_of_surfaces::create_table "
					<< h << " / " << nb_cubic_surfaces << endl;
		}

		Table[h * nb_cols + 0] = std::to_string(h);


		if (f_v) {
			cout << "collineation stabilizer order" << endl;
		}
		if (SC[h].f_has_group) {
			Table[h * nb_cols + 1] = SC[h].Sg->group_order_stringify();
		}

		if (f_v) {
			cout << "projectivity stabilizer order" << endl;
		}
		if (PA->A->is_semilinear_matrix_group()) {


			Table[h * nb_cols + 2] = SOG[h].projectivity_group_gens->group_order_stringify();
		}
		else {
			Table[h * nb_cols + 2] = SC[h].Sg->group_order_stringify();
		}

		Table[h * nb_cols + 3] = std::to_string(SC[h].SO->nb_pts);
		Table[h * nb_cols + 4] = std::to_string(SC[h].SO->nb_lines);
		Table[h * nb_cols + 5] = std::to_string(SC[h].SO->SOP->nb_Eckardt_points);
		Table[h * nb_cols + 6] = std::to_string(SC[h].SO->SOP->nb_Double_points);
		Table[h * nb_cols + 7] = std::to_string(SC[h].SO->SOP->nb_Single_points);
		Table[h * nb_cols + 8] = std::to_string(SC[h].SO->SOP->nb_pts_not_on_lines);
		Table[h * nb_cols + 9] = std::to_string(SC[h].SO->SOP->nb_Hesse_planes);
		Table[h * nb_cols + 10] = std::to_string(SC[h].SO->SOP->nb_axes);
		Table[h * nb_cols + 11] = std::to_string(SOG[h].Orbits_on_Eckardt_points->nb_orbits);
		Table[h * nb_cols + 12] = std::to_string(SOG[h].Orbits_on_Double_points->nb_orbits);
		Table[h * nb_cols + 13] = std::to_string(SOG[h].Orbits_on_points_not_on_lines->nb_orbits);
		Table[h * nb_cols + 14] = std::to_string(SOG[h].Orbits_on_lines->nb_orbits);
		Table[h * nb_cols + 15] = std::to_string(SOG[h].Orbits_on_single_sixes->nb_orbits);
		Table[h * nb_cols + 16] = std::to_string(SOG[h].Orbits_on_tritangent_planes->nb_orbits);
		Table[h * nb_cols + 17] = std::to_string(SOG[h].Orbits_on_Hesse_planes->nb_orbits);
		Table[h * nb_cols + 18] = std::to_string(SOG[h].Orbits_on_trihedral_pairs->nb_orbits);


		Table[h * nb_cols + 19] = "\"" + SC[h].SO->stringify_eqn() + "\"";

		Table[h * nb_cols + 20] = "\"" + SC[h].Surf->stringify_eqn_maple(SC[h].SO->eqn) + "\"";

		Table[h * nb_cols + 21] = "\"" + SC[h].SO->stringify_Lines() + "\"";


	}

	if (f_v) {
		cout << "table_of_surfaces::create_table done" << endl;
	}
}

void table_of_surfaces::export_csv(
		std::string *Table,
		int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_surfaces::export_csv" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	string fname;
	fname = "table_of_cubic_surfaces_q" + std::to_string(PA->F->q) + "_info.csv";

	{
		ofstream f(fname);
		int i, j;

		f << "Row,OCN,CollStabOrder,ProjStabOrder,nbPts,nbLines,"
				"nbE,nbDouble,nbSingle,nbPtsNotOn,nbHesse,nbAxes,"
				"nbOrbE,nbOrbDouble,nbOrbPtsNotOn,nbOrbLines,"
				"nbOrbSingleSix,nbOrbTriPlanes,nbOrbHesse,nbOrbTrihedralPairs,"
				"Eqn20,Equation,Lines";



		f << endl;
		for (i = 0; i < nb_cubic_surfaces; i++) {
			f << i;
			for (j = 0; j < nb_cols; j++) {
				f << "," << Table[i * nb_cols + j];
			}
			f << endl;
		}
		f << "END" << endl;
	}


	if (f_v) {
		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "table_of_surfaces::export_csv done" << endl;
	}
}

void table_of_surfaces::export_sql(
		std::string *Table,
		int nb_cols,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "table_of_surfaces::export_sql" << endl;
	}

	orbiter_kernel_system::file_io Fio;

	string fname;
	fname = "table_of_cubic_surfaces_q" + std::to_string(PA->F->q) + "_data.sql";


	{
		ofstream f(fname);
		int i;

		std::string *Row;

		for (i = 0; i < nb_cubic_surfaces; i++) {

			Row = Table + i * nb_cols;

			f << "UPDATE `cubicvt`.`surface` SET ";
			f << "`CollStabOrder` = '" << Row[1] << "', ";
			f << "`ProjStabOrder` = '" << Row[2] << "', ";
			f << "`nbPts` = '" << Row[3] << "', ";
			f << "`nbLines` = '" << Row[4] << "', ";
			f << "`nbE` = '" << Row[5] << "', ";
			f << "`nbDouble` = '" << Row[6] << "', ";
			f << "`nbSingle` = '" << Row[7] << "', ";
			f << "`nbPtsNotOn` = '" << Row[8] << "',";
			f << "`nbHesse` = '" << Row[9] << "', ";
			f << "`nbAxes` = '" << Row[10] << "', ";
			f << "`nbOrbE` = '" << Row[11] << "', ";
			f << "`nbOrbDouble` = '" << Row[12] << "', ";
			f << "`nbOrbPtsNotOn` = '" << Row[13] << "', ";
			f << "`nbOrbLines` = '" << Row[14] << "', ";
			f << "`nbOrbSingleSix` = '" << Row[15] << "', ";
			f << "`nbOrbTriPlanes` = '" << Row[16] << "', ";
			f << "`nbOrbHesse` = '" << Row[17] << "', ";
			f << "`nbOrbTrihedralPairs` = '" << Row[18] << "', ";
			f << "`Eqn20` = '" << Row[19] << "', ";
			f << "`Equation` = '$" << Row[20] << "$', ";
			f << "`Lines` = '" << Row[21] << "' ";
			f << "WHERE `Q` = '" << PA->F->q << "' AND `OCN` = '" << Row[0] << "';" << endl;
		}

	}


	if (f_v) {
		cout << "Written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	if (f_v) {
		cout << "table_of_surfaces::export_sql done" << endl;
	}
}


// from Oznur Oztunc, 11/28/2021:
// oznr83@gmail.com
//UPDATE `cubicvt`.`surface` SET `CollStabOrder` = '12', `ProjStabOrder` = '12', `nbPts` = '691', `nbLines` = '27', `nbE` = '4', `nbDouble` = '123', `nbSingle` = '390', `nbPtsNotOn` = '174',`nbHesse` = '0', `nbAxes` = '1', `nbOrbE` = '2', `nbOrbDouble` = '16', `nbOrbPtsNotOn` = '16', `nbOrbLines` = '5', `nbOrbSingleSix` = '10', `nbOrbTriPlanes` = '10', `nbOrbHesse` = '0', `nbOrbTrihedralPairs` = '19', `nbOrbTritangentPlanes` = '10',`Eqn20` = '0,0,0,0,0,0,8,0,10,0,0,18,0,2,0,0,18,10,2,1', `Equation` = '$8X_0^2*X_3+10X_1^2*X_2+18X_1*X_2^2+2X_0*X_3^2+18X_0*X_1*X_2+10X_0*X_1*X_3+2X_0*X_2*X_3+X_1*X_2*X_3$', `Lines` = '529,292560,1083,4965,290982,88471,169033,6600,8548,576,293089,0,3824,9119,1698,242212,12168,59424,229610,292854,242075,120504,179157,279048,30397,181283,12150' WHERE `Q` = '23' AND `OCN` = '1';



}}}}





