/*
 * combo_activity.cpp
 *
 *  Created on: Jan 3, 2025
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_combinatorics {



combo_activity::combo_activity()
{
	Record_birth();
	Descr = NULL;

	pOwP = NULL;
	nb_objects = 0;

	nb_output = 0;
	Output = NULL;

}

combo_activity::~combo_activity()
{
	Record_death();
}


void combo_activity::init(
		combo_activity_description *Descr,
		canonical_form::combinatorial_object_with_properties **pOwP,
		int nb_objects,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combo_activity::init" << endl;
	}

	combo_activity::Descr = Descr;
	combo_activity::pOwP = pOwP;
	combo_activity::nb_objects = nb_objects;

	if (f_v) {
		cout << "combo_activity::init done" << endl;
	}
}



void combo_activity::perform_activity(
		other::orbiter_kernel_system::activity_output *&AO,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "combo_activity::perform_activity" << endl;
		cout << "combo_activity::perform_activity verbose_level = " << verbose_level << endl;
	}


	if (Descr->f_report) {
		if (f_v) {
			cout << "combo_activity::perform_activity_combo "
					"f_report" << endl;
		}

		int i;

		for (i = 0; i < nb_objects; i++) {

			if (f_v) {
				cout << "combo_activity::perform_activity_combo "
						"before OwP->latex_report_wrapper" << endl;
			}
			pOwP[i]->latex_report_wrapper(
					pOwP[i]->label,
					Descr->Objects_report_options,
					verbose_level);


			if (f_v) {
				cout << "combo_activity::perform_activity_combo "
						"after OwP->latex_report_wrapper" << endl;
			}
		}
	}

	else if (Descr->f_get_group) {
		if (f_v) {
			cout << "combo_activity::perform_activity_combo "
					"f_get_group" << endl;
		}

		if (nb_objects != 1) {
			cout << "combo_activity::perform_activity_combo "
					"f_get_group, nb_objects != 1" << endl;
			exit(1);
		}

#if 0
		if (f_v) {
			cout << "combo_activity::perform_activity_combo "
					"before OwP->latex_report_wrapper" << endl;
		}
		pOwP[0]->latex_report_wrapper(
				pOwP[0]->label,
				Descr->Objects_report_options,
				verbose_level);



		if (f_v) {
			cout << "combo_activity::perform_activity_combo "
					"after OwP->latex_report_wrapper" << endl;
		}
#endif


		nb_output = 1;
		Output = NEW_OBJECT(other::orbiter_kernel_system::orbiter_symbol_table_entry);

		string output_label;
		string output_label_tex;



		output_label = pOwP[0]->label + "_group";
		output_label_tex = pOwP[0]->label_tex + "\\_group";


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"output_label = " << output_label << endl;
		}

		groups::any_group *Any_group;


		Any_group = NEW_OBJECT(groups::any_group);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"before Any_group->init_perm_group_direct" << endl;
		}


		Any_group->init_perm_group_direct(
				pOwP[0]->A_perm,
				output_label, output_label_tex,
				verbose_level);

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"after Any_group->init_perm_group_direct" << endl;
		}

		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"before Output->init_any_group" << endl;
		}

		Output->init_any_group(
				output_label,
				Any_group, verbose_level);


		if (f_v) {
			cout << "combinatorial_object_activity::perform_activity "
					"after Output->init_any_group done" << endl;
		}

	}



	if (f_v) {
		cout << "combo_activity::perform_activity done" << endl;
	}
}


#if 0
void compute_TDO(
		int max_TDO_depth, int verbose_level);
#endif






}}}




