/*
 * orbits_create.cpp
 *
 *  Created on: Nov 5, 2022
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_algebra {


orbits_create::orbits_create()
{
	Descr = NULL;

	Group = NULL;

	f_has_Orb = FALSE;
	Orb = NULL;

	f_has_On_subsets = FALSE;
	On_subsets = NULL;

	f_has_On_Subspaces = FALSE;
	On_Subspaces = NULL;

	f_has_On_tensors = FALSE;
	On_tensors = NULL;

	f_has_Cascade = FALSE;
	Cascade = NULL;

	f_has_On_polynomials = FALSE;
	On_polynomials = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

}


orbits_create::~orbits_create()
{
}

void orbits_create::init(apps_algebra::orbits_create_description *Descr, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "orbits_create::init" << endl;
	}
	orbits_create::Descr = Descr;

	if (Descr->f_group) {

		Group = Get_object_of_type_any_group(Descr->group_label);
	}
	else {
		cout << "orbits_create::init please specify the group using -group <label>" << endl;
		exit(1);
	}


	prefix.assign(Group->label);

	if (Descr->f_on_points) {

		if (f_v) {
			cout << "orbits_create::init f_on_points" << endl;
		}


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->orbits_on_points" << endl;
		}

		Group->orbits_on_points(Orb, verbose_level);

		f_has_Orb = TRUE;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity after AG->orbits_on_points" << endl;
		}

#if 0
		if (Descr->f_stabilizer) {

			int orbit_idx = 0;

			if (f_v) {
				cout << "group_theoretic_activity::perform_activity before Orb->stabilizer_of" << endl;
			}
			Orb->stabilizer_of(orbit_idx, verbose_level);
			if (f_v) {
				cout << "group_theoretic_activity::perform_activity after Orb->stabilizer_of" << endl;
			}
		}

		if (Descr->f_stabilizer_of_orbit_rep) {

			if (f_v) {
				cout << "group_theoretic_activity::perform_activity f_stabilizer_of_orbit_rep" << endl;
			}
			if (f_v) {
				cout << "group_theoretic_activity::perform_activity f_stabilizer_of_orbit_rep "
						"stabilizer_of_orbit_rep_orbit_idx=" << Descr->stabilizer_of_orbit_rep_orbit_idx << endl;
			}

			int orbit_idx = Descr->stabilizer_of_orbit_rep_orbit_idx;

			if (f_v) {
				cout << "group_theoretic_activity::perform_activity f_stabilizer_of_orbit_rep "
						"orbit_idx=" << orbit_idx << endl;
			}

			if (f_v) {
				cout << "group_theoretic_activity::perform_activity before Orb->stabilizer_of" << endl;
			}
			Orb->stabilizer_of(orbit_idx, verbose_level);
			if (f_v) {
				cout << "group_theoretic_activity::perform_activity after Orb->stabilizer_of" << endl;
			}
		}


		if (Descr->f_report) {

			Orb->create_latex_report(verbose_level);

		}

		if (Descr->f_export_trees) {

			string fname_tree_mask;
			int orbit_idx;

			fname_tree_mask.assign("orbit_");
			fname_tree_mask.append(Group->A->label);
			fname_tree_mask.append("_%d.layered_graph");

			for (orbit_idx = 0; orbit_idx < Orb->Sch->nb_orbits; orbit_idx++) {

				cout << "orbit " << orbit_idx << " / " <<  Orb->Sch->nb_orbits
						<< " before Sch->export_tree_as_layered_graph" << endl;

				Orb->Sch->export_tree_as_layered_graph(orbit_idx,
						fname_tree_mask,
						verbose_level - 1);
			}
		}
#endif

		//FREE_OBJECT(Orb);

	}

	if (Descr->f_on_subsets) {

		if (f_v) {
			cout << "orbits_create::init f_on_subsets" << endl;
		}

		poset_classification::poset_classification_control *Control =
				Get_object_of_type_poset_classification_control(Descr->on_subsets_poset_classification_control_label);


		if (f_v) {
			cout << "orbits_create::init before Group->orbits_on_subsets" << endl;
		}

		Group->orbits_on_subsets(Control, On_subsets,
				Descr->on_subsets_size, verbose_level);

		f_has_On_subsets = TRUE;

		if (f_v) {
			cout << "orbits_create::init after Group->orbits_on_subsets" << endl;
		}

	}
	if (Descr->f_on_subspaces) {

		if (f_v) {
			cout << "orbits_create::init f_on_subspaces" << endl;
		}

		poset_classification::poset_classification_control *Control =
				Get_object_of_type_poset_classification_control(Descr->on_subspaces_poset_classification_control_label);


		if (f_v) {
			cout << "orbits_create::init before Group->do_orbits_on_subspaces" << endl;
		}

		Group->do_orbits_on_subspaces(Control,
				On_Subspaces,
				Descr->on_subspaces_dimension,
				verbose_level);

		if (f_v) {
			cout << "orbits_create::init after Group->do_orbits_on_subspaces" << endl;
		}

		f_has_On_Subspaces = TRUE;


		if (f_v) {
			cout << "orbits_create::init after Group->do_orbits_on_subspaces" << endl;
		}

	}
	if (Descr->f_on_tensors) {

		if (f_v) {
			cout << "orbits_create::init f_on_tensors" << endl;
		}


		if (f_v) {
			cout << "orbits_create::init before Group->do_tensor_classify" << endl;
		}

		Group->do_tensor_classify(
				Descr->on_tensors_poset_classification_control_label,
				On_tensors,
				Descr->on_tensors_dimension,
				verbose_level);

		f_has_On_tensors = TRUE;

		if (f_v) {
			cout << "orbits_create::init after Group->do_tensor_classify" << endl;
		}

	}
	if (Descr->f_on_partition) {

		if (f_v) {
			cout << "orbits_create::init f_on_partition" << endl;
		}



		Cascade = NEW_OBJECT(orbit_cascade);


		if (f_v) {
			cout << "orbits_create::init before Cascade->init" << endl;
		}

		Cascade->init(Group->A->degree,
				Descr->on_partition_k,
				Group,
				Descr->on_partition_poset_classification_control_label,
				verbose_level);

		f_has_Cascade = TRUE;

		if (f_v) {
			cout << "orbits_create::init after Cascade->init" << endl;
		}

	}


	if (Descr->f_on_polynomials) {


		if (f_v) {
			cout << "orbits_create::init f_on_polynomials" << endl;
		}

		if (!Group->f_linear_group) {
			cout << "orbits_create::init group must be linear" << endl;
			exit(1);
		}

		On_polynomials = NEW_OBJECT(orbits_on_polynomials);

		if (f_v) {
			cout << "orbits_create::init before On_polynomials->init" << endl;
		}
		On_polynomials->init(Group->LG,
				Descr->on_polynomials_degree,
				//Descr->f_recognize, Descr->recognize_text,
				verbose_level);

		if (f_v) {
			cout << "orbits_create::init after On_polynomials->init" << endl;
		}

		f_has_On_polynomials = TRUE;




	}




	if (f_v) {
		cout << "orbits_create::init done" << endl;
	}
}


}}}





