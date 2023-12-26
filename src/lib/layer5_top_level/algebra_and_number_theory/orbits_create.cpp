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

	f_has_Orb = false;
	Orb = NULL;

	f_has_On_subsets = false;
	On_subsets = NULL;

	f_has_On_Subspaces = false;
	On_Subspaces = NULL;

	f_has_On_tensors = false;
	On_tensors = NULL;

	f_has_Cascade = false;
	Cascade = NULL;

	f_has_On_polynomials = false;
	On_polynomials = NULL;

	f_has_Of_One_polynomial = false;
	Of_One_polynomial = NULL;

	f_has_classification_by_canonical_form = false;
	Canonical_form_classifier = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

}


orbits_create::~orbits_create()
{
}

void orbits_create::init(
		apps_algebra::orbits_create_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	data_structures::string_tools ST;

	if (f_v) {
		cout << "orbits_create::init" << endl;
	}
	orbits_create::Descr = Descr;

	if (Descr->f_group) {

		Group = Get_object_of_type_any_group(Descr->group_label);
		prefix.assign(Group->label);
	}



	if (Descr->f_on_points) {

		if (f_v) {
			cout << "orbits_create::init f_on_points" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		//f_override_generators = false;
		//std::string override_generators_label;

		if (f_v) {
			cout << "group_theoretic_activity::perform_activity before AG->orbits_on_points" << endl;
		}

		Group->orbits_on_points(Orb, verbose_level);



		f_has_Orb = true;


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

			fname_tree_mask = "orbit_" + Group->A->label + "_%d.layered_graph";

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

	if (Descr->f_on_points_with_generators) {

		if (f_v) {
			cout << "orbits_create::init f_on_points_with_generators" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}


		apps_algebra::vector_ge_builder *Gens;

		Gens = Get_object_of_type_vector_ge(Descr->on_points_with_generators_gens_label);


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"before AG->orbits_on_points_from_generators" << endl;
		}

		Group->orbits_on_points_from_generators(Gens->V, Orb, verbose_level);


		f_has_Orb = true;


		if (f_v) {
			cout << "group_theoretic_activity::perform_activity "
					"after AG->orbits_on_points_from_generators" << endl;
		}
	}

	if (Descr->f_on_subsets) {

		if (f_v) {
			cout << "orbits_create::init f_on_subsets" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		poset_classification::poset_classification_control *Control =
				Get_object_of_type_poset_classification_control(Descr->on_subsets_poset_classification_control_label);


		if (f_v) {
			cout << "orbits_create::init before Group->orbits_on_subsets" << endl;
		}

		Group->orbits_on_subsets(Control, On_subsets,
				Descr->on_subsets_size, verbose_level);

		f_has_On_subsets = true;

		if (f_v) {
			cout << "orbits_create::init after Group->orbits_on_subsets" << endl;
		}

	}
	if (Descr->f_of_one_subset) {

		if (f_v) {
			cout << "orbits_create::init f_of_one_subset" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		long int *set;
		int sz;
		std::string label_set;

		Get_lint_vector_from_label(Descr->of_one_subset_label, set, sz, verbose_level - 2);

		label_set = Descr->of_one_subset_label;

		long int *Table;
		int size;

		if (f_v) {
			cout << "orbits_create::init before Group->orbits_of_one_subset" << endl;
		}
		Group->orbits_of_one_subset(
				set, sz,
				label_set,
				Group->A, Group->A,
				Table, size,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init after Group->orbits_of_one_subset" << endl;
		}


		if (f_v) {
			cout << "orbits_create::init after f_of_one_subset" << endl;
		}
	}

	if (Descr->f_on_subspaces) {

		if (f_v) {
			cout << "orbits_create::init f_on_subspaces" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
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

		f_has_On_Subspaces = true;


		if (f_v) {
			cout << "orbits_create::init after Group->do_orbits_on_subspaces" << endl;
		}

	}
	if (Descr->f_on_tensors) {

		if (f_v) {
			cout << "orbits_create::init f_on_tensors" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}


		if (f_v) {
			cout << "orbits_create::init before Group->do_tensor_classify" << endl;
		}

		Group->do_tensor_classify(
				Descr->on_tensors_poset_classification_control_label,
				On_tensors,
				Descr->on_tensors_dimension,
				verbose_level);

		f_has_On_tensors = true;

		if (f_v) {
			cout << "orbits_create::init after Group->do_tensor_classify" << endl;
		}

	}
	if (Descr->f_on_partition) {

		if (f_v) {
			cout << "orbits_create::init f_on_partition" << endl;
		}
		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
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

		f_has_Cascade = true;

		if (f_v) {
			cout << "orbits_create::init after Cascade->init" << endl;
		}

	}


	if (Descr->f_on_polynomials) {


		if (f_v) {
			cout << "orbits_create::init f_on_polynomials" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init ring = " << Descr->on_polynomials_ring << endl;
		}

		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		if (!Group->f_linear_group) {
			cout << "orbits_create::init group must be linear" << endl;
			exit(1);
		}

		On_polynomials = NEW_OBJECT(orbits_on_polynomials);



		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->on_polynomials_ring);

		if (f_v) {
			cout << "orbits_create::init "
					"before On_polynomials->init" << endl;
		}
		On_polynomials->init(
				Group->LG,
				HPD,
				verbose_level);

		if (f_v) {
			cout << "orbits_create::init "
					"after On_polynomials->init" << endl;
		}

		f_has_On_polynomials = true;




	}


	if (Descr->f_of_one_polynomial) {


		if (f_v) {
			cout << "orbits_create::init f_of_one_polynomial" << endl;
		}
		if (f_v) {
			cout << "orbits_create::init ring = " << Descr->of_one_polynomial_ring << endl;
		}

		if (!Descr->f_group) {
			cout << "orbits_create::init please specify the group using -group <label>" << endl;
			exit(1);
		}

		if (!Group->f_linear_group) {
			cout << "orbits_create::init group must be linear" << endl;
			exit(1);
		}

		Of_One_polynomial = NEW_OBJECT(orbits_on_polynomials);



		ring_theory::homogeneous_polynomial_domain *HPD;


		HPD = Get_ring(Descr->of_one_polynomial_ring);


		expression_parser::symbolic_object_builder *Symbol;

		Symbol = Get_symbol(Descr->of_one_polynomial_equation);

		if (f_v) {
			cout << "orbits_create::init "
					"before Of_One_polynomial->orbit_of_one_polynomial" << endl;
		}
		Of_One_polynomial->orbit_of_one_polynomial(
				Group->LG,
				HPD,
				Symbol,
				verbose_level);
		if (f_v) {
			cout << "orbits_create::init "
					"after Of_One_polynomial->orbit_of_one_polynomial" << endl;
		}

		f_has_Of_One_polynomial = true;

	}




	if (Descr->f_classification_by_canonical_form) {


		if (f_v) {
			cout << "orbits_create::init f_classification_by_canonical_form" << endl;
		}



		if (!Descr->Canonical_form_classifier_description->f_output_fname) {
			cout << "Please specify the output file name using -output_fname <fname>" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "orbits_create::init getting projective space "
					<< Descr->Canonical_form_classifier_description->space_label << endl;
		}

#if 0
		Descr->Canonical_form_classifier_description->PA =
				Get_object_of_projective_space(
						Descr->Canonical_form_classifier_description->space_label);
#endif

		if (Descr->Canonical_form_classifier_description->f_algorithm_substructure) {

			if (f_v) {
				cout << "orbits_create::init f_algorithm_substructure" << endl;
			}

			Canonical_form_classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);

			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"before Canonical_form_classifier->init" << endl;
			}
			Canonical_form_classifier->init(
					Descr->Canonical_form_classifier_description,
					verbose_level);
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"after Canonical_form_classifier->init" << endl;
			}
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"before Classifier.classify" << endl;
			}
			Canonical_form_classifier->classify(verbose_level);
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"after Classifier.classify" << endl;
			}

			Descr->Canonical_form_classifier_description->Canon_substructure = Canonical_form_classifier;

#if 0
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"before Classifier.report" << endl;
			}
			Canonical_form_classifier->report(
					Descr->Canonical_form_classifier_description->fname_base_out,
					verbose_level);
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"after Classifier.report" << endl;
			}
#endif

			f_has_classification_by_canonical_form = true;

		}
		else if (Descr->Canonical_form_classifier_description->f_algorithm_nauty) {

			if (f_v) {
				cout << "orbits_create::init f_algorithm_nauty" << endl;
			}

			Canonical_form_classifier = NEW_OBJECT(canonical_form::canonical_form_classifier);

			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"before Canonical_form_classifier->init" << endl;
			}
			Canonical_form_classifier->init(
					Descr->Canonical_form_classifier_description,
					verbose_level);
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"after Canonical_form_classifier->init" << endl;
			}
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"before Classifier.classify" << endl;
			}
			Canonical_form_classifier->classify(verbose_level);
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"after Classifier.classify" << endl;
			}

			Descr->Canonical_form_classifier_description->Canon_substructure = Canonical_form_classifier;

#if 0
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"before Classifier.report" << endl;
			}
			Canonical_form_classifier->report(
					Descr->Canonical_form_classifier_description->fname_base_out,
					verbose_level);
			if (f_v) {
				cout << "projective_space_global::classify_quartic_curves_with_substructure "
						"after Classifier.report" << endl;
			}
#endif

			f_has_classification_by_canonical_form = true;

		}
		else {
			cout << "orbits_create::init please specify which algorithm should be used" << endl;
			exit(1);
		}

		if (f_v) {
			cout << "orbits_create::init f_classification_by_canonical_form done" << endl;
		}

	}


	if (f_v) {
		cout << "orbits_create::init done" << endl;
	}
}


}}}





