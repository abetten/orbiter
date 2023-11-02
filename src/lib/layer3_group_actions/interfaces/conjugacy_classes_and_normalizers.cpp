/*
 * conjugacy_classes_and_normalizers.cpp
 *
 *  Created on: Nov 2, 2023
 *      Author: betten
 */






#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace interfaces {


conjugacy_classes_and_normalizers::conjugacy_classes_and_normalizers()
{
	A = NULL;
	//std::string fname;
	nb_classes = 0;
	perms = NULL;
	class_size = NULL;
	class_order_of_element = NULL;
	class_normalizer_order = NULL;
	class_normalizer_number_of_generators = NULL;
	normalizer_generators_perms = NULL;

}

conjugacy_classes_and_normalizers::~conjugacy_classes_and_normalizers()
{
	if (perms) {
		FREE_int(perms);
	}
	if (class_size) {
		FREE_lint(class_size);
	}
	if (class_order_of_element) {
		FREE_int(class_order_of_element);
	}
	if (class_normalizer_order) {
		FREE_lint(class_normalizer_order);
	}
	if (class_normalizer_number_of_generators) {
		FREE_int(class_normalizer_number_of_generators);
	}
	if (normalizer_generators_perms) {
		FREE_pint(normalizer_generators_perms);
	}

}

void conjugacy_classes_and_normalizers::read_magma_output_file(
		actions::action *A,
		std::string &fname,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j, h;

	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file" << endl;
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
				"fname=" << fname << endl;
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
				"degree=" << A->degree << endl;
	}

	conjugacy_classes_and_normalizers::A = A;
	conjugacy_classes_and_normalizers::fname.assign(fname);

	{
		ifstream fp(fname);

		fp >> nb_classes;
		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"We found " << nb_classes
					<< " conjugacy classes" << endl;
		}

		perms = NEW_int(nb_classes * A->degree);
		class_size = NEW_lint(nb_classes);
		class_order_of_element = NEW_int(nb_classes);

		for (i = 0; i < nb_classes; i++) {
			fp >> class_order_of_element[i];
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class " << i << " / " << nb_classes
						<< " order=" << class_order_of_element[i] << endl;
			}
			fp >> class_size[i];
			if (f_v) {
				cout << "class_size[i] = " << class_size[i] << endl;
			}
			for (j = 0; j < A->degree; j++) {
				fp >> perms[i * A->degree + j];
			}
		}
		if (false) {
			cout << "perms:" << endl;
			Int_matrix_print(perms, nb_classes, A->degree);
		}
		for (i = 0; i < nb_classes * A->degree; i++) {
			perms[i]--;
		}

		class_normalizer_order = NEW_lint(nb_classes);
		class_normalizer_number_of_generators = NEW_int(nb_classes);
		normalizer_generators_perms = NEW_pint(nb_classes);

		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"reading normalizer generators:" << endl;
		}
		for (i = 0; i < nb_classes; i++) {
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class " << i << " / " << nb_classes << endl;
			}
			fp >> class_normalizer_order[i];

			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"class " << i << " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;

			if (class_normalizer_order[i] <= 0) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class_normalizer_order[i] <= 0" << endl;
				cout << "class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
				exit(1);
			}
			if (f_v) {
				cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
						"class " << i << " / " << nb_classes
						<< " class_normalizer_order[i]=" << class_normalizer_order[i] << endl;
			}
			fp >> class_normalizer_number_of_generators[i];
			normalizer_generators_perms[i] =
					NEW_int(class_normalizer_number_of_generators[i] * A->degree);
			for (h = 0; h < class_normalizer_number_of_generators[i]; h++) {
				for (j = 0; j < A->degree; j++) {
					fp >> normalizer_generators_perms[i][h * A->degree + j];
				}
			}
			for (h = 0; h < class_normalizer_number_of_generators[i] * A->degree; h++) {
				normalizer_generators_perms[i][h]--;
			}
		}
		if (f_v) {
			cout << "conjugacy_classes_and_normalizers::read_magma_output_file "
					"we read all class representatives "
					"from file " << fname << endl;
		}
	}
	if (f_v) {
		cout << "conjugacy_classes_and_normalizers::read_magma_output_file done" << endl;
	}
}


}}}


