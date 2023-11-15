/*
 * action_on_module.cpp
 *
 *  Created on: Mar 2, 2023
 *      Author: betten
 */


#include "layer1_foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace induced_actions {


action_on_module::action_on_module()
{
	A = NULL;
	n = q = 0;
	M = NULL;
	F = NULL;
	low_level_point_size = 0;

	SO = NULL;
	module_basis = NULL;
	module_dimension_m = 0;
	module_dimension_n = 0;

	module_basis_base_transposed = NULL;

	module_basis_base_cols = NULL;
	module_basis_rref = NULL;
	module_basis_transformation = NULL;

	v1 = NULL;
	v2 = NULL;

	A_on_the_lines = NULL;
	A_on_module = NULL;

}

action_on_module::~action_on_module()
{
	A = NULL;

	if (module_basis) {
		FREE_int(module_basis);
	}
	if (module_basis_base_transposed) {
		delete [] module_basis_base_transposed;
	}
#if 0
	if (module_basis_base_cols) {
		FREE_int(module_basis_base_cols);
	}
	if (module_basis_rref) {
		FREE_int(module_basis_rref);
	}
	if (module_basis_transformation) {
		FREE_int(module_basis_transformation);
	}
#endif

	if (v1) {
		FREE_int(v1);
	}
	if (v2) {
		FREE_int(v2);
	}
}

void action_on_module::init_action_on_module(
		algebraic_geometry::surface_object *SO,
		actions::action *A_on_the_lines,
		std::string &module_type,
		int *module_basis, int module_dimension_m, int module_dimension_n,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "action_on_module::init_action_on_module" << endl;
	}

	action_on_module::SO = SO;
	action_on_module::A_on_the_lines = A_on_the_lines;
	action_on_module::module_dimension_m = module_dimension_m;
	action_on_module::module_dimension_n = module_dimension_n;

	low_level_point_size = module_dimension_m;

	action_on_module::module_basis = NEW_int(module_dimension_m * module_dimension_n);
	Int_vec_copy(module_basis, action_on_module::module_basis, module_dimension_m * module_dimension_n);

	module_basis_base_transposed = new double [module_dimension_n * module_dimension_m];

#if 0
	module_basis_base_cols = NEW_int(45);
	module_basis_rref = NEW_int(module_dimension * 45);
	module_basis_transformation = NEW_int(module_dimension * module_dimension);
#endif

	v1 = NEW_int(module_dimension_n);
	v2 = NEW_int(module_dimension_n);

	int i, j;

	for (i = 0; i < module_dimension_n; i++) {
		for (j = 0; j < module_dimension_m; j++) {

			module_basis_base_transposed[i * module_dimension_m + j] = module_basis[j * module_dimension_n + i];
		}
	}



#if 0

	Int_vec_copy(module_basis, module_basis_rref, module_dimension * 45);

	int f_special = false;
	int f_complete = true;
	int f_P = true;
	int rk;


	F->Linear_algebra->identity_matrix(module_basis_transformation, module_dimension);



	if (f_v) {
		cout << "action_on_module::init_action_on_module before F->Linear_algebra->Gauss_int" << endl;
	}

	rk = F->Linear_algebra->Gauss_int(
			module_basis_rref,
			f_special, f_complete, module_basis_base_cols,
			f_P, module_basis_transformation /* P */, module_dimension, 45, module_dimension,
			verbose_level);
	// returns the rank which is the number of entries in base_cols
	// A is a m x n matrix,
	// P is a m x Pn matrix (if f_P is true)

	if (f_v) {
		cout << "action_on_module::init_action_on_module after F->Linear_algebra->Gauss_int" << endl;
	}
	if (f_v) {
		cout << "action_on_module::init_action_on_module rk = " << rk << endl;
	}
	if (rk != module_dimension) {
		cout << "action_on_module::init_action_on_module rk != module_dimension" << endl;
		exit(1);
	}
	if (f_v) {
		cout << "action_on_module::init_action_on_module module_basis_rref=" << endl;
		Int_matrix_print(module_basis_rref, module_dimension, 45);
	}
#endif



	if (strcmp(module_type.c_str(), "on_tritangent_planes") == 0) {
		A_on_module =
				A_on_the_lines->Induced_action->create_induced_action_on_sets(
				SO->SOP->SmoothProperties->nb_tritangent_planes, 3,
				SO->Surf->Schlaefli->Schlaefli_tritangent_planes->Lines_in_tritangent_planes,
				0 /*verbose_level*/);
		if (module_dimension_n != 45) {
			cout << "module_dimension_n should be 45" << endl;
			exit(1);

		}
	}
	else if (strcmp(module_type.c_str(), "on_double_sixes") == 0) {


		actions::action *A_single_sixes;

		if (f_v) {
			cout << "creating action on half double sixes:" << endl;
		}
		A_single_sixes = A_on_the_lines->Induced_action->create_induced_action_on_sets(
				72, 6, SO->Surf->Schlaefli->Schlaefli_double_six->Double_six,
				0 /*verbose_level*/);
		if (f_v) {
			cout << "creating action on half double sixes done" << endl;
		}

		long int double_six_sets[72];
		int i, j;

		for (i = 0; i < 36; i++) {
			for (j = 0; j < 2; j++) {
				double_six_sets[i * 2 + j] = i * 2 + j;
			}
		}

		if (f_v) {
			cout << "creating action on half double sixes:" << endl;
		}
		A_on_module = A_single_sixes->Induced_action->create_induced_action_on_sets(
				36, 2, double_six_sets,
				0 /*verbose_level*/);

		if (module_dimension_n != 36) {
			cout << "module_dimension_n should be 36" << endl;
			exit(1);

		}
	}
	else if (strcmp(module_type.c_str(), "on_lines") == 0) {
		A_on_module = A_on_the_lines;
		if (module_dimension_n != 27) {
			cout << "module_dimension_n should be 27" << endl;
			exit(1);

		}
	}

	if (f_v) {
		cout << "action_on_module::init_action_on_module done" << endl;
	}
}


void action_on_module::compute_image_int_low_level(
		int *Elt, int *input, int *output, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	if (f_v) {
		cout << "action_on_module::compute_image_int_low_level" << endl;
	}
	if (f_vv) {
		cout << "action_on_module::compute_image_int_low_level: input=";
		Int_vec_print(cout, input, low_level_point_size);
		cout << endl;
	}

	linear_algebra::module Module;

	//int m3, n3;

	//m3 = 1;
	//n3 = 45;
	if (f_vv) {
		cout << "action_on_module::compute_image_int_low_level input=";
		Int_vec_print(cout, input, module_dimension_m);
		cout << endl;
	}


	Module.matrix_multiply_over_Z_low_level(
			input, module_basis, 1, module_dimension_m, module_dimension_m, module_dimension_n,
			v1, verbose_level - 2);

	if (f_vv) {
		cout << "action_on_module::compute_image_int_low_level v1=";
		Int_vec_print(cout, v1, module_dimension_n);
		cout << endl;
	}

	long int i, j;
	int a;

	for (i = 0; i < module_dimension_n; i++) {
		a = v1[i];
		j = A_on_module->Group_element->element_image_of(
				i, Elt, 0 /*verbose_level*/);
		v2[j] = a;
	}
	if (f_vv) {
		cout << "action_on_module::compute_image_int_low_level v2=";
		Int_vec_print(cout, v2, module_dimension_n);
		cout << endl;
	}

#if 1
	double *D;

	D = new double [module_dimension_n * (module_dimension_m + 1)];

	for (i = 0; i < module_dimension_n; i++) {
		for (j = 0; j < module_dimension_m; j++) {

			D[i * (module_dimension_m + 1) + j] = module_basis[j * module_dimension_m + i];
		}
		D[i * (module_dimension_m + 1) + module_dimension_m] = v2[i];
	}




	orbiter_kernel_system::numerics Num;
	int *base_cols;
	int f_complete = true;
	int r;

	base_cols = NEW_int(module_dimension_m + 1);

	r = Num.Gauss_elimination(
				D, module_dimension_n, module_dimension_m + 1,
			base_cols, f_complete,
			verbose_level - 5);

	if (r != module_dimension_m) {
		cout << "something is wrong, r = " << r << endl;
		cout << "should be = " << module_dimension_m << endl;
		exit(1);
	}

	int kernel_m, kernel_n;
	double *kernel;

	kernel = new double [module_dimension_n * (module_dimension_m + 1)];

	Num.get_kernel(D, module_dimension_n, module_dimension_m + 1,
		base_cols, r /* nb_base_cols */,
		kernel_m, kernel_n,
		kernel);

	cout << "kernel_m = " << kernel_m << endl;
	cout << "kernel_n = " << kernel_n << endl;

	if (kernel_m != module_dimension_m + 1)	{
		cout << "kernel_m != module_dimension_m + 1" << endl;
		exit(1);
	}
	if (kernel_n != 1)	{
		cout << "kernel_n != 1" << endl;
		exit(1);
	}
	double d, dv;

	d = kernel[module_dimension_m];

	if (ABS(d) < 0.001) {
		cout << "ABS(d) < 0.001" << endl;
		exit(1);
	}
	dv = -1. / d;
	for (i = 0; i < module_dimension_m + 1; i++) {
		kernel[i] *= dv;
	}

	for (i = 0; i < module_dimension_m; i++) {
		output[i] = kernel[i];
	}

	FREE_int(base_cols);

	if (f_vv) {
		cout << "action_on_module::compute_image_int_low_level output=";
		Int_vec_print(cout, output, low_level_point_size);
		cout << endl;
	}
#endif

	if (f_v) {
		cout << "action_on_module::compute_image_int_low_level done" << endl;
	}
}




}}}



