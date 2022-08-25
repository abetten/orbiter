// spread_create.cpp
// 
// Anton Betten
//
// March 22, 2018
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace spreads {


spread_create::spread_create()
{
	Descr = NULL;

	//std::string prefix;
	//std::string label_txt;
	//std::string label_tex;

	G = NULL;

	q = 0;
	F = NULL;
	k = 0;

	f_semilinear = FALSE;

	A = NULL;
	degree = 0;

	set = NULL;
	sz = 0;

	f_has_group = FALSE;
	Sg = NULL;
}


spread_create::~spread_create()
{
#if 0
	if (F) {
		FREE_OBJECT(F);
	}
#endif
	if (set) {
		FREE_lint(set);
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
}


void spread_create::init(spread_create_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	number_theory::number_theory_domain NT;

	
	if (f_v) {
		cout << "spread_create::init" << endl;
	}
	spread_create::Descr = Descr;
	if (!Descr->f_kernel_field) {
		cout << "spread_create::init !Descr->f_kernel_field" << endl;
		exit(1);
	}

	F = Get_object_of_type_finite_field(Descr->kernel_field_label);

	q = F->q;

	if (!Descr->f_k) {
		cout << "spread_create::init !Descr->f_k" << endl;
		exit(1);
	}
	k = Descr->k;

	if (!Descr->f_group) {
		cout << "spread_create::init !Descr->f_group" << endl;
		exit(1);
	}

	G = Get_object_of_type_any_group(Descr->group_label);

	if (!G->f_linear_group) {
		cout << "spread_create::init the group must be a linear group" << endl;
		exit(1);
	}

	A = G->A_base;
	if (!A->is_matrix_group()) {
		cout << "spread_create::init the base group is not a matrix group" << endl;
		exit(1);
	}
	

	f_semilinear = A->is_semilinear_matrix_group();


	if (f_v) {
		cout << "spread_create::init q = " << q << endl;
		cout << "spread_create::init k = " << k << endl;
		cout << "spread_create::init f_semilinear = " << f_semilinear << endl;
		cout << "spread_create::init A->matrix_group_dimension() = " << A->matrix_group_dimension() << endl;
	}

	if (A->matrix_group_dimension() != 2 * k) {
		cout << "spread_create::init dimension of the matrix group must be 2 * k" << endl;
		exit(1);
	}


	if (Descr->f_family) {
		if (f_v) {
			cout << "spread_create::init "
					"before Surf->create_surface_family "
					"family_name=" << Descr->family_name << endl;
		}


	}



	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "spread_create::init "
					"spread from catalogue" << endl;
		}
		int nb_iso;
		knowledge_base K;

		nb_iso = K.Spread_nb_reps(q, k);
		if (Descr->iso >= nb_iso) {
			cout << "spread_create::init "
					"iso >= nb_iso, this spread does not exist" << endl;
			exit(1);
		}

		long int *rep;

		rep = K.Spread_representative(q, k, Descr->iso, sz);
		set = NEW_lint(sz);
		Lint_vec_copy(rep, set, sz);

		Sg = NEW_OBJECT(groups::strong_generators);

		if (f_v) {
			cout << "spread_create::init "
					"before Sg->stabilizer_of_spread_from_catalogue" << endl;
		}

		Sg->stabilizer_of_spread_from_catalogue(A, 
			q, k, Descr->iso, 
			verbose_level);
		f_has_group = TRUE;

		char str[1000];

		sprintf(str, "catalogue_q%d_k%d_%d", q, k, Descr->iso);
		prefix.assign(str);
		sprintf(str, "catalogue_q%d_k%d_%d", q, k, Descr->iso);
		label_txt.assign(str);
		sprintf(str, "catalogue\\_q%d\\_k%d\\_%d", q, k, Descr->iso);
		label_tex.assign(str);
		if (f_v) {
			cout << "spread_create::init "
					"after Sg->stabilizer_of_spread_from_catalogue" << endl;
		}
	}
	else if (Descr->f_spread_set) {

		if (f_v) {
			cout << "spread_create::init "
					"spread from spread set, label = " << Descr->spread_set_label << endl;
		}
		long int *spread_set_matrices;
		int sz;

		Get_vector_or_set(Descr->spread_set_label, spread_set_matrices, sz);
		if (f_v) {
			int k2;

			k2 = Descr->k * Descr->k;

			cout << "spread_create::init spread_set_matrices sz = " << sz << endl;
			Lint_matrix_print(set, sz / k2, k2);
			cout << "spread_create::init spread_set_matrices = " << endl;
			Lint_matrix_print(set, sz / k2, k2);
		}

		exit(1);
	}
	else {
		cout << "spread_create::init we do not "
				"recognize the type of spread" << endl;
		exit(1);
	}


	if (f_v) {
		cout << "spread_create::init set = ";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}

	if (f_has_group) {
		cout << "spread_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
	}



	if (f_v) {
		cout << "spread_create::init done" << endl;
	}
}

void spread_create::apply_transformations(
		std::vector<std::string> transform_coeffs,
		std::vector<int> f_inverse_transform, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "spread_create::apply_transformations not yet implemented" << endl;
	}
}

}}}

