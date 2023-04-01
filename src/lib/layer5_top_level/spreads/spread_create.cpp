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
	G_on_subspaces = NULL;

	q = 0;
	F = NULL;
	k = 0;

	f_semilinear = FALSE;

	A = NULL;
	degree = 0;

	Grass = NULL;

	set = NULL;
	sz = 0;

	f_has_group = FALSE;
	Sg = NULL;

	Andre = NULL;
}


spread_create::~spread_create()
{
#if 0
	if (F) {
		FREE_OBJECT(F);
	}
#endif
	if (Grass) {
		FREE_OBJECT(Grass);
	}
	if (set) {
		FREE_lint(set);
	}
	if (Sg) {
		FREE_OBJECT(Sg);
	}
	if (Andre) {
		FREE_OBJECT(Andre);
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

	F = Get_finite_field(Descr->kernel_field_label);

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


	if (Descr->f_group_on_subspaces) {
		G_on_subspaces = Get_object_of_type_any_group(Descr->group_on_subspaces_label);
	}

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

	Grass = NEW_OBJECT(geometry::grassmann);
	Grass->init(2 * k, k, F, verbose_level);


	if (Descr->f_family) {
		if (f_v) {
			cout << "spread_create::init family not yet implemented "
					"family_name=" << Descr->family_name << endl;
		}
		exit(1);
	}



	else if (Descr->f_catalogue) {

		if (f_v) {
			cout << "spread_create::init "
					"spread from catalogue" << endl;
		}
		int nb_iso;
		knowledge_base::knowledge_base K;

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
		if (f_v) {
			cout << "spread_create::init "
					"after Sg->stabilizer_of_spread_from_catalogue" << endl;
		}

		f_has_group = TRUE;

		char str[1000];

		snprintf(str, sizeof(str), "catalogue_q%d_k%d_%d", q, k, Descr->iso);
		prefix.assign(str);
		snprintf(str, sizeof(str), "catalogue_q%d_k%d_%d", q, k, Descr->iso);
		label_txt.assign(str);
		snprintf(str, sizeof(str), "catalogue\\_q%d\\_k%d\\_%d", q, k, Descr->iso);
		label_tex.assign(str);
	}

	else if (Descr->f_spread_set) {

		if (f_v) {
			cout << "spread_create::init "
					"spread from spread set, label = " << Descr->spread_set_label << endl;
		}
		long int *spread_set_matrices;
		int spread_set_matrices_sz;
		int k2;

		k2 = Descr->k * Descr->k;

		Get_vector_or_set(Descr->spread_set_data, spread_set_matrices, spread_set_matrices_sz);
		if (f_v) {

			cout << "spread_create::init spread_set_matrices "
					"spread_set_matrices_sz = " << spread_set_matrices_sz << endl;
			Lint_matrix_print(spread_set_matrices, spread_set_matrices_sz / k2, k2);
			cout << "spread_create::init spread_set_matrices = " << endl;
			Lint_matrix_print(spread_set_matrices, spread_set_matrices_sz / k2, k2);
		}

		if (f_v) {
			cout << "spread_create::init before Grass->make_spread_from_spread_set" << endl;
		}
		Grass->make_spread_from_spread_set(
				spread_set_matrices, spread_set_matrices_sz / k2,
				set, sz,
				verbose_level);
		if (f_v) {
			cout << "spread_create::init after Grass->make_spread_from_spread_set, sz = " << sz << endl;
		}
		prefix.assign(Descr->spread_set_label);
		label_txt.assign(Descr->spread_set_label);
		label_tex.assign(Descr->spread_set_label_tex);

		//exit(1);
	}

	else {
		cout << "spread_create::init we do not "
				"recognize the type of spread" << endl;
		exit(1);
	}



	if (Descr->f_transform) {
		int h;

		if (!Descr->f_group_on_subspaces) {
			cout << "-transform needs -group_on_subspaces" << endl;
			exit(1);
		}

		int *Elt1;
		int *Elt2;
		//int *Elt3;
		long int *image_set = NULL;

		Elt1 = NEW_int(G->A_base->elt_size_in_int);
		Elt2 = NEW_int(G->A_base->elt_size_in_int);
		//Elt3 = NEW_int(G->A_base->elt_size_in_int);
		image_set = NEW_lint(sz);

		for (h = 0; h < Descr->transform_text.size(); h++) {
			if (Descr->transform_f_inv[h]) {
				cout << "-transform_inv " << Descr->transform_text[h] << endl;
			}
			else {
				cout << "-transform " << Descr->transform_text[h] << endl;
			}

			int *transformation_coeffs;
			int transformation_coeffs_sz;
			//int coeffs_out[20];

			if (f_v) {
				cout << "spread_create::init "
						"applying transformation " << h << " / "
						<< Descr->transform_text.size() << ":" << endl;
			}

			Int_vec_scan(Descr->transform_text[h], transformation_coeffs, transformation_coeffs_sz);

			if (transformation_coeffs_sz != G->A_base->make_element_size) {
				cout << "spread_create::init "
						"need exactly " << G->A_base->make_element_size
						<< " coefficients for the transformation" << endl;
				cout << "Descr->transform_text[i]=" << Descr->transform_text[h] << endl;
				cout << "transformation_coeffs_sz=" << transformation_coeffs_sz << endl;
				exit(1);
			}

			G->A_base->Group_element->make_element(Elt1, transformation_coeffs, verbose_level);

			if (Descr->transform_f_inv[h]) {
				G->A_base->Group_element->element_invert(Elt1, Elt2, 0 /*verbose_level*/);
			}
			else {
				G->A_base->Group_element->element_move(Elt1, Elt2, 0 /*verbose_level*/);
			}

			//A->element_transpose(Elt2, Elt3, 0 /*verbose_level*/);
#if 0
			G->A_base->element_invert(Elt2, Elt3, 0 /*verbose_level*/);

			if (f_v) {
				cout << "spread_create::init "
						"applying the transformation given by:" << endl;
				cout << "$$" << endl;
				G->A_base->print_quick(cout, Elt2);
				cout << endl;
				cout << "$$" << endl;
				cout << "spread_create::init "
						"The inverse is:" << endl;
				cout << "$$" << endl;
				G->A_base->print_quick(cout, Elt3);
				cout << endl;
				cout << "$$" << endl;
			}
#endif
			// apply the transformation:

			int i;

			for (i = 0; i < sz; i++) {
				image_set[i] = G_on_subspaces->A->Group_element->element_image_of(set[i], Elt2, verbose_level - 1);
			}
			Lint_vec_copy(image_set, set, sz);


			if (f_has_group) {
				groups::strong_generators *SG2;

				SG2 = NEW_OBJECT(groups::strong_generators);
				if (f_v) {
					cout << "spread_create::init "
							"before SG2->init_generators_for_the_conjugate_group_avGa" << endl;
				}
				SG2->init_generators_for_the_conjugate_group_avGa(Sg, Elt2, verbose_level);

				if (f_v) {
					cout << "spread_create::init "
							"after SG2->init_generators_for_the_conjugate_group_avGa" << endl;
				}

				FREE_OBJECT(Sg);
				Sg = SG2;
			}

		} // next h

		FREE_lint(image_set);
		FREE_int(Elt1);
		FREE_int(Elt2);
		//FREE_int(Elt3);
	}



	if (f_v) {
		cout << "spread_create::init set of size " << sz << endl;
		cout << "spread_create::init set = ";
		Lint_vec_print(cout, set, sz);
		cout << endl;
	}

	long int *Part = NULL;
	int s;

	Grass->make_partition(set, sz, Part, s, verbose_level - 1);

	if (f_v) {
		cout << "spread_create::init Partition:" << endl;
		Lint_matrix_print(Part, sz, s);
	}

	if (f_has_group) {
		cout << "spread_create::init the stabilizer is:" << endl;
		Sg->print_generators_tex(cout);
	}


	Andre = NEW_OBJECT(geometry::andre_construction);

	if (f_v) {
		cout << "spread_create::init before Andre->init" << endl;
	}
	Andre->init(F, k, set, verbose_level);
	if (f_v) {
		cout << "spread_create::init after Andre->init" << endl;
		cout << "spread_create::init before Andre->report" << endl;
		Andre->report(cout, verbose_level);
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

