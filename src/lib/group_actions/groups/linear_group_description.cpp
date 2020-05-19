// linear_group_description.cpp
//
// Anton Betten
// December 25, 2015

#include "foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace group_actions {


linear_group_description::linear_group_description()
{
	f_projective = FALSE;
	f_general = FALSE;
	f_affine = FALSE;
	f_GL_d_q_wr_Sym_n = FALSE;
	GL_wreath_Sym_d = 0;
	GL_wreath_Sym_n = 0;

	n = 0;
	input_q = 0;
	f_override_polynomial = FALSE;
	override_polynomial = NULL;
	F = NULL;
	f_semilinear = FALSE;
	f_special = FALSE;
	
	f_wedge_action = FALSE;
	f_PGL2OnConic = FALSE;
	f_monomial_group = FALSE;
	f_diagonal_group = FALSE;
	f_null_polarity_group = FALSE;
	f_symplectic_group = FALSE;
	f_singer_group = FALSE;
	f_singer_group_and_frobenius = FALSE;
	singer_power = 1;
	f_subfield_structure_action = FALSE;
	s = 1;
	f_subgroup_from_file = FALSE;
	f_borel_subgroup_upper = FALSE;
	f_borel_subgroup_lower = FALSE;
	f_identity_group = FALSE;
	subgroup_fname = NULL;
	subgroup_label = NULL;
	f_orthogonal_group = FALSE;
	orthogonal_group_epsilon = 0;

	f_on_k_subspaces = FALSE;
	on_k_subspaces_k = 0;

	f_on_tensors = FALSE;
	f_on_rank_one_tensors = FALSE;

	f_subgroup_by_generators = FALSE;
	subgroup_order_text = NULL;
	nb_subgroup_generators = 0;
	subgroup_generators_as_string = NULL;

	f_Janko1 = FALSE;
	//null();
}

linear_group_description::~linear_group_description()
{
	freeself();
}

void linear_group_description::null()
{
}

void linear_group_description::freeself()
{
	null();
}

void linear_group_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);

	int argc;
	char **argv;
	int i;

	if (f_v) {
		cout << "linear_group_description::read_arguments_from_string" << endl;
	}
	chop_string(str, argc, argv);

	if (f_vv) {
		cout << "argv:" << endl;
		for (i = 0; i < argc; i++) {
			cout << i << " : " << argv[i] << endl;
		}
	}


	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
	if (f_v) {
		cout << "linear_group_description::read_arguments_from_string "
				"done" << endl;
	}
}

int linear_group_description::read_arguments(
	int argc, const char **argv,
	int verbose_level)
{
	int i;

	cout << "linear_group_description::read_arguments" << endl;
	for (i = 0; i < argc; i++) {

#if 0
		if (argv[i][0] != '-') {
			continue;
			}
#endif

		// the general linear groups:
		// GL, GGL, SL, SSL
		if (strcmp(argv[i], "-GL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = FALSE;
			cout << "-GL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-GGL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = FALSE;
			cout << "-GGL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-SL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = TRUE;
			cout << "-SL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-SSL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = TRUE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = TRUE;
			cout << "-SSL " << n << " " << input_q << endl;
		}


		// the projective linear groups:
		// PGL, PGGL, PSL, PSSL
		else if (strcmp(argv[i], "-PGL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = FALSE;
			cout << "-PGL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-PGGL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = FALSE;
			cout << "-PGGL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-PSL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = FALSE;
			f_special = TRUE;
			cout << "-PSL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-PSSL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = TRUE;
			f_general = FALSE;
			f_affine = FALSE;
			f_semilinear = TRUE;
			f_special = TRUE;
			cout << "-PSSL " << n << " " << input_q << endl;
		}



		// the affine groups:
		// AGL, AGGL, ASL, ASSL
		else if (strcmp(argv[i], "-AGL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = FALSE;
			f_special = FALSE;
			cout << "-AGL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-AGGL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = TRUE;
			f_special = FALSE;
			cout << "-AGGL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-ASL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = FALSE;
			f_special = TRUE;
			cout << "-ASL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-ASSL") == 0) {
			n = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			f_projective = FALSE;
			f_general = FALSE;
			f_affine = TRUE;
			f_semilinear = TRUE;
			f_special = TRUE;
			cout << "-ASSL " << n << " " << input_q << endl;
		}
		else if (strcmp(argv[i], "-override_polynomial") == 0) {
			f_override_polynomial = TRUE;
			override_polynomial = argv[++i];
			cout << "-override_polynomial" << override_polynomial << endl;
		}
		else if (strcmp(argv[i], "-GL_d_q_wr_Sym_n") == 0) {
			f_GL_d_q_wr_Sym_n = TRUE;
			GL_wreath_Sym_d = atoi(argv[++i]);
			input_q = atoi(argv[++i]);
			GL_wreath_Sym_n = atoi(argv[++i]);
			cout << "-GL_d_q_wr_Sym_n " << GL_wreath_Sym_d << " " << input_q << " " << GL_wreath_Sym_n << endl;
		}



		else if (strcmp(argv[i], "-wedge") == 0) {
			f_wedge_action = TRUE;
			cout << "-wedge" << endl;
		}
		else if (strcmp(argv[i], "-PGL2OnConic") == 0) {
			f_PGL2OnConic = TRUE;
			cout << "-PGL2OnConic" << endl;
		}
		else if (strcmp(argv[i], "-monomial") == 0) {
			f_monomial_group = TRUE;
			cout << "-monomial " << endl;
		}
		else if (strcmp(argv[i], "-diagonal") == 0) {
			f_diagonal_group = TRUE;
			cout << "-diagonal " << endl;
		}
		else if (strcmp(argv[i], "-null_polarity_group") == 0) {
			f_null_polarity_group = TRUE;
			cout << "-null_polarity_group" << endl;
		}
		else if (strcmp(argv[i], "-symplectic_group") == 0) {
			f_symplectic_group = TRUE;
			cout << "-symplectic_group" << endl;
		}
		else if (strcmp(argv[i], "-singer") == 0) {
			f_singer_group = TRUE;
			singer_power = atoi(argv[++i]);
			cout << "-singer " << singer_power << endl;
		}
		else if (strcmp(argv[i], "-singer_and_frobenius") == 0) {
			f_singer_group_and_frobenius = TRUE;
			singer_power = atoi(argv[++i]);
			cout << "-f_singer_group_and_frobenius " << singer_power << endl;
		}
		else if (strcmp(argv[i], "-subfield_structure_action") == 0) {
			f_subfield_structure_action = TRUE;
			s = atoi(argv[++i]);
			cout << "-subfield_structure_action " << s << endl;
		}
		else if (strcmp(argv[i], "-subgroup_from_file") == 0) {
			f_subgroup_from_file = TRUE;
			subgroup_fname = argv[++i];
			subgroup_label = argv[++i];
			cout << "-subgroup_from_file " << subgroup_fname
					<< " " << subgroup_label << endl;
		}
		else if (strcmp(argv[i], "-borel_upper") == 0) {
			f_borel_subgroup_upper = TRUE;
			cout << "-borel_upper" << endl;
		}
		else if (strcmp(argv[i], "-borel_lower") == 0) {
			f_borel_subgroup_lower = TRUE;
			cout << "-borel_lower" << endl;
		}
		else if (strcmp(argv[i], "-identity_group") == 0) {
			f_identity_group = TRUE;
			cout << "-identity_group" << endl;
		}
		else if (strcmp(argv[i], "-on_k_subspaces") == 0) {
			f_on_k_subspaces = TRUE;
			on_k_subspaces_k = atoi(argv[++i]);
			cout << "-on_k_subspaces " << on_k_subspaces_k << endl;
		}
		else if (strcmp(argv[i], "-on_tensors") == 0) {
			f_on_tensors = TRUE;
			cout << "-on_tensors " << endl;
		}
		else if (strcmp(argv[i], "-on_rank_one_tensors") == 0) {
			f_on_rank_one_tensors = TRUE;
			cout << "-on_rank_one_tensors " << endl;
		}
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = atoi(argv[++i]);
			cout << "-orthogonal " << orthogonal_group_epsilon << endl;
		}
		else if (strcmp(argv[i], "-O") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = 0;
			cout << "-O" << endl;
		}
		else if (strcmp(argv[i], "-O+") == 0 ||
				strcmp(argv[i], "-Oplus") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = 1;
			cout << "-O+" << endl;
		}
		else if (strcmp(argv[i], "-O-") == 0 ||
				strcmp(argv[i], "-Ominus") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = -1;
			cout << "-O-" << endl;
		}
		else if (strcmp(argv[i], "-subgroup_by_generators") == 0) {
			f_subgroup_by_generators = TRUE;
			subgroup_label = argv[++i];
			subgroup_order_text = argv[++i];
			nb_subgroup_generators = atoi(argv[++i]);
			subgroup_generators_as_string = (const char **) NEW_pchar(nb_subgroup_generators);
			for (int h = 0; h < nb_subgroup_generators; h++) {
				subgroup_generators_as_string[h] = argv[++i];
			}
			cout << "-subgroup_by_generators " << subgroup_label
					<< " " << nb_subgroup_generators << endl;
			for (int h = 0; h < nb_subgroup_generators; h++) {
				cout << " " << subgroup_generators_as_string[h] << endl;
			}
			cout << endl;
		}
		else if (strcmp(argv[i], "-Janko1") == 0) {
			f_Janko1 = TRUE;
			cout << "-Janko1" << endl;
		}
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			break;
		}
		else {
			cout << "linear_group_description::read_arguments "
					"unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "linear_group_description::read_arguments done" << endl;
	return i + 1;
}

}}


