// linear_group_description.C
//
// Anton Betten
// December 25, 2015

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"



linear_group_description::linear_group_description()
{
	null();
}

linear_group_description::~linear_group_description()
{
	freeself();
}

void linear_group_description::null()
{
	f_projective = FALSE;
	f_general = FALSE;
	f_affine = FALSE;
	n = 0;
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
	singer_power = 1;
	f_subfield_structure_action = FALSE;
	f_subgroup_from_file = FALSE;
	f_borel_subgroup_upper = FALSE;
	f_borel_subgroup_lower = FALSE;
	f_identity_group = FALSE;
	f_orthogonal_group = FALSE;
	orthogonal_group_epsilon = 0;

	f_on_k_subspaces = FALSE;
	on_k_subspaces_k = 0;
}

void linear_group_description::freeself()
{
	null();
}

void linear_group_description::read_arguments_from_string(
		const char *str, int verbose_level)
{
	int f_vv = (verbose_level >= 2);
	int l, i, len;
	char *s;
	char *p_buf;
	char *buf;
	int argc;
	char **argv;

	l = strlen(str);
	s = NEW_char(l + 1);
	buf = NEW_char(l + 1);

	strcpy(s, str);
	p_buf = s;
	i = 0;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		//s_scan_token(&p_buf, str);
		//s_scan_token(&p_buf, str);
		/* r =*/ s_scan_token_arbitrary(&p_buf, buf);

		if (f_vv) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
			}
#if 0
		if (strcmp(str, ",") == 0) {
			continue;
			}
#endif
		i++;
		}
	argc = i;
	argv = NEW_pchar(argc);
	i = 0;
	p_buf = s;
	while (TRUE) {
		if (*p_buf == 0) {
			break;
			}
		//s_scan_token(&p_buf, str);
		//s_scan_token(&p_buf, str);
		/* r =*/ s_scan_token_arbitrary(&p_buf, buf);

		if (f_vv) {
			cout << "Token " << setw(6) << i << " is '"
					<< buf << "'" << endl;
			}
		len = strlen(buf);
		argv[i] = NEW_char(len + 1);
		strcpy(argv[i], buf);
#if 0
		if (strcmp(str, ",") == 0) {
			continue;
			}
#endif
		i++;
		}
	cout << "argv:" << endl;
	for (i = 0; i < argc; i++) {
		cout << i << " : " << argv[i] << endl;
	}

	read_arguments(
		argc, (const char **) argv,
		verbose_level);

	FREE_char(s);
	FREE_char(buf);
	for (i = 0; i < argc; i++) {
		FREE_char(argv[i]);
	}
	FREE_pchar(argv);
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
			cout << "-singer" << singer_power << endl;
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
		else if (strcmp(argv[i], "-borel_subgroup_upper") == 0) {
			f_borel_subgroup_upper = TRUE;
			cout << "-borel_subgroup_upper" << endl;
			}
		else if (strcmp(argv[i], "-borel_subgroup_lower") == 0) {
			f_borel_subgroup_lower = TRUE;
			cout << "-borel_subgroup_lower" << endl;
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
		else if (strcmp(argv[i], "-orthogonal") == 0) {
			f_orthogonal_group = TRUE;
			orthogonal_group_epsilon = atoi(argv[++i]);
			cout << "-orthogonal" << orthogonal_group_epsilon << endl;
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
		else if (strcmp(argv[i], "-end") == 0) {
			cout << "-end" << endl;
			return i;
			}
		else {
			cout << "linear_group_description::read_arguments unrecognized option " << argv[i] << endl;
		}
	} // next i
	cout << "linear_group_description::read_arguments done" << endl;
	return i;
}


