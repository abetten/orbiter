// group.C
//
// Anton Betten
// December 24, 2003

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

group::group()
{
	null();
}

group::~group()
{
	freeself();
}

void group::null()
{
	A = NULL;
	f_has_ascii_coding = FALSE;
	ascii_coding = NULL;

	f_has_strong_generators = FALSE;
	SG = NULL;
	tl = NULL;

	f_has_sims = FALSE;
	S = NULL;
}

void group::freeself()
{
	delete_ascii_coding();
	delete_strong_generators();
	delete_sims();
}

group::group(action *A)
{
	null();
	init(A);
};

group::group(action *A, const char *ascii_coding)
{
	null();
	init(A);
	init_ascii_coding(ascii_coding);
};

group::group(action *A, vector_ge &SG, int *tl)
{
	null();
	init(A);
	init_strong_generators(SG, tl);
};

void group::init(action *A)
{
	null();
	group::A = A;
}

void group::init_ascii_coding_to_sims(const char *ascii_coding)
{
	if (strlen(ascii_coding)) {
		init_ascii_coding(ascii_coding);
		
		decode_ascii(0);
		
		// now strong generators are available
		
		}
	else {
		//cout << "trivial group" << endl;
		init_strong_generators_empty_set();	
		}
	
	schreier_sims(0);
}

void group::init_ascii_coding(const char *ascii_coding)
{
	delete_ascii_coding();
	
	group::ascii_coding = NEW_char(strlen(ascii_coding) + 1);
	strcpy(group::ascii_coding, ascii_coding);
	f_has_ascii_coding = TRUE;
}

void group::delete_ascii_coding()
{
	if (f_has_ascii_coding) {
		FREE_char(ascii_coding);
		f_has_ascii_coding = FALSE;
		ascii_coding = NULL;
		}
}

void group::init_strong_generators_empty_set()
{
	int i;
	
	delete_strong_generators();
	
	group::SG = NEW_OBJECT(vector_ge);
	group::SG->init(A);
	group::SG->allocate(0);
	group::tl = NEW_int(A->base_len);
	for (i = 0; i < A->base_len; i++) {
		group::tl[i] = 1;
		}
	f_has_strong_generators = TRUE;
}

void group::init_strong_generators(vector_ge &SG, int *tl)
{
	int i;
	
	delete_strong_generators();
	
	group::SG = NEW_OBJECT(vector_ge);
	group::SG->init(A);
	group::SG->allocate(SG.len);
	for (i = 0; i < SG.len; i++) {
		group::SG->copy_in(i, SG.ith(i));
		}
	group::tl = NEW_int(A->base_len);
	for (i = 0; i < A->base_len; i++) {
		group::tl[i] = tl[i];
		}
	f_has_strong_generators = TRUE;
}

void group::init_strong_generators_by_hdl(int nb_gen,
		int *gen_hdl, int *tl, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;
	
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl" << endl;
		}
	if (f_v) {
		cout << "gen_hdl=";
		int_vec_print(cout, gen_hdl, nb_gen);
		cout << endl;
		cout << "tl=";
		int_vec_print(cout, tl, A->base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"before delete_strong_generators" << endl;
	}
	delete_strong_generators();
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"after delete_strong_generators" << endl;
	}
	
	SG = NEW_OBJECT(vector_ge);
	SG->init(A);
	SG->allocate(nb_gen);
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"before A->element_retrieve" << endl;
		}
	for (i = 0; i < nb_gen; i++) {
		A->element_retrieve(gen_hdl[i], SG->ith(i), verbose_level);
		}
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"after A->element_retrieve" << endl;
		}
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl "
				"before allocating tl of size " << A->base_len << endl;
		}
	group::tl = NEW_int(A->base_len);
	if (nb_gen) {
		for (i = 0; i < A->base_len; i++) {
			group::tl[i] = tl[i];
			}
		}
	else {
		for (i = 0; i < A->base_len; i++) {
			group::tl[i] = 1;
			}
		}
	f_has_strong_generators = TRUE;
	if (f_v) {
		cout << "group::init_strong_generators_by_hdl done" << endl;
		}
}

void group::delete_strong_generators()
{
	if (f_has_strong_generators) {
		FREE_OBJECT(SG);
		FREE_int(tl);
		SG = NULL;
		tl = NULL;
		f_has_strong_generators = FALSE;
		}
}

void group::delete_sims()
{
	if (f_has_sims) {
		if (S) {
			FREE_OBJECT(S);
			S = NULL;
		}
		f_has_sims = FALSE;
		}
}

void group::require_ascii_coding()
{
	if (!f_has_ascii_coding) {
		cout << "group::require_ascii_coding() "
				"!f_has_ascii_coding" << endl;
		exit(1);
		}
}

void group::require_strong_generators()
{
	if (!f_has_strong_generators) {
		cout << "group::require_strong_generators() "
			"!f_has_strong_generators" << endl;
		exit(1);
		}
}

void group::require_sims()
{
	if (!f_has_sims) {
		cout << "group::require_sims() !f_has_sims" << endl;
		exit(1);
		}
}

void group::group_order(longinteger_object &go)
{
	longinteger_domain D;
	
	if (f_has_sims) {
		D.multiply_up(go, S->orbit_len, A->base_len);
		}
	else if (f_has_strong_generators) {
		D.multiply_up(go, tl, A->base_len);
		}
	else {
		cout << "group::group_order() need sims or strong_generators" << endl;
		exit(1);
		}
}

void group::print_group_order(ostream &ost)
{
	longinteger_object go;
	group_order(go);
	ost << go;
}

void group::print_tl()
{
	int i;
	
	if (f_has_strong_generators) {
		for (i = 0; i < A->base_len; i++) 
			cout << tl[i] << " ";
		cout << endl;
		}
}

void group::code_ascii(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int sz, i, j;
	char *p;

	if (f_v) {
		cout << "group::code_ascii action " << A->label
				<< " base_len=" << A->base_len << endl;
		}
	require_strong_generators();
	sz = 2 * ((2 + A->base_len + A->base_len) * sizeof(int4)
			+ A->coded_elt_size_in_char * SG->len) + 1;
	ascii_coding = NEW_char(sz);
	p = ascii_coding;

	//cout << "group::code_ascii action A->base_len=" << A->base_len << endl;
	code_int4(p, (int4) A->base_len);
		// in GALOIS/util.C
	//cout << "group::code_ascii action SG->len=" << SG->len << endl;
	code_int4(p, (int4) SG->len);
	for (i = 0; i < A->base_len; i++) {
		code_int4(p, (int4) A->base[i]);
		}
	for (i = 0; i < A->base_len; i++) {
		code_int4(p, (int4) tl[i]);
		}
	for (i = 0; i < SG->len; i++) {
		A->element_pack(SG->ith(i), A->elt1, FALSE);
		for (j = 0; j < A->coded_elt_size_in_char; j++)
			code_uchar(p, A->elt1[j]);
		}
	*p++ = 0;
	if (p - ascii_coding != sz) {
		cout << "group::code_ascii(): p - ascii_coding != sz" << endl;
		exit(1);
		}
	f_has_ascii_coding = TRUE;
	if (f_v) {
		cout << "group::code_ascii() " << ascii_coding << endl;
		}
}

void group::decode_ascii(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, j;
	int len, nbsg;
	int *base1;
	char *p, *p0;
	int str_len;

	require_ascii_coding();
	//cout << "group::decode_ascii ascii_coding=" << ascii_coding << endl;
	p = ascii_coding;
	p0 = p;
	str_len = strlen(ascii_coding);
	len = decode_int4(p);
	nbsg = decode_int4(p);
	if (len != A->base_len) {
		cout << "group::decode_ascii() len != A->base_len" << endl;
		cout << "len=" << len << " (from file)" << endl;
		cout << "A->base_len=" << A->base_len << endl;
		cout << "action A is " << A->label << endl;
		exit(1);
		}
	delete_strong_generators();
	SG = NEW_OBJECT(vector_ge);
	SG->init(A);
	SG->allocate(nbsg);
	base1 = NEW_int(A->base_len);
	tl = NEW_int(A->base_len);
	for (i = 0; i < A->base_len; i++) {
		base1[i] = decode_int4(p);
		}
	for (i = 0; i < A->base_len; i++) {
		if (base1[i] != A->base[i]) {
			cout << "group::decode_ascii base mismatch" << endl;
			exit(1);
			}
		}
	for (i = 0; i < A->base_len; i++) {
		tl[i] = decode_int4(p);
		}
	for (i = 0; i < nbsg; i++) {
		for (j = 0; j < A->coded_elt_size_in_char; j++) {
			decode_uchar(p, A->elt1[j]);
			}
		A->element_unpack(A->elt1, SG->ith(i), FALSE);
		}
	FREE_int(base1);
	if (p - p0 != str_len) {
		cout << "group::decode_ascii p - p0 != str_len" << endl;
		cout << "p - p0 = " << p - p0 << endl;
		cout << "str_len = " << str_len << endl;
		exit(1);
		}
	f_has_strong_generators = TRUE;
	if (f_v) {
		if (SG->len < 10) {
			SG->print(cout);
			}	
		cout << "found a group with " << SG->len
				<< " strong generators" << endl;
		}
}

void group::schreier_sims(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	
	if (f_v) {
		cout << "group::schreier_sims" << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	require_strong_generators();
	if (f_v) {
		cout << "group::schreier_sims before delete_sims" << endl;
		}
	delete_sims();
	if (f_v) {
		cout << "group::schreier_sims after delete_sims" << endl;
		}
	S = NEW_OBJECT(sims);
	if (FALSE) {
		cout << "group::schreier_sims calling S->init(A)" << endl;
		}
	S->init(A);
	if (FALSE) {
		cout << "group::schreier_sims calling S->init_generators" << endl;
		}
	if (FALSE) {
		cout << "generators" << endl;
		SG->print(cout);
		}
	S->init_generators(*SG, verbose_level - 2);
	if (f_v) {
		cout << "group::schreier_sims after S->init_generators" << endl;
		cout << "tl: ";
		int_vec_print(cout, tl, A->base_len);
		cout << endl;
	}
	if (f_v) {
		cout << "group::schreier_sims before "
				"compute_base_orbits_known_length" << endl;
	}
	S->compute_base_orbits_known_length(tl, verbose_level - 2);
	if (f_v) {
		cout << "group::schreier_sims after "
				"compute_base_orbits_known_length" << endl;
	}
	
	if (f_v) {
		cout << "group::schreier_sims done. Found a group of order ";
		S->print_group_order(cout);
		cout << endl;
		}
	f_has_sims = TRUE;
}

void group::get_strong_generators(int verbose_level)
{
	require_sims();
	delete_strong_generators();
	SG = NEW_OBJECT(vector_ge);
	SG->init(A);
	tl = NEW_int(A->base_len);
	S->extract_strong_generators_in_order(*SG, tl, verbose_level - 1);
}

void group::point_stabilizer(group &stab, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	require_sims();

	vector_ge stab_gens;
	int *tl;
	
	if (f_v) {
		cout << "group::point_stabilizer() "
				"computing stabilizer of point " << pt << endl;
		}
	
	
	tl = NEW_int(A->base_len);
	S->point_stabilizer(stab_gens, tl, pt, verbose_level - 1);
	
#if 0
	if (f_v) {
		if (f_vv) {
			stab_gens.print(cout);
			}
		cout << stab_gens.len << " strong generators computed" << endl;
		}
#endif
	stab.freeself();
	stab.init(A);
	stab.init_strong_generators(stab_gens, tl);
	FREE_int(tl);
	if (f_v) {
		cout << "stabilizer of point " << pt << " has order ";
		stab.print_group_order(cout);
		cout << " ";
		int_vec_print(cout, stab.tl, A->base_len);
		cout << " with " << stab_gens.len << " strong generators" << endl;
		if (f_vv) {
			stab_gens.print(cout);
			}
		}
}

void group::point_stabilizer_with_action(action *A2,
		group &stab, int pt, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	require_sims();

	vector_ge stab_gens;
	int *tl;
	
	if (f_v) {
		cout << "group::point_stabilizer_with_action() ";
		cout << "computing stabilizer of point " << pt 
			<< " in action " << A2->label 
			<< " internal action is " << stab.A->label << endl;
		cout << "verbose_level = " << verbose_level << endl;
		}
	
	
	tl = NEW_int(A->base_len);
	if (f_v) {
		cout << "group::point_stabilizer_with_action() "
				"calling S->point_stabilizer_with_action" << endl;
		}
	S->point_stabilizer_with_action(A2, stab_gens,
			tl, pt, verbose_level - 1);
	if (f_v) {
		cout << "group::point_stabilizer_with_action() "
				"after S->point_stabilizer_with_action" << endl;
		}
	
#if 0
	if (f_v) {
		if (f_vv) {
			stab_gens.print(cout);
			}
		cout << stab_gens.len << " strong generators computed" << endl;
		}
#endif
	stab.freeself();
	stab.init(A);
	stab.init_strong_generators(stab_gens, tl);
	FREE_int(tl);
	if (f_v) {
		cout << "stabilizer of point " << pt << " has order ";
		stab.print_group_order(cout);
		cout << " ";
		int_vec_print(cout, stab.tl, A->base_len);
		cout << " with " << stab_gens.len << " strong generators" << endl;
		if (f_vv) {
			stab_gens.print(cout);
			}
		}
}

void group::induced_action(action &induced_action,
		group &H, group &K, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	
	if (f_v) {
		cout << "group::induced_action" << endl;
		}
	{
	vector_ge H_SG, K_SG;
	int *H_tl, *K_tl;
	int n = 0;
	{
	sims HH, KK;
	longinteger_object go, H_order, K_order, HK_order, quo, rem;
	int drop_out_level, image;
	longinteger_domain D;
	
	require_sims();
	
	group_order(go);
	
	HH.init(&induced_action);
	HH.init_trivial_group(verbose_level - 1);
	HH.group_order(H_order);
	
	KK.init(A);
	KK.init_trivial_group(verbose_level - 1);
	KK.group_order(K_order);
	
	D.mult(H_order, K_order, HK_order);
	if (f_v) {
		cout << "step " << n << " H_order " << H_order
			<< " K_order = " << K_order
			<< " HK_order " << HK_order << " of " << go << endl;
		}
	
	while (D.compare_unsigned(HK_order, go) != 0) {
		
		if (f_v) {
			cout << "step " << n << ":" << endl;
			}
		S->random_element(A->Elt1, verbose_level - 1);
		if (f_v) {
			cout << "random group element:" << endl;
			A->element_print(A->Elt1, cout);
			}
		
		if (HH.strip(A->Elt1, A->Elt2 /* residue */,
				drop_out_level, image, verbose_level - 1)) {
			if (f_vv) {
				cout << "element strips through H" << endl;
				}
			if (KK.strip(A->Elt2, A->Elt3 /* residue */,
					drop_out_level, image, verbose_level - 1)) {
				if (f_vv) {
					cout << "element strips through K" << endl;
					}
				}
			else {
				KK.add_generator_at_level(A->Elt3,
						drop_out_level, verbose_level - 1);
				}
			}
		else {
			HH.add_generator_at_level(A->Elt2,
					drop_out_level, verbose_level - 1);
			}

		HH.group_order(H_order);
		KK.group_order(K_order);
		D.mult(H_order, K_order, HK_order);

		if (f_v) {
			cout << "step " << n << " H_order " << H_order
				<< " K_order = " << K_order
				<< " HK_order " << HK_order << " of " << go << endl;
			D.integral_division(go, HK_order, quo, rem, 0);
			cout << "remaining factor: " << quo
				<< " remainder " << rem << endl;
			}
		n++;
		}

#if 0
	if (f_v) {
		cout << "group::induced_action() "
				"finished after " << n << " steps" << endl;
		cout << "H_order " << H_order << " K_order = " << K_order << endl;
		cout << "# generators for H = " << HH.gens.len
				<< ", # generators for K = " << KK.gens.len << endl;
		cout << "H:" << endl;
		HH.print(f_vv);
		cout << "K:" << endl;
		KK.print(f_vv);
		}
#endif

	H_tl = NEW_int(induced_action.base_len);
	K_tl = NEW_int(A->base_len);
	
	HH.extract_strong_generators_in_order(H_SG, H_tl, verbose_level - 2);
	KK.extract_strong_generators_in_order(K_SG, K_tl, verbose_level - 2);
	

	//cout << "group::induced_action() deleting HH,KK" << endl;
	}
	
	H.init(&induced_action);
	K.init(A);
	H.init_strong_generators(H_SG, H_tl);
	K.init_strong_generators(K_SG, K_tl);
	if (f_v) {
		cout << "group::induced_action() finished after "
				<< n << " iterations" << endl;
		cout << "order of the induced group  = ";
		H.print_group_order(cout);
		cout << endl;
		cout << "order of the kernel = ";
		K.print_group_order(cout);
		cout << endl;
		}
#if 0
	if (f_vv) {
		cout << "induced group:" << endl;
		HH.print(FALSE);
		cout << endl;
		cout << "kernel:" << endl;
		KK.print(FALSE);
		cout << endl;
		cout << H.SG->len << " strong generators for induced group:" << endl;
		H.SG->print(cout);
		cout << endl;
		cout << K.SG->len << " strong generators for kernel:" << endl;
		K.SG->print(cout);
		cout << endl;
		}
#endif
	FREE_int(H_tl);
	FREE_int(K_tl);
	//cout << "group::induced_action() deleting SG" << endl;
	}
	if (f_v) {
		cout << "group::induced_action() finished" << endl;
		}	
}

void group::extension(group &N, group &H, int verbose_level)
	// N needs to have strong generators, 
	// H needs to have sims
	// N and H may have different actions, 
	// the action of N is taken for the extension.
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	action *A = N.A;
	sims G;
	longinteger_object go_N, go_H, go_G, cur_go, quo, rem;
	longinteger_domain D;
	int n = 0, drop_out_level, image;
	int *p_gen;
	
	if (f_v) {
		cout << "group::extension" << endl;
		}
	N.require_strong_generators();
	N.group_order(go_N);
	H.group_order(go_H);
	D.mult(go_N, go_H, go_G);
	
	if (f_v) {
		cout << "group::extension() |N| = " << go_N << " |H| = " 
			<< go_H << " |G| = |N|*|H| = " << go_G << endl;
		}
	H.require_sims();
	
	init(N.A);
	G.init(N.A);
	G.init_generators(*N.SG, f_v);
	G.compute_base_orbits(verbose_level - 1);
	G.group_order(cur_go);

	while (D.compare_unsigned(go_G, cur_go) != 0) {
		
		if (f_v) {
			cout << "step " << n << ":" << endl;
			}
		if (n % 2 || G.nb_gen[0] == 0) {
			H.S->random_element(A->Elt1, verbose_level - 1);
			p_gen = A->Elt1;
			if (f_v) {
				cout << "random group element:" << endl;
				A->element_print(p_gen, cout);
				}
			}
		else {
			G.random_schreier_generator(verbose_level - 1);
			p_gen = G.schreier_gen;
			if (f_v) {
				cout << "random schreier generator:" << endl;
				A->element_print(p_gen, cout);
				}
			}
		
		
		if (G.strip(p_gen, A->Elt2 /* residue */,
				drop_out_level, image, verbose_level - 1)) {
			if (f_vv) {
				cout << "element strips through" << endl;
				}
			}
		else {
			G.add_generator_at_level(A->Elt2,
					drop_out_level, verbose_level - 1);
			}

		G.group_order(cur_go);

		if (f_v) {
			cout << "step " << n 
				<< " cur_go " << cur_go << " of " << go_G << endl;
			D.integral_division(go_G, cur_go, quo, rem, 0);
			cout << "remaining factor: " << quo
					<< " remainder " << rem << endl;
			}
		n++;
		}
	
	vector_ge SG;
	int *tl;
	
	tl = NEW_int(A->base_len);
	
	G.extract_strong_generators_in_order(SG, tl, verbose_level - 2);
	
	init(A);
	init_strong_generators(SG, tl);
	
	if (f_v) {
		cout << "group::extension() finished after "
				<< n << " iterations" << endl;
		cout << "order of the extension = ";
		print_group_order(cout);
		cout << endl;
		}
	FREE_int(tl);
}

void group::print_strong_generators(ostream &ost,
		int f_print_as_permutation)
{
	int i, l;
	
	if (!f_has_strong_generators) {
		cout << "group::print_strong_generators "
				"no strong generators" << endl;
		exit(1);
		}
	ost << "group::print_strong_generators a group with tl=";
	int_vec_print(ost, tl, A->base_len);
	l = SG->len;
	ost << " and with " << l << " strong generators" << endl;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << ":" << endl;
		A->element_print_quick(SG->ith(i), ost);
		ost << endl;
		if (f_print_as_permutation) {
			A->element_print_as_permutation(SG->ith(i), ost);
			ost << endl;
			}
		}
}

void group::print_strong_generators_with_different_action(
		ostream &ost, action *A2)
{
	print_strong_generators_with_different_action_verbose(
			ost, A2, 0);
}

void group::print_strong_generators_with_different_action_verbose(
		ostream &ost, action *A2, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int i, l;
	int *Elt;
	
	if (f_v) {
		cout << "group::print_strong_generators_with_different_"
				"action_verbose" << endl;
		}
	if (!f_has_strong_generators) {
		cout << "group::print_strong_generators_with_different_"
				"action no strong generators" << endl;
		exit(1);
		}
	ost << "group::print_strong_generators_with_different_"
			"action_verbose a group with tl=";
	int_vec_print(ost, tl, A->base_len);
	l = SG->len;
	ost << " and with " << l << " strong generators" << endl;
	for (i = 0; i < l; i++) {
		ost << "generator " << i << ":" << endl;
		A->element_print_quick(SG->ith(i), ost);
		ost << endl;
		Elt = SG->ith(i);
		if (f_vv) {
			if (f_v) {
				cout << "group::print_strong_generators_with_"
						"different_action_verbose computing images "
						"individually" << endl;
				}
			int j; //, k;
			for (j = 0; j < A2->degree; j++) {
				//cout << "group::print_strong_generators_with_"
				//"different_action_verbose  computing image of "
				// << j << endl;
				/*k =*/ A2->element_image_of(j, Elt,
						0 /*verbose_level - 2*/);
				//cout << "group::print_strong_generators_with_"
				//"different_action_verbose  image of "
				// << j << " is " << k << endl;
				}
			}
		ost << "as permutation in action " << A2->label
				<< " of degree " << A2->degree << ":" << endl;
		A2->element_print_as_permutation_verbose(Elt,
				ost, 0/*verbose_level - 2*/);
		ost << endl;
		}
}


