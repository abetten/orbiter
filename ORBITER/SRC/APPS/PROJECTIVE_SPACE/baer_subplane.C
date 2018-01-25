// baer_subplane.C
// 
// Anton Betten
// 12/15/2010
//
// creates a Baer subplane in PG(2,q^2).
//
//
//

#include "orbiter.h"


// global data:

INT t0; // the system time when the program started


void Baer_subplane(INT q, INT verbose_level);

int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT i;
	INT q;
	
	t0 = os_ticks();

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}

	Baer_subplane(q, verbose_level);
	

	the_end(t0);
}


void Baer_subplane(INT q, INT verbose_level)
{
	//INT f_with_group = FALSE;
	//INT f_semilinear = TRUE;
	const BYTE *override_poly_Q = NULL;
	const BYTE *override_poly_q = NULL;
	//INT f_basis = TRUE;
	projective_space *P2;
	INT Q;
	INT *S;
	INT sz;
	finite_field *FQ;
	finite_field *Fq;
	INT *v;
	INT i, j, a, b, index, f_is_in_subfield;

	Q = q * q;
	P2 = new projective_space;

	FQ = new finite_field;
	Fq = new finite_field;

	FQ->init_override_polynomial(Q, override_poly_Q, 0);
	Fq->init_override_polynomial(q, override_poly_q, 0);

	P2->init(2, FQ, 
		//f_with_group, 
		//FALSE /* f_line_action */, 
		TRUE /* f_init_incidence_structure */, 
		//f_semilinear, f_basis, 
		verbose_level);


	index = (Q - 1) / (q - 1);
	cout << "index=" << index << endl;
	
	v = NEW_INT(3);	
	S = NEW_INT(P2->N_points);
	sz = 0;
	for (i = 0; i < P2->N_points; i++) {
		PG_element_unrank_modified(*FQ, v, 1, 3, i);
		for (j = 0; j < 3; j++) {
			a = v[j];
			b = FQ->log_alpha(a);
			f_is_in_subfield = FALSE;
			if (a == 0 || (b % index) == 0) {
				f_is_in_subfield = TRUE;
				}
			if (!f_is_in_subfield) {
				break;
				}
			}
		if (j == 3) {
			S[sz++] = i;
			}
		}
	cout << "the Baer subplane PG(2," << q << ") inside PG(2," << Q << ") has size " << sz << ":" << endl;
	for (i = 0; i < sz; i++) {
		cout << S[i] << " ";
		}
	cout << endl;



	BYTE fname[1000];
	sprintf(fname, "Baer_subplane_%ld_%ld.txt", q, Q);
	write_set_to_file(fname, S, sz, verbose_level);



	FREE_INT(v);
	FREE_INT(S);
	delete P2;
}



