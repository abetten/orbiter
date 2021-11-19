/*
 * geo_isot.cpp
 *
 *  Created on: Aug 14, 2021
 *      Author: betten
 */

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace foundations {

#undef DEBUG_HBAR

#undef DEBUG_ISO

#define MAX_ISO 250000


#define MAX_SET_SIZE MAX_VB

typedef struct ordered_set ORDERED_SET;
typedef struct iso2 ISO2;



struct ordered_set {
	int a[MAX_SET_SIZE];
	int size;
};

struct iso2 {
	iso_info *info;
	int Adim_n;
	int Bdim_n;
	int *AtheX;
	int *BtheX;
	int *A_R;
	int *B_R;
	int f_A_R_allocated;
	int f_B_R_allocated;
	int v_t;
	int b_t;
	int tdo_m_t;
	int tdo_n_t;
	int *tdo_V_t;
	int *tdo_B_t;
	ORDERED_SET *E; /* MAX_VB */
	
	int f_use_ddp;
	int f_use_ddb;

	short *Addp; /* (v \atop 2) entries */
	short *Bddp;
	short *Addb; /* (b \atop 2) entries */
	short *Bddb;

	/* for the partitioning of the rows: */
	int hbar[MAX_VB];
	/* column that is responsible for 
	 * a hbar before this row. 
	 * if equal to b, 
	 * then no hbar is present. 
	 * initially hbar[i] = -1 
	 * for all i in i_hbar[]. */
	int hlen[MAX_VB];
	int grid_entry[MAX_VB];
	/* the number of hbars 
	 * that lie over this row;
	 * or: number into the 
	 * hbar fields for this row. 
	 * this will be used as 
	 * an index into type[][]. */
	int G_max;
	/* number of grid_entries. */
	cperm Ap;
		/* row permutation for A; 
		 * degree iso->v_t */
	cperm Apv; /* Ap^-1 */
	cperm Bp;
		/* row permutation for B; 
		 * degree iso->v_t */
	cperm Bpv; /* Bp^-1 */
		/* given AtheX and BtheX, 
		 * applying Ap /Bp to the rows, 
		 * q to the columns, You get the 
		 * matrix C of the isomorphism. 
		 * q is inside ISO_GRID and depends on 
		 * the actual level k. */
	
	/* for the partitioning of the columns: */
	/* type_len = G_max
	 *   + 1 if derivative (Ad, Bd) is present
	 *   + k if (Add, Bdd) is present:
	 *         	(k is the actual level)
	 *         	for all the columns as yet in C. */
	iso_grid *G1; /* for A */
	iso_grid *G2; /* for B */

	/* ISO_GEO_DATA A, B;
	int hlen1[MAX_VB]; */
};

static void iso2_nil(ISO2 *iso);
static void iso2_init(ISO2 *iso, iso_info *info);
static void iso2_ext(ISO2 *iso);
static int iso2_do_level(ISO2 *iso, int i);
/* Rueckgabe FALSE, 
 * falls wegen f_break_after_first
 * abgebrochen wurde. */
static void iso2_B_add_point(
	ISO2 *iso, int k, int b0);
static void iso2_A_add_point(ISO2 *iso, int k);
static void iso2_A_del_point(ISO2 *iso, int k);
static void check_hbar(ISO2 *iso, int k);
static void iso2_check_iso(
	ISO2 *iso, int k, ORDERED_SET *E);
static void iso2_print_Ei(ISO2 *iso, int i);
static void iso2_calc_grid(
	ISO2 *iso, iso_grid *G, int k,
	int *theX, int dim_n, int *d,
	short *dd, cperm *p);
/* bislang: max_r konstant, statt r Array
 * type_len nur ohne 
 * zweite Ableitungen (d und dd). 
 * iso->grid_entry[] 
 * muss bereits berechnet sein. */
static void iso2_radix_sort(
	ISO2 *iso, iso_grid *G, int radix,
	int first, int last);
static int iso2_insert_idx(
	iso_grid *G, int first,
	int len, int radix,
	int search_this, int *idx);
static void print_hbar(ISO2 *iso);


/*
 *
 */

void iso_info::init_A_int(int *theA, int f_full)
{
	iso_info::AtheX = theA;
	iso_info::Af_full = f_full;
}

void iso_info::init_B_int(int *theB, int f_full)
{
	iso_info::BtheX = theB;
	iso_info::Bf_full = f_full;
}

void iso_info::init_ddp(int f_ddp, short *Addp, short *Bddp)
{
	f_use_ddp = FALSE;
	if (f_ddp) {
		f_use_ddp = TRUE;
		iso_info::Addp = Addp;
		iso_info::Bddp = Bddp;
		}
}

void iso_info::init_ddb(int f_ddb, short *Addb, short *Bddb)
{
	f_use_ddb = FALSE;
	if (f_ddb) {
		f_use_ddb = TRUE;
		iso_info::Addb = Addb;
		iso_info::Bddb = Bddb;
		}
}

void iso_info::init_tdo(tdo_scheme *tdos)
{
	int i, j, len;

	len = tdos->nb_rows();
	tdo_m = len;
	for (i = 0; i < len; i++)
		tdo_V[i] = tdos->Vi(i);

	len = tdos->nb_cols();
	tdo_n = len;
	for (j = 0; j < len; j++)
		tdo_B[j] = tdos->Bj(j);
}

void iso_info::init_tdo_V_B(int V, int B, int *Vi, int *Bj)
{
	int i, j;

	tdo_m = V;
	for (i = 0; i < V; i++) {
		tdo_V[i] = Vi[i];
	}
	tdo_n = B;
	for (j = 0; j < B; j++) {
		tdo_B[j] = Bj[j];
	}
}

void iso_info::iso_test(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "iso_info::iso_test" << endl;
	}

	ISO2 *iso = NULL;

	iso = new iso2;

	iso2_nil(iso);


	iso2_init(iso, this);

	if (f_v) {
		cout << "iso_info::iso_test before iso2_do_level" << endl;
	}
	iso2_do_level(iso, 0);
	if (f_v) {
		cout << "iso_info::iso_test after iso2_do_level" << endl;
	}

	iso2_ext(iso);

	delete iso;

	if (f_v) {
		cout << "iso_info::iso_test done" << endl;
	}
}

void init_ISO2()
{
	cout << "sizeof(iso_info) = " << sizeof(iso_info) << endl;
	cout << "sizeof(ISO2) = " << sizeof(ISO2) << endl;
	cout << "sizeof(iso_grid) = " << sizeof(iso_grid) << endl;
}

static void iso2_nil(ISO2 *iso)
{
	iso->info = NULL;
	iso->AtheX = NULL;
	iso->BtheX = NULL;
	iso->A_R = NULL;
	iso->B_R = NULL;
	iso->f_A_R_allocated = FALSE;
	iso->f_B_R_allocated = FALSE;
	iso->E = NULL;
	iso->G1 = NULL;
	iso->G2 = NULL;

	/* igd_nil(&iso->A);
	igd_nil(&iso->B); */
}

static void iso2_init(ISO2 *iso, iso_info *info)
{
	int i, j, k, I, first;
	int last_p1, len, size, dim_n;
	int *theX1, *R1, dim_n1;
	
	iso->info = info;

	size = info->v * info->max_r * sizeof(int);
	iso->AtheX = new int [info->v * info->max_r];
	iso->BtheX = new int [info->v * info->max_r];

	if (info->Af_full)
		dim_n = MAX_R;
	else
		dim_n = info->max_r;
	for (i = 0; i < info->v; i++) {
		for (j = 0; j < info->max_r; j++) {
			k = info->AtheX[i * dim_n + j];
			iso->AtheX[i * info->max_r + j] = k;
		}
	}
	iso->Adim_n = info->max_r;
	if (info->f_transpose_it) {
		inc_transpose(
			info->R, iso->AtheX, 
			FALSE /* f_full */ , 
			info->max_r, info->v, info->b, 
			&theX1, &dim_n1, &R1);
		delete [] iso->AtheX;
		iso->AtheX = theX1;
		iso->Adim_n = dim_n1;
		iso->A_R = R1;
		iso->v_t = info->b;
		iso->b_t = info->v;
#ifdef DEBUG_ISO
		printf("info->b = %d "
			"info->v = %d\n",
			info->b, info->v);
			fflush(stdout);
		inc_theX_print_int_char(iso->A_R,
			iso->AtheX, 
			TRUE /* f_int */,
			FALSE /* f_full */, 
			iso->Adim_n, iso->v_t);
		fflush(stdout);
#endif
		iso->f_A_R_allocated = TRUE;
		iso->tdo_m_t = info->tdo_n;
		iso->tdo_n_t = info->tdo_m;
		iso->tdo_V_t = info->tdo_B;
		iso->tdo_B_t = info->tdo_V;
		iso->f_use_ddp = info->f_use_ddb;
		iso->Addp = info->Addb;
		iso->f_use_ddb = info->f_use_ddp;
		iso->Addb = info->Addp;
	}
	else {
		iso->A_R = info->R;
		iso->f_A_R_allocated = FALSE;
		iso->v_t = info->v;
		iso->b_t = info->b;
		iso->tdo_m_t = info->tdo_m;
		iso->tdo_n_t = info->tdo_n;
		iso->tdo_V_t = info->tdo_V;
		iso->tdo_B_t = info->tdo_B;
		iso->f_use_ddp = info->f_use_ddp;
		iso->Addp = info->Addp;
		iso->f_use_ddb = info->f_use_ddb;
		iso->Addb = info->Addb;
	}

	if (info->Bf_full) {
		dim_n = MAX_R;
	}
	else {
		dim_n = info->max_r;
	}
	for (i = 0; i < info->v; i++) {
		for (j = 0; j < info->max_r; j++) {
			k = info->BtheX[i * dim_n + j];
			iso->BtheX[i * info->max_r + j] = k;
		}
	}
	iso->Bdim_n = info->max_r;
	if (info->f_transpose_it) {
		inc_transpose(
			info->R, iso->BtheX, 
			FALSE /* f_full */ , 
			info->max_r, info->v, info->b, 
			&theX1, &dim_n1, &R1);
		delete [] iso->BtheX;
		iso->BtheX = theX1;
		iso->Bdim_n = dim_n1;
		iso->B_R = R1;
		iso->f_B_R_allocated = TRUE;
		iso->Bddp = info->Bddb;
		iso->Bddb = info->Bddp;
		}
	else {
		iso->B_R = info->R;
		iso->f_B_R_allocated = FALSE;
		iso->Bddp = info->Bddp;
		iso->Bddb = info->Bddb;
		}
	if (info->f_transpose_it) {
		for (i = 0; i < iso->v_t; i++) {
			if (iso->A_R[i] != iso->B_R[i]) {
				cout << "iso2_int iso->A_R[i] != iso->B_R[i]" << endl;
				exit(1);
				}
			}
		}
	iso->Ap.init_and_identity(iso->v_t);
	iso->Apv.init_and_identity(iso->v_t);
	iso->Bp.init_and_identity(iso->v_t);
	iso->Bpv.init_and_identity(iso->v_t);
	
	iso->E = new ORDERED_SET [iso->b_t];

	iso->G1 = new iso_grid;
	iso->G2 = new iso_grid;

	iso->G1->q.init_and_identity(iso->b_t);
	iso->G1->qv.init_and_identity(iso->b_t);
	iso->G2->q.init_and_identity(iso->b_t);
	iso->G2->qv.init_and_identity(iso->b_t);
	
#if 0
	/* ein einziger hbar Bereich: */
	for (i = 0; i < info->v; i++) {
		iso->hbar[i] = info->v;
		iso->hlen[i] = 0;
		iso->grid_entry[i] = 0;
		}
	iso->hbar[0] = -1;
	iso->hlen[0] = info->v;
	iso->G_max = 1;
#endif
	/* init grid: */

	iso->G_max = 0;
	for (i = 0; i < iso->v_t; i++) {
		iso->hbar[i] = iso->b_t;
		iso->hlen[i] = 0;
		}
	iso->G_max = iso->tdo_m_t;
	first = 0;
	for (I = 0; I < iso->G_max; I++) {
		len = iso->tdo_V_t[I];
		iso->hbar[first] = -1;
		iso->hlen[first] = len;
		last_p1 = first + len;
		for (i = first; i < last_p1; i++) {
			iso->grid_entry[i] = I;
			}
		first = last_p1;
		}
}

static void iso2_ext(ISO2 *iso)
{
	if (iso->f_A_R_allocated) {
		delete [] iso->A_R;
		iso->A_R = NULL;
		iso->f_A_R_allocated = FALSE;
		}
	if (iso->f_B_R_allocated) {
		delete [] iso->B_R;
		iso->B_R = NULL;
		iso->f_B_R_allocated = FALSE;
		}
	if (iso->AtheX) {
		delete [] iso->AtheX;
		iso->AtheX = NULL;
		}
	if (iso->BtheX) {
		delete [] iso->BtheX;
		iso->BtheX = NULL;
		}
	if (iso->E) {
		delete [] iso->E;
		iso->E = NULL;
		}
	if (iso->G1) {
		delete iso->G1;
		iso->G1 = NULL;
		}
	if (iso->G2) {
		delete iso->G2;
		iso->G2 = NULL;
		}
}



static int iso2_do_level(ISO2 *iso, int i)
/* Rueckgabe FALSE, 
 * falls wegen f_break_after_first
 * abgebrochen wurde. */
{
	int j;
	char s[256];
	
	if (iso->info->f_very_verbose) {
		printf("*** ISO2: level %d ***\n", i);
	}

	if (i == iso->b_t - 1) {
		iso->info->nb_isomorphisms++;
		if (iso->info->nb_isomorphisms > MAX_ISO) {
			return FALSE;
		}
		if (iso->info->f_verbose) {
			printf("*** found isomorphism nr.%d !\n",
				iso->info->nb_isomorphisms);
			s[0] = 0;
			{
				cperm c;
				
				c.init_and_identity(iso->Bp.l);
				iso->Bp.mult(&iso->Apv, &c);
				c.print();

				//printf("%s\n", s);
			}
		}
		
		/* if (iso->info->f_verbose) {
			print_iso(A, isoA);
			print_iso(B, isoB);
			} */

		if (iso->info->f_break_after_fst) {
			return(FALSE);
				/* terminate program */
		}
		else {
			return(TRUE);
				/* proceed further */
		}
	}
		
	iso2_calc_grid(iso, iso->G1, i, iso->AtheX, 
		iso->Adim_n, iso->info->Ad, 
		iso->Addb, &iso->Ap);
	
	if (iso->info->f_very_verbose) {
		printf("A:\n");
		print_theX_pq(iso->AtheX, iso->Adim_n, 
			iso->v_t, iso->b_t, iso->A_R, 
			&iso->Apv, &iso->G1->qv);
		iso->G1->print();
	}

	iso2_calc_grid(iso, iso->G2, i, 
		iso->BtheX, 
		iso->Bdim_n, iso->info->Bd, 
		iso->Bddb, &iso->Bp);
	
	if (iso->info->f_very_verbose) {
		printf("B:\n");
		print_theX_pq(iso->BtheX, iso->Bdim_n, 
			iso->v_t, iso->b_t, iso->B_R, 
			&iso->Bpv, &iso->G2->qv);
		iso->G2->print();
	}
	
	iso2_check_iso(iso, i, &iso->E[i]);
	
	if (iso->info->f_very_verbose) {
		iso2_print_Ei(iso, i);
	}
	
	if (iso->E[i].size == 0L) {
		return TRUE;
	}
	
	iso2_A_add_point(iso, i);
	
	if (iso->info->f_very_verbose) {
		print_hbar(iso);
	}

	while (iso->E[i].size > 0L) {
	
		iso2_B_add_point(iso, i, iso->E[i].a[0]);
		
		if (!iso2_do_level(iso, i + 1L)) {
			return(FALSE);
		}
		
		/* delete first of E[i]: */
		for (j = 1; j < iso->E[i].size; j++) {
			iso->E[i].a[j - 1] = 
			iso->E[i].a[j];
		}
		iso->E[i].size--;
		if (iso->info->f_very_verbose) {
			iso2_print_Ei(iso, i);
		}
		
	}

	iso2_A_del_point(iso, i);
	
	if (iso->info->f_very_verbose) {
		cout << "after A_del_points" << endl;
		print_hbar(iso);
	}

	return(TRUE);
}

static void iso2_B_add_point(
	ISO2 *iso, int k, int b0)
{
	int i, first, len, k0;
	int b, last_len, j, l, j1, o;
	
	b = iso->G2->q.data[b0];
	if (b != k) {
		iso->G2->q.mult_apply_tau_r(k, b);
		iso->G2->qv.mult_apply_tau_l(k, b);
	}
	k0 = iso->G2->qv.data[k];
	if (k0 != b0) {
		cout << "iso2_B_add_point k0 != b0" << endl;
		exit(1);
	}
	first = 0; /* Zeile */
	i = 0; /* horizontal grid_entry */
	last_len = -1;
	while (TRUE) {
		/* we must have a hbar at first: */
		if (iso->hbar[first] > k) {
			cout << "iso2_B_add_point iso->hbar[first] > k" << endl;
			exit(1);
		}
		len = iso->hlen[first];
		if (iso->hbar[first] == k) {
			if (last_len == -1) {
				cout << "iso2_B_add_point last_len == -1" << endl;
				exit(1);
			}
			l = 0;
			for (j = -last_len; j < len; j++) {
				j1 = iso->Bpv.data[first + j];
				for (o = 0; o < iso->B_R[j1]; o++) {
					/* before: max_r */
					if (iso->BtheX[(j1 * iso->Bdim_n + o)] == k0) {
						break;
					}
				}
				if (o < iso->B_R[j1]) {
					/* this means: (j1, k0) is in BtheX. 
					 * swap row first - last_len + l 
					 * with first + j. */
					if (- last_len + l != j) {
						iso->Bp.mult_apply_tau_r(first - last_len + l, first + j);
						iso->Bpv.mult_apply_tau_l(first - last_len + l, first + j);
					}
					l++;
					if (l > last_len) {
						cout << "iso2_B_add_point l > last_len" << endl;
						exit(1);
					}
				}
				else {
					/* (j1, k0) not in BtheX. */
				}
			} /* next j */
		} /* if (iso->hbar[first] == k) */
		i++;
		first += len;
		last_len = len;
		if (i >= iso->G_max) {
			break;
		}
	}
	if (first != iso->v_t) {
		cout << "iso2_B_add_point first != iso->v_t" << endl;
		exit(1);
	}
}

static void iso2_A_add_point(
	ISO2 *iso, int k)
{
	int i, first, len, k1;
	int k_type_idx, new_len;
	int ge, j, l, j1, o;
	
	check_hbar(iso, k);

	k1 = iso->G1->q.data[k];
	k_type_idx = iso->G1->type_idx[k1];
	if (k != k1) {
		iso->G1->q.mult_apply_tau_r(k, k1);
		iso->G1->qv.mult_apply_tau_l(k, k1);
	}
	first = 0; /* Zeile */
	i = 0; /* h grid_entry */
	ge = 0; /* new h grid_entry */
	while (TRUE) {
		/* we must have a hbar at first: */
		if (iso->hbar[first] >= k) {
			cout << "iso2_A_add_point iso->hbar[first] >= k" << endl;
			exit(1);
		}
		len = iso->hlen[first];
		new_len = iso->G1->type[k_type_idx][i];
		/* printf("i = %d ge = %d len = %d new_len = %d\n", i, ge, len, new_len); */
		if (new_len && new_len < len) {
			/* add a new hbar: */
			iso->hbar[first + new_len] = k;
			iso->hlen[first] = new_len;
			iso->hlen[first + new_len] = len - new_len;
			for (j = 0; j < new_len; j++) {
				iso->grid_entry[first + j] = ge;
			}
			ge++;
			for (j = new_len; j < len; j++) {
				iso->grid_entry[first + j] = ge;
			}
			
			/* resort the rows: */
			l = 0;
				/* the number of rows 
				 * with the k-th block;
				 * 0 <= l <= new_len. */
			for (j = 0; j < len; j++) {
				j1 = iso->Apv.data[first + j];
				for (o = 0; o < iso->A_R[j1]; o++) {
					/* before: max_r */
					if (iso->AtheX[(j1 * iso->Adim_n + o)] == k) {
						break;
					}
				}
				if (o < iso->A_R[j1]) {
					/* this means: (j1, k) is in AtheX. 
					 * swap row first + l 
					 * with first + j. */
					if (j > l) {
						iso->Ap.mult_apply_tau_r(first + l, first + j);
						iso->Apv.mult_apply_tau_l(first + l, first + j);
					}
					l++;
					if (l > new_len) {
						cout << "iso2_A_add_point l > new_len" << endl;
						exit(1);
					}
				}
				else {
					/* (j1, k) not in AtheX. */
				}
			} /* next j */
		} /* if (new_len && new_len < len) */
		else {
			for (j = 0; j < len; j++) {
				iso->grid_entry[first + j] = ge;
			}
		}
		ge++;
		i++;
		first += len;
		if (i >= iso->G_max) {
			break;
		}
	}
	iso->G_max = ge;
}

static void iso2_A_del_point(ISO2 *iso, int k)
{
	int i, first, new_len, last_len, ge, j;
	
	first = 0; /* Zeile */
	i = 0; /* new horizontal grid_entry */
	ge = 0; /* old horizontal grid_entry */
	last_len = -1;
	while (TRUE) {
		/* we must have a hbar at first: */
		if (iso->hbar[first] > k) {
			cout << "iso2_A_del_point iso->hbar[first] > k" << endl;
			exit(1);
		}
		new_len = iso->hlen[first];
		if (iso->hbar[first] == k) {
			if (last_len == -1) {
				cout << "iso2_A_del_point last_len == -1" << endl;
				exit(1);
			}
			iso->hbar[first] = iso->b_t;
				/* no hbar */
			iso->hlen[first - last_len] += new_len;
			ge--;
			for (j = 0; j < new_len; j++) {
				iso->grid_entry[first + j] = ge;
			}
			ge++;
			last_len = last_len + new_len;
		}
		else {
			for (j = 0; j < new_len; j++) {
				iso->grid_entry[first + j] = ge;
			}
			ge++;
			last_len = new_len;
		}
		first += new_len;
		i++;
		if (i >= iso->G_max) {
			break;
		}
	}
	if (first != iso->v_t) {
		cout << "iso2_A_del_point first != iso->v_t" << endl;
		exit(1);
	}
	iso->G_max = ge;
}

static void check_hbar(ISO2 *iso, int k)
{
	int i, j, len, first;
	
	first = 0;
	i = 0;
	while (TRUE) {
		if (iso->hbar[first] >= k) {
			cout << "check_hbar iso->hbar[first] >= k" << endl;
			print_hbar(iso);
			exit(1);
		}
		len = iso->hlen[first];
		for (j = 0; j < len; j++) {
			if (iso->grid_entry[first + j] != i) {
				cout << "check_hbar iso->grid_entry[first + j] != i" << endl;
				print_hbar(iso);
				exit(1);
			}
		}
		i++;
		first += len;
		if (i >= iso->G_max) {
			break;
		}
	}
	if (first != iso->v_t) {
		cout << "check_hbar first != iso->v_t" << endl;
		print_hbar(iso);
		exit(1);
	}
}

static void iso2_check_iso(
	ISO2 *iso, int k, ORDERED_SET *E)
{
	int i, j, first, i1, i2, k1, ge;
	
	E->size = 0L;
	if (iso->G1->G_max != iso->G2->G_max) {
		return;
	}
	for (i = 0; i < iso->G1->G_max; i++) {
		if (iso->G1->first[i] != iso->G2->first[i]) {
			return;
		}
		if (iso->G1->len[i] != iso->G2->len[i]) {
			return;
		}
	}
	for (i = 0; i < iso->G1->G_max; i++) {
		first = iso->G1->first[i];
		i1 = iso->G1->type_idx[first];
		i2 = iso->G2->type_idx[first];
		for (j = 0; j < iso->G1->n; j++) {
			if (iso->G1->type[i1][j] != iso->G2->type[i2][j]) {
				return;
			}
		}
	}
	/* now: a bijection exists. */
	
	if (k != iso->G2->first[0]) {
		cout << "iso2_check_iso k != iso->G2->first[0]" << endl;
		exit(1);
	}
	k1 = iso->G1->q.data[k];
		/* der k-te Block liegt momentan 
		 * in der k1-ten Spalte. */
	ge = iso->G1->grid_entry[k1];
	/* links (A): nimm den Block, 
	 *    in dem k liegt. 
	 * rechts (B): alle Bloecke 
	 *   des Grideintrags ge
	 * nach E[k]. */
	first = iso->G2->first[ge];
	for (i = 0; i < iso->G2->len[ge]; i++) {
		i1 = first + i;
		i2 = iso->G2->qv.data[i1];
		/* die urspruengliche Blocknummer. */
		E->a[E->size++] = i2;
	}
}

static void iso2_print_Ei(ISO2 *iso, int i)
{
	int j;
	ORDERED_SET *p;
	
	p = &iso->E[i];
	printf("E[%d] = { ", i);
	for (j = 0; j < p->size; j++) {
		printf("%2d ", p->a[j]);
	}
	printf("}\n");
}

static void iso2_calc_grid(
	ISO2 *iso, iso_grid *G, int k,
	int *theX, int dim_n, int *d,
	short *dd, cperm *p)
/* (bislang: max_r konstant, 
 * statt R Array) jetzt: R[]
 * type_len nur ohne 
 *   zweite Ableitungen (d und dd). 
 * iso->grid_entry[] muss 
 *   bereits berechnet sein. */
{
	int len, i, j, l, x, i1, j1, x1, ge;
	int begin_d, begin_dd, d1;
	/* TDOSS *tdoss; */
	int first, tdo_n, Bj;
	combinatorics_domain Combi;
	
	G->m = iso->b_t;
	G->n = iso->G_max;
	begin_d = G->n;
	if (iso->info->f_use_d) {
		G->n++;
		begin_dd = begin_d + 1;
	}
	else {
		begin_dd = begin_d + 1;
	}
	if (iso->f_use_ddb) {
		G->n += k;
	}
	len = G->m - k;
	for (i = 0; i < len; i++) {
		for (j = 0; j < G->n; j++) {
			G->type[i][j] = 0;
		}
	}
	for (i = 0; i < G->m; i++) {
		if (i >= k) {
			G->type_idx[i] = i - k;
		}
		else {
			G->type_idx[i] = - 1;
		}
	}
	for (i = 0; i < iso->v_t; i++) {
		for (j = 0; j < iso->A_R[i]; j++) {
			/* before: max_r */
			x = theX[i * dim_n + j];
			i1 = p->data[i];
			x1 = G->q.data[x];
			if (x1 >= k) {
				ge = iso->grid_entry[i1];
				G->type[x1 - k][ge]++;
			}
		}
	}
	G->G_max = 0;
	G->first[0] = k;
	
	if (iso->info->f_use_d) {
		for (i = 0; i < G->m; i++) {
			d1 = d[i];
			i1 = G->q.data[i];
			if (i1 >= k) {
				G->type[i1 - k][begin_d] = d1;
			}
		}
	}
	if (iso->f_use_ddb) {
		for (i = 0; i < G->m; i++) {
			for (j = 0; j < G->m; j++) {
				if (j == i) {
					continue;
				}
				l = Combi.ij2k(i, j, G->m);
				d1 = dd[l];
				i1 = G->q.data[i];
				j1 = G->q.data[j];
				if (i1 >= k && j1 < k) {
					G->type[i1 - k]
					[begin_dd + j1] = d1;
				}
			}
		}
	}

	if (iso->info->f_very_verbose) {
		printf("in iso2_calc_grid() (k = %d):\n", k);
		for (i = 0; i < len; i++) {
			for (j = 0; j < G->n; j++) {
				printf("%d ", G->type[i][j]);
			}
			printf("\n");
		}
	}
	
	/* tdoss = iso->info->tdo; */
	/* tdo_n = tdoss_n(tdoss); */
	tdo_n = iso->tdo_n_t;
	first = 0;
	for (j = 0; j < tdo_n; j++) {
		/* Bj = tdoss_Bj(tdoss, j); */
		Bj = iso->tdo_B_t[j];
		if (k < first + Bj) {
			iso2_radix_sort(
			iso, G, 0, k, first + Bj - 1);
			first += Bj;
			break;
		}
		first += Bj;
	}
	for (j++; j < tdo_n; j++) {
		/* Bj = tdoss_Bj(tdoss, j); */
		Bj = iso->tdo_B_t[j];
		iso2_radix_sort(
		iso, G, 0, first, first + Bj - 1);
		first += Bj;
	}
}

static void iso2_radix_sort(
	ISO2 *iso, iso_grid *G, int radix, int first, int last)
{
	int f_found, idx, k, l, t, i1, j;
	int first0, first1, res, k1;
	cperm *perm, *perm_inv;

	if (first == last || radix == G->n) {
		/* Berech first .. last 
		 * als neuen grid entry nach G eintragen. 
		 * grid_entry[first..last] setzen. */
		k = G->G_max;
		if (G->first[k] != first) {
			cout << "iso2_radix_sort G->first[k] != first" << endl;
			exit(1);
		}
		for (l = first; l <= last; l++) {
			G->grid_entry[l] = k;
		}
		G->len[k] = last - first + 1;
		G->first[k + 1] = last + 1;
		G->G_max++;
		if (iso->info->f_very_verbose) {
			cout << "iso2_radix_sort new entry: first = " << first << " len = " << G->len[k] << "  : ";
			i1 = G->type_idx[first];
			for (j = 0; j < G->n; j++) {
				cout << G->type[i1][j] << " ";
			}
			cout << endl;
		}
		return;
	}
	for (k = first; k <= last; k++) {
		f_found = iso2_insert_idx(G, first, k - first, radix, k, &idx);
		if (idx != k) {
			/* s = (idx idx+1 ... k) 
			 * auf den aktuellen Stand 
			 * der Matrix anwenden. 
			 *   q := q * s, qv := s^-1 * qv */
			perm = &G->q;
			perm_inv = &G->qv;
			perm->mult_apply_forwc_r(idx, k - idx + 1);
			perm_inv->mult_apply_backwc_l(idx, k - idx + 1);
			t = G->type_idx[k];
			for (l = k; l > idx; l--) {
				G->type_idx[l] = G->type_idx[l - 1];
			}
			G->type_idx[idx] = t;
			/* grid_entry ist noch 
			 * nicht gesetzt, 
			 * braucht nicht mitge-
			 * schoben zu werden. */
			}
		}
	first0 = first;
	first1 = G->type_idx[first0];
	for (k = first + 1; k <= last; k++) {
		k1 = G->type_idx[k];
		res = G->type[k1][radix] - G->type[first1][radix];
		if (res > 0) {
			cout << "iso2_radix_sort not descending" << endl;
			exit(1);
		}
		if (res < 0) {
			iso2_radix_sort(iso, G, radix + 1, first0, k - 1);
			first0 = k;
			first1 = G->type_idx[first0];
		}
		if (k == last) {
			iso2_radix_sort(iso, G, radix + 1, first0, k);
		}
	}
}

static int iso2_insert_idx(
		iso_grid *G, int first, int len, int radix, int search_this, int *idx)
{
	int i, st1, cur, cur1, res;
	int f_found;
	
	st1 = G->type_idx[search_this];
	f_found = FALSE;
	for (i = 0; i < len; i++) {
		cur = first + i;
		cur1 = G->type_idx[cur];
		res = G->type[cur1][radix] - 
		G->type[st1][radix];
		if (res == 0) {
			f_found = TRUE;
		}
		if (res < 0) {
			*idx = cur;
			return f_found;
			}
		}
	*idx = first + len;
	return f_found;
}



static void print_hbar(ISO2 *iso)
{
	int i, len, first;
	
	for (i = 0; i < iso->info->v; i++) {
		printf("%d %d\n", iso->hbar[i], iso->hlen[i]);
	}
	first = 0;
	i = 0;
	while (TRUE) {
		len = iso->hlen[first];
		printf("i = %d: at %d len %d\n", i, first, len);
		i++;
		first += len;
		if (i >= iso->G_max) {
			break;
		}
	}
}


}}

