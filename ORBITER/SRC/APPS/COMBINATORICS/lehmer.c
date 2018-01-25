// all permutations of S_n by lehmercode
//
// Anton Betten, 8.1.2001

#include <stdio.h>
#include <stdlib.h>

#define TRUE 1
#define FALSE 0

#define MAX_N 100

void first_lehmercode(int n, int *v)
{
	int i;
	
	for (i = 0; i < n; i++)
		v[i] = 0;
}

int next_lehmercode(int n, int *v)
{
	int i;
	
	for (i = 0; i < n; i++) {
		if (v[i] < n - 1 - i) {
			v[i]++;
			for (i--; i >= 0; i--)
				v[i] = 0;
			return TRUE;
			}
		}
	return FALSE;
}

void print_lehmercode(int n, int *v) 
{
	int i;
	
	printf("code: [ ");
	for (i = 0; i < n; i++) {
		printf("%d", v[i]);
		if (i < n - 1) 
			printf(", ");
		}
	printf("]");
}

void print_permutation(int n, int *v) 
{
	int i;
	
	printf("perm: [ ");
	for (i = 0; i < n; i++) {
		printf("%d", v[i]);
		if (i < n - 1) 
			printf(", ");
		}
	printf("]");
}

void print_permutation_cyclic_notation(int n, int *v) 
{
	int i, seen[MAX_N], first, cur, next;
	
	for (i = 0; i < n; i++)
		seen[i] = FALSE;
	for (i = 0; i < n; i++) {
		if (seen[i])
			continue;
		first = i;
		cur = i;
		seen[cur] = TRUE;
		printf("(%d", first);
		while (TRUE) {
			next = v[cur];
			if (next == first) {
				printf(")");
				break;
				}
			cur = next;
			seen[cur] = TRUE;
			printf(", %d", cur);
			}
		}
}

void lehmercode_to_permutation(int n, int *code, int *perm)
{
	int digits[MAX_N], i, j, k;
	
	for (i = 0; i < n; i++)
		digits[i] = i;
	
	for (i = 0; i < n; i++) {

		// digits is an array of length n - i

		k = code[i];
		perm[i] = digits[k];
		for (j = k; j < n - i - 1; j++) 
			digits[j] = digits[j + 1];
		}
}

int main(int argc, char **argv)
{
	int n, code[MAX_N], perm[MAX_N], l = 0;
	
	if (argc < 1) {
		printf("arguments missing: give me n, please!\n");
		exit(1);
		}
	sscanf(argv[1], "%d", &n);
	printf("n=%d\n", n);
	first_lehmercode(n, code);
	do {
		l++;
		printf("%d: ", l);
		print_lehmercode(n, code);
		lehmercode_to_permutation(n, code, perm);
		printf("\t");
		print_permutation(n, perm);
		printf("\t");
		print_permutation_cyclic_notation(n, perm);
		printf("\n");
		
		} while (next_lehmercode(n, code));
}
