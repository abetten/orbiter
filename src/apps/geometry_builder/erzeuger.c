/* erzeuger.c */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	char str[1000];
	char *pfrom, *plen, *control_file;
	int from, len, i;
	
	if (argc != 4) {
		printf("usage: erzeuger from len control_file\n");
		exit(1);
		}
	pfrom = argv[1];
	plen = argv[2];
	control_file = argv[3];
	sscanf(pfrom, "%d", &from);
	sscanf(plen, "%d", &len);
	printf("erzeuger: from = %d len = %d\n", from, len);
	for (i = 0; i < len; i++) {
		printf("calling gsdb2005.out with no = %d\n", from + i);
		fflush(stdout);
		sprintf(str, "gsdb2005.out -no %d %s", from + i, control_file);
		system(str);
		}


	return 0;
}

