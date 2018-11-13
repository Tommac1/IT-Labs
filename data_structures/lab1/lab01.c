// SDIZO N1 20A LAB01
// Tomasz Mączkowski
// tom.maczkowski@gmail.com
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct s {
	int i;
	char c;
	float f;
};

typedef struct s s1;

void sort(s1 ** s, int n);
void swap(s1 * ss1, s1 * ss2);
s1 ** draw(int n);
void erase(s1 ** s, int n);
int countChars(s1 ** s, int n, char c);

int main() {
	srand(time(NULL));
	
	int N;
	char X;
	int i;
	int count;
	
	FILE* fp = fopen("inlab01.txt", "r"); 
	if (NULL == fp) {
		return -1;
	}
	fscanf (fp, "%d %c", &N, &X); 
	fclose(fp);
	
	s1 ** ssy;
	
	clock_t begin, end; 
	double time_spent; 
	begin = clock(); 
	
	ssy = draw(N);
	sort(ssy, N);
	count = countChars(ssy, N, X);
	
	for (i = 0; i < 20; ++i) {
		printf("struct %d: i = %d, c = %c, f = %.2f\n", 
			i, ssy[i]->i, ssy[i]->c, ssy[i]->f);
	}
	
	printf("Count of char %c: %d\n", X, count);
	
	erase(ssy, N);
	
	end = clock(); 
	time_spent = (double)(end - begin); // CLOCKS_PER_SEC; 
	
	printf("Time spent: %.2f.\n", time_spent);
	
	return 0;
}

s1 ** draw(int n) {
	s1 ** structs = malloc(sizeof(s1 *) * n);
	
	int i;
	int j;
	int present = 0;
	for (i = 0; i < n; ++i) {
		structs[i] = malloc(sizeof(s1));
		do {
			structs[i]->i = (rand() % 10001) - 1000;
			present = 0;
			for (j = 0; (j < i) && (present == 0); ++j) {
				if (structs[j]->i == structs[i]->i) {
					present = 1;
				}
			}
		} while (present == 1);
		
		structs[i]->c = (rand() % 23) + 66;
		structs[i]->f = 1000 + i;
	}
	
	return structs;
}

void erase(s1 ** s, int n) {
	int i;
	for (i = 0; i < n; ++i) {
		free(s[i]);
	}
	
	free(s);
}

void sort(s1 ** s, int n) {
	int i;
	int j;
	char swapped = 1;
	
	for (i = 0; (i < n - 1) && (swapped == 1); ++i) {
		swapped = 0;
		for (j = 0; j < n - i - 1; ++j) {
			if (s[j]->i > s[j + 1]->i) {
				swap(s[j + 1], s[j]);
				swapped = 1;
			}
		}
	}
}

void swap(s1 * ss1, s1 * ss2) {
	s1 temp = *ss1;
	*ss1 = *ss2;
	*ss2 = temp;
}

int countChars(s1 ** s, int n, char c) {
	int i;
	int ret = 0;
	
	for (i = 0; i < n; ++i) {
		if (s[i]->c == c) {
			++ret;
		}
	}
	
	return ret;
}
