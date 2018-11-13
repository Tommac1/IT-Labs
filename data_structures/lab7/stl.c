/* SDIZO N1 20A LAB05 		*/
/* Tomasz Mączkowski  		*/
/* tom.maczkowski@gmail.com */

/* System headers */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <set>

#define MAX_DATA 	32
#define FILE_PATH	"rand.txt"
#define MODE_INSERT	0
#define MODE_SEARCH	1
#define MODE_DELETE	2
#define RANDOM_MAX	10000000


int processRandFileSet(std::set<int> *structure, int N, const char *path, const int mode);
int insertToSet(std::set<int> *structure, int key);
int findInSet(std::set<int> *structure, int key);
int removeFromSet(std::set<int> *structure, int key);
int removeXrandomNodesFromSet(std::set<int> *structure, int N);
int findXrandomNodesInSet(std::set<int> *structure, int N);
int insertXrandomNodesToSet(std::set<int> *structure, int N);

int main(int argc, char *argv[])
{
	int ret = -1; 	/* Returning variable */
	int N = 0; 		/* Number of randomized nodes */
//	int x = 0;		/* Switch: random number generation or linear */
	int res = 0;

	if (3 != argc) {
		printf("usage: %s N x\n", argv[0]);
		return ret;
	}

	N = atoi(argv[1]);
//	x = atoi(argv[2]);

	/* Called only once in function */
	srand(7);

	/* For time counting (in CLOCKS_PER_SEC) */
	clock_t begin = 0;
	clock_t end = 0;
	double time_spent = 0.0;

	/* Init empty set */
	std::set<int> *structure = new std::set<int>;

	printf("%s %s %s\n", argv[0], argv[1], argv[2]);

	/* Program flow */

	/* TASK 2 */
	begin = clock();
	res = processRandFileSet(structure, N, FILE_PATH, MODE_INSERT);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Inserting time spent: %.0fms\n"
		   "Inserted %d nodes\n", time_spent * 1000, res);

	/* TASK 3 */
	begin = clock();
	res = processRandFileSet(structure, N, FILE_PATH, MODE_SEARCH);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Finding time spent: %.0fms\n"
		   "Found %d nodes\n", time_spent * 1000, res);

	/* TASK 4 */
	begin = clock();
	res = findXrandomNodesInSet(structure, N);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Finding time spent: %.0fms\n"
		   "Found %d nodes\n", time_spent * 1000, res);

	/* TASK 5 */
	begin = clock();
	res = removeXrandomNodesFromSet(structure, N);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Deleting time spent: %.0fms\n"
		   "Deleted %d nodes\n", time_spent * 1000, res);

	/* TASK 6 */
	begin = clock();
	res = processRandFileSet(structure, (N / 2), FILE_PATH, MODE_DELETE);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Deleting time spent: %.0fms\n"
		   "Deleted %d nodes\n", time_spent * 1000, res);


	printf("DONE\n");

	/* Memory leaks */
	delete(structure);

	return ret;
}

int processRandFileSet(std::set<int> *structure, int N, const char *path, const int mode)
{
	FILE *fp;
	int result = 0;
	int key;
	char *line = NULL;
	size_t len = 0;
	int read = 0;

	fp = fopen(path, "r");

	if (NULL != fp)
	{
		for (; N > 0; --N)
		{
			if (0 < (read = getline(&line, &len, fp)))
			{
				key = atoi(line);
				if (MODE_INSERT == mode)
				{
					result += insertToSet(structure, key);
				}
				else if (MODE_SEARCH == mode)
				{
					result += findInSet(structure, key);
				}
				else if (MODE_DELETE == mode)
				{
					result += removeFromSet(structure, key);
				}
				else
				{
					printf("UNKNOWN MODE %d\n", mode);
					return -1;
				}
			}

		}
		fclose(fp);
	}

	return result;
}

int insertToSet(std::set<int> *structure, int key)
{
	std::pair<std::set<int>::iterator, bool> inserted;
	int result = 0;

	inserted = structure->insert(key);

	if (false != inserted.second)
	{
		result = 1;
	}

	return result;
}

int findInSet(std::set<int> *structure, int key)
{
	int result = 0;
	std::set<int>::iterator it = structure->find(key);

	if(it != structure->end())
	{
		result = 1;
	}

	return result;
}

int removeFromSet(std::set<int> *structure, int key)
{
	int result = 0;

	result = structure->erase(key);

	return result;
}

int removeXrandomNodesFromSet(std::set<int> *structure, int N)
{
	int key = 0;
	int removed = 0;

	/* RAND, RAND, RAND ... */
	for(; 0 < N; --N)
	{
		key = (rand() % RANDOM_MAX);
		removed += removeFromSet(structure, key);
	}

	return removed;
}

int findXrandomNodesInSet(std::set<int> *structure, int N)
{
	int key = 0;
	int found = 0;

	/* RAND, RAND, RAND ... */
	for(; 0 < N; --N)
	{
		key = (rand() % RANDOM_MAX);
		found += findInSet(structure, key);
	}

	return found;
}

int insertXrandomNodesToSet(std::set<int> *structure, int N)
{
	int key = 0;
	int inserted = 0;

	/* RAND, RAND, RAND ... */
	for(; 0 < N; --N)
	{
		key = (rand() % RANDOM_MAX);
		inserted += insertToSet(structure, key);
	}

	return inserted;
}
