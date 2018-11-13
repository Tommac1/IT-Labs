/* SDIZO N1 20A LAB05 		*/
/* Tomasz Mączkowski  		*/
/* tom.maczkowski@gmail.com */

/* System headers */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

/* Defines and constants */
#define PRINT_FORWARDS 	 1
#define PRINT_BACKWARDS -1
#define PRINT_DEFAULT 	 0

/* Randomizing modes */
#define ALL_RANDOM			0
#define RANDOM_WITH_LINEAR	1
#define FILE_NAME 		 "inlab02.txt"

#define IS_EVEN(i) (i & 1)

/* Main double-sided cyclic linear linked list node */
typedef struct tNode {
	int key;
	struct tNode *next;
	struct tNode *prev;
} Node;

//--------------------------------------------------------------------------------------------------
// Function declarations
//--------------------------------------------------------------------------------------------------

/*
 * Init empty list
 */
Node *initList();
/*
 * Create node on heap
 */
Node *createNode(int, Node *, Node *);
/*
 * Insert new node to list
 */
int insertToList(Node **, const int);
/*
 * Insert X nodes with random keys to the list
 */
int insertXrandomNodes(Node ** head, int);
/*
 * Count nodes in list
 */
int countNodes(Node * const);
/*
 * Print first X elements of list with given mode (forward, backward)
 */
void printList(Node * const, int, int);
/*
 * Check if list is empty
 */
bool isEmpty(Node *);
/*
 * Check is list is full (ie. no memory
 */
bool isFull(Node *);
/*
 * Update head to be the lowest key element
 */
void updateHead(Node **);
/*
 * Search list to find given value
 */
Node *searchList(Node *, int);
/*
 * Delete node with given key from list
 */
int deleteFromList(Node **, int);
/*
 * Delete whole list from heap
 */
void removeList(Node **);

int removeXrandomNodesList(Node **root, int N);
int insertXrandomNodesList(Node **root, int N, int x);
int lookForXrandomNodesList(Node **root, int N);

//--------------------------------------------------------------------------------------------------
// Main runnable
//--------------------------------------------------------------------------------------------------

int main(int argc, char *argv[])
{
	int ret = -1; 	/* Returning variable */
	int N = 0; 		/* Number of randomized nodes */
	int x = 0;		/* Switch: random number generation or linear */
	int res = 0;

	if (3 != argc) {
		printf("usage: %s N x\n", argv[0]);
		return ret;
	}

	N = atoi(argv[1]);
	x = atoi(argv[2]);

	/* Called only once in function */
	srand(0);

	/* For time counting (in CLOCKS_PER_SEC) */
	clock_t begin = 0;
	clock_t end = 0;
	double time_spent = 0.0;

	/* Init empty tree */
	Node *root = initList();

	printf("%s %s %s\n", argv[0], argv[1], argv[2]);

	/* Program flow */
	/* Insertion */
	begin = clock();
	res = insertXrandomNodesList(&root, N, x);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Inserting time spent: %.4fms.\n"
			"Inserted %d nodes.\n", time_spent * 1000, res);

	/* Searching */
	begin = clock();
	res = lookForXrandomNodesList(&root, N);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Finding time spent: %.4fms.\n"
			"Found %d nodes.\n", time_spent * 1000, res);

	/* Deletion */
	begin = clock();
	res = removeXrandomNodesList(&root, N);
	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Deleting time spent: %.4fms.\n"
			"Deleted %d nodes.\n", time_spent * 1000, res);

	printf("DONE\n");

	/* Memory leaks */
	removeList(&root);

	return ret;
}

//--------------------------------------------------------------------------------------------------
// Function definitions
//--------------------------------------------------------------------------------------------------

Node *initList()
{
	return NULL;
}

Node *createNode(int key, Node *next, Node *prev)
{
	Node *ret = malloc(sizeof(Node));
	if (NULL != ret) {
		ret->key = key;
		ret->next = next;
		ret->prev = prev;
	}
	return ret;
}

/* Returns 0 if full - 1 if not. */
bool isFull(Node *head)
{
	int retL = 0;
	Node *new = malloc(sizeof(Node));
	if (NULL != new) {
		retL = 1;
		free(new);
	}
	return retL;
}

/* Returns 0 if empty - 1 if not. */
bool isEmpty(Node *head)
{
	int retL = 1;
	if (NULL == head) {
		retL = 0;
	}
	return retL;
}

int insertToList(Node **head, const int key)
{
	Node *next = *head;
	Node *prev = *head;
	Node *new = NULL;
	int inserted = 1;

	if (isEmpty(*head)) {
        /* List not empty */
		if (1 == countNodes(*head)) {
			/* List with 1 element */
			new = createNode(key, next, prev);
			if (NULL != new) {
				prev->next = new;
				next->prev = new;

				if (key < (*head)->key) {
					/* New key becomes the head */
					*head = new;
				}
			}
			else {
				//printf("Error inserting to list.\n");
				inserted = 0;
			}
		}
		else {
			if (key < next->key) {
				/* Key is smaller - insert at beginning */
				prev = (*head)->prev;
				next = *head;

				new = createNode(key, next, prev);
				if (NULL != new) {
					prev->next = new;
					next->prev = new;
					/* New element is at beginning of the list */
					*head = new;
				}
			}
			else {
				/* Iterate to find perfect spot to insert value */
				while (next->key < key && (next->key >= prev->key)) {
					prev = next;
					next = next->next;
				}
				if (next->key == key) {
					//printf("Error: multiple key value %d\n", key);
					inserted = 0;
				}
				else {
					/* Given key not present */
					new = createNode(key, next, prev);
					if (NULL != new) {
						prev->next = new;
						next->prev = new;
					}
					else {
						//printf("Error inserting to list.\n");
						inserted = 0;
					}
				}
			}
		}
	//updateHead(head);
	}
	else {
		/* List is empty */
        *head = createNode(key, *head, *head);
        if (NULL != *head) {
            (*head)->next = *head;
            (*head)->prev = *head;
        }
		else {
			//printf("Error inserting to list.\n");
			inserted = 0;
		}
	}
	return inserted;
}

int insertXrandomNodesList(Node **root, int N, int x)
{
	int key = 0;
	int linear = 1;
	int inserted = 0;

	if (ALL_RANDOM == x) {
		/* RAND, RAND, RAND ... */
		for(; 0 < N; --N) {
			key = (rand() % 100000);
			inserted += insertToList(root, key);
		}
	}
	else if (RANDOM_WITH_LINEAR == x) {
		/* RAND, 1, RAND, 2, RAND, 3 RAND, 4 ... */
		for(; 0 < N; --N) {
//			if (0 == isEven(N)) {
			if (0 == IS_EVEN(N)) {
				key = (rand() % 100000);
				inserted += insertToList(root, key);
			}
			else {
				inserted += insertToList(root, linear);
				++linear;
			}
		}
	}
	else {
		printf("Unknown randomizing mode: %d\n", x);
	}
	return inserted;
}

int lookForXrandomNodesList(Node **root, int N)
{
	int key = 0;
	int found = 0;
	Node *ret = NULL;

	/* RAND, RAND, RAND ... */
	for(; 0 < N; --N) {
		key = (rand() % 100000);
		if (NULL != (ret = searchList(*root, key))) {
			++found;
		}
	}
	return found;
}

int removeXrandomNodesList(Node **root, int N)
{
	int key = 0;
	int removed = 0;

	/* RAND, RAND, RAND ... */
	for(; 0 < N; --N) {
		key = (rand() % 100000);
		if (0 != deleteFromList(root, key)) {
			++removed;
		}
	}
	return removed;
}

int countNodes(Node * const head)
{
	int ret = 0;
	Node *next = head;
	Node *prev = NULL;

	if (next != NULL) {
		ret++;
		prev = next;
		next = next->next;
		while (next->key > prev->key) {
			ret++;
			prev = next;
			next = next->next;
		}
	}

	return ret;
}

void printList(Node * const head, int howMany, int mode)
{
    Node *next = head;
    Node *prev = head;
    int firstKey = 0;
    int it = 0;

    if ((PRINT_DEFAULT == mode) || (PRINT_FORWARDS == mode)) {
		if (next != NULL) {
			if (1 == countNodes(head)) {
				printf("%d\n", head->key);
			}
			else {
				while ((next->key >= prev->key) && (0 != howMany)) {
					it++;
					printf("%d->", next->key);
					prev = next;
					next = next->next;
					if (it == 10) {
						/* For better readability */
						printf("\n");
						it = 0;
					}
					howMany--;
				}
				printf("\n");
			}
		}
    }
    else if (PRINT_BACKWARDS == mode) {
    	if (next != NULL) {
			if (1 == countNodes(head)) {
				printf("%d\n", head->key);
			}
			else {
				prev = head->prev;
				next = head->prev;
				firstKey = head->key;
				while ((next->key >= prev->key) && (0 != howMany)) {
					it++;
					printf("%d<-", prev->key);
					next = prev;
					prev = prev->prev;
					if (it == 10) {
						/* For better readability */
						printf("\n");
						it = 0;
					}
					howMany--;
				}
				printf("%d\n", firstKey);
			}
		}
    }
    else {
    	printf("Invalid printing mode\n");
    }
}

void updateHead(Node **head)
{
	Node *next = NULL;
	Node *prev = *head;

	if (NULL != *head) {
		if ((*head)->next != NULL) {
			next = (*head)->next;
			if (next->key != prev->key) {
				while (next->key > prev->key) {
					prev = next;
					next = next->next;
				}
				*head = next;
			}
		}
	}
}

Node *searchList(Node *head, int key)
{
	Node *lookup = head;
	Node *prev = NULL;

	if (NULL != lookup) {
		prev = head;
		while ((lookup->key < key) && (lookup->key >= prev->key)) {
			prev = lookup;
			lookup = lookup->next;
		}

		if (lookup->key != key) {
			lookup = NULL;
		}
	}

//	printf("Node with key %d %s found in the list.\n", key,
//			(NULL == lookup) ? "not" : "");

	return lookup;
}

int deleteFromList(Node **head, int key)
{
	Node *next = NULL;
	Node *prev = NULL;
	Node *temp;
	int removed = 0;

	if (NULL != *head) {
		temp = *head;
		next = (*head)->next;
		prev = (*head)->prev;

		while ((key > temp->key) && (next != *head)) {
			prev = temp;
			temp = next;
			next = next->next;
		}

		if (key == temp->key) {
			prev->next = next;
			next->prev = prev;

			if (*head == temp) {
				*head = (*head)->next;
			}

			free(temp);
			temp = NULL;
			removed = 1;
		}
		else {
//			printf("Couldn't find node with key %d for removal.\n", key);
		}
	}
	return removed;
}

void removeList(Node **head)
{
	Node *prev = NULL;
	Node *next = NULL;

	if (NULL != *head) {
		/* List not empty */
		next = (*head)->next;
		prev = *head;

		while (*head != next->next) {
			/* While not list overflow - remove every node */
			free(prev);
			prev = next;
			next = next->next;
		}
		/* Dangerous dangling pointers */
		*head = NULL;
	}
}
