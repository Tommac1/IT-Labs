/* SDIZO N1 20A LAB02 		*/
/* Tomasz Mączkowski  		*/
/* tom.maczkowski@gmail.com */

/* System headers */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

/* Defines and constants */
#define FILE_NAME 		 "inlab02.txt"
#define PRINT_FORWARDS 	 1
#define PRINT_BACKWARDS -1
#define PRINT_DEFAULT 	 0

/* Main double-sided cyclic linear linked list node */
typedef struct tNode {
	int key;
	double d;
	char c;
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
void insertToList(Node **, const int);
/*
 * Insert X nodes with random keys to the list
 */
void insertXrandomNodes(Node ** head, int);
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
void deleteFromList(Node **, int);
/*
 * Delete whole list from heap
 */
void removeList(Node **);

//--------------------------------------------------------------------------------------------------
// Main runnable
//--------------------------------------------------------------------------------------------------

int main()
{
	int ret = 0; 			/* Returning variable */
	int X; 					/* Number of randomized nodes */
	int keys[5] = { 0 }; 	/* Keys read from file */
	FILE *fp;				/* File pointer */
	int i; 					/* Handy iterator */

	/* Called only once in function */
	srand(time(NULL));

	/* For time counting in CLOCKS_PER_SEC */
	clock_t begin;
	clock_t end;
	double time_spent = 0.0;

	fp = fopen(FILE_NAME, "r");
	if (NULL != fp) {
		/* File opened successfully */
		i = fscanf(fp, "%d %d %d %d %d %d\n",
				&X, &keys[0], &keys[1], &keys[2], &keys[3], &keys[4]);
		free(fp);
		fp = NULL;
		if (6 == i) {
			/* Read data from file successfull */
            i = 0;

        	/* Start time */
        	begin = clock();

        	/* Initialize head of the list */
        	Node *head = initList();

			/* Lookup for k1 */
			searchList(head, keys[0]);

            /* Insert x random nodes to list */
            insertXrandomNodes(&head, X);
            /* Print number of nodes in list */
            printf("Count nodes: %d\n", countNodes(head));

            /* Print first 20 elems from head in arscending order */
            printList(head, 20, PRINT_FORWARDS);

            /* Insert k2 key */
            insertToList(&head, keys[1]);
            /* Print first 20 elems from head in arscending order */
            printList(head, 20, PRINT_FORWARDS);

            /* Insert k3 key */
            insertToList(&head, keys[2]);
            /* Print first 20 elems from head in arscending order */
            printList(head, 20, PRINT_FORWARDS);

            /* Insert k4 key */
            insertToList(&head, keys[3]);
            /* Print first 20 elems from head in arscending order */
            printList(head, 20, PRINT_FORWARDS);

            /* Insert k5 key */
            insertToList(&head, keys[4]);

            /* Delete k3 key from list */
            deleteFromList(&head, keys[2]);

            /* Print first 20 elems from head in arscending order */
            printList(head, 20, PRINT_FORWARDS);

            /* Delete k2 key from list */
            deleteFromList(&head, keys[1]);

            /* Print first 20 elems from head in arscending order */
            printList(head, 20, PRINT_FORWARDS);

            /* Delete k5 key from list */
            deleteFromList(&head, keys[4]);

            /* Print number of nodes in list */
            printf("Count nodes: %d\n", countNodes(head));

            /* Search for k5 key in list */
            searchList(head, keys[4]);

            /* Print first 11 elems from head in descending order */
            printList(head, 11, PRINT_BACKWARDS);

            /* Delete whole list */
            printf("Removing list...\n");
            removeList(&head);

            /* Print first 11 elems from head in descending order */
            printList(head, 11, PRINT_BACKWARDS);

            /* Print number of nodes in list */
            printf("Count nodes: %d\n", countNodes(head));

        	/* Stop time */
        	end = clock();
        	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        	printf("Time spent: %.4fs\n", time_spent);
		}
		else {
			/* Error reading from file */
			ret = -1;
			printf("Error: reading from file\n");
		}
	}
	else {
		/* Error opening a file */
		ret = -1;
		printf("Error opening a file: %s\n", FILE_NAME);
	}

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
		ret->d = rand();
		ret->c = 'T';
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

void insertToList(Node **head, const int key)
{
	Node *next = *head;
	Node *prev = *head;
	Node *new = NULL;

	if (isEmpty(*head)) {
        /* List not empty */
		if (1 == countNodes(*head)) {
			/* List with 1 element */
			new = createNode(key, next, prev);
			if (NULL != new) {
				prev->next = new;
				next->prev = new;
			}
			else {
				printf("Error inserting to list.\n");
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
					printf("Error: multiple key value %d\n", key);
				}
				else {
					/* Given key not present */
					new = createNode(key, next, prev);
					if (NULL != new) {
						prev->next = new;
						next->prev = new;
					}
					else {
						printf("Error inserting to list.\n");
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
			printf("Error inserting to list.\n");
		}
	}
}

void insertXrandomNodes(Node ** head, int X)
{
	int i = 0;
	int key = 0;

	for(i = 0; i < X; i++) {
		key = (rand() % 99901) + 99;
		insertToList(head, key);
	}
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

	printf("Node with key %d %s found in the list.\n", key,
			(NULL == lookup) ? "not" : "");

	return lookup;
}

void deleteFromList(Node **head, int key)
{
	Node *next = NULL;
	Node *prev = NULL;
	Node *temp;

	if (NULL != *head) {
		temp = *head;
		next = (*head)->next;
		prev = (*head)->prev;

		while (key > temp->key) {
			prev = temp;
			temp = next;
			next = next->next;
		}

		if (key == temp->key) {
			prev->next = next;
			next->prev = prev;

			free(temp);
			temp = NULL;
		}
		else {
			printf("Couldn't find node with key %d for removal.\n", key);
		}
	}
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
