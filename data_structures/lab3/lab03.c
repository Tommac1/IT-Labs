/* SDIZO N1 20A LAB03 		*/
/* Tomasz Mączkowski  		*/
/* tom.maczkowski@gmail.com */

/* System headers */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

/* Constant and type defines */
#define FILE_NAME 	"inlab03.txt"
#define ARRAY_SIZE 	10
#define STACK_SIZE 	256
#define FULL_STACK 	255
#define EMPTY_STACK 0

/* Tree printing modes */
#define INORDER_MODE	0
#define PREORDER_MODE	1
#define POSTORDER_MODE	2

static int visitedNodes = 0;

/* Grade of the node */
#define NODE_IS_LEAF 	0
#define NODE_1ST_GRADE	1
#define NODE_2ND_GRADE	2

/* Main tree node */
struct sNode {
	int key;
	char c[ARRAY_SIZE];
	struct sNode *left;
	struct sNode *right;
};
typedef struct sNode Node;

/* Useful stack structure for storing met nodes, while searching */
struct sStack {
	int ptr;
	Node **nodeStack;
};
typedef struct sStack Stack;

//--------------------------------------------------------------------------------------------------
// Function declarations
//--------------------------------------------------------------------------------------------------

/*
 * Init empty tree
 */
Node *initTree();
/*
 * Create tree node with given key and left, right children
 */
Node *createNodeTree(const int, Node *, Node *);
/*
 * Insert node to the tree of given root and key
 */
void insertNodeTree(Node **, const int);
/*
 * Remove whole tree from memory
 */
void removeTree(Node **);
/*
 * Insert X nodes with random key to the tree of given root
 */
void insertXrandomNodesTree(Node **, int);
/*
 * Search tree of given root for key value and then
 * return pointer path of nodes in stack, if found - or
 * print an error and return NULL
 */
Stack *searchTree(Node *, const int);
/*
 * Removes node with given key
 */
void removeNodeTree(Node **, const int);
/*
 * Remove node which is leaf
 */
void removeNode0thGradeTree(Node **, Node *, Stack *);
/*
 * Remove node which has 1 child
 */
void removeNode1stGradeTree(Node **, Node *, Stack *);
/*
 * Remove node which has both children
 */
void removeNode2ndGradeTree(Node **, Node *, Stack *);
/*
 * Calculate the grade of the node: 0 - leaf, 1 - one child,
 * 2 - both children
 */
int calculateGradeOfNodeTree(Node *);
/*
 * Print tree in inorder mode (left -> parent -> right)
 */
void printInorderTree(Node *);
/*
 * Print tree in preorder mode (parent -> left -> right)
 */
void printPreorderTree(Node *);
/*
 * Print tree in postorder mode (left -> right -> parent)
 */
void printPostorderTree(Node *);
/*
 * Print tree with given mode:
 * 	INORDER_MODE
 *  PREORDER_MODE
 *  POSTORDER_MODE
 */
void printTree(Node *, int);
/*
 * Init empty stack
 */
Stack *initStack();
/*
 * Delete the stack from memory
 */
void removeStack(Stack *);
/*
 * Push tree node to stack
 */
void pushStack(Stack *, Node *);
/*
 * Pop tree node from stack
 */
Node *popStack(Stack *);
/*
 * Check if stack is empty
 */
bool isEmptyStack(Stack *);
/*
 * Check if stack is full
 */
bool isFullStack(Stack *);

//--------------------------------------------------------------------------------------------------
// Main runnable
//--------------------------------------------------------------------------------------------------

int main(void) {
	int ret = 0; 			/* Returning variable */
	int x = 0; 				/* Number of randomized nodes */
	int keys[4] = { 0 }; 	/* Keys read from file */
	FILE *fp = NULL;
	int i = 0; 				/* Handy iterator */

	/* Called only once in function */
	srand(time(NULL));

	/* For time counting (in CLOCKS_PER_SEC) */
	clock_t begin = 0;
	clock_t end = 0;
	double time_spent = 0.0;

	fp = fopen(FILE_NAME, "r");
	if (NULL != fp) {
		/* File opened successfully */
		i = fscanf(fp, "%d %d %d %d %d\n",
				&x, &keys[0], &keys[1], &keys[2], &keys[3]);
		/* File pointer no longer used */
		free(fp);
		fp = NULL;

		if (5 == i) {
			/* Success reading from file */
			i = 0;

			/* Start time */
			begin = clock();

			/* Init empty tree */
			Node *root = initTree();

			/* Program flow */
			removeNodeTree(&root, keys[0]);

			insertNodeTree(&root, keys[0]);

			insertXrandomNodesTree(&root, x);

			printTree(root, INORDER_MODE);

			printTree(root, PREORDER_MODE);

			insertNodeTree(&root, keys[1]);

			printTree(root, INORDER_MODE);

			insertNodeTree(&root, keys[2]);

			insertNodeTree(&root, keys[3]);

			removeNodeTree(&root, keys[0]);

			printTree(root, PREORDER_MODE);

			searchTree(root, keys[0]);

			removeNodeTree(&root, keys[1]);

			printTree(root, INORDER_MODE);

			removeNodeTree(&root, keys[2]);

			removeNodeTree(&root, keys[3]);

			/* Stop time */
			end = clock();
			time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
			printf("Time spent: %.4fs.\n", time_spent);

			/* Memory leaks */
			removeTree(&root);
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

void removeTree(Node **root)
{
	if (NULL != *root) {

		removeTree(&(*root)->left);
		removeTree(&(*root)->right);

		/* Free the heap space */
		free(*root);
		/* Dangerous dangling pointers */
		*root = NULL;
	}
}

Node *createNodeTree(const int key, Node *left, Node *right)
{
	Node *new = malloc(sizeof(Node));
	if (NULL != new) {
		new->key = key;
		/* Fancy way to convert from int to char* */
		snprintf(new->c, ARRAY_SIZE, "%d", key);
		new->left = left;
		new->right = right;
	}
	return new;
}

void insertNodeTree(Node **root, const int key)
{
	Node *new = *root;
	Node *parent = NULL;
	/* Indicator which child node of parent to connect */
	bool wentLeft = false;
	bool keyPresent = false;

	if (NULL != *root) {
		/* Tree is not empty */
		while (NULL != new) {
			parent = new;
			if (key < new->key) {
				new = new->left;
				wentLeft = true;
			}
			else if (key > new->key) {
				new = new->right;
				wentLeft = false;
			}
			else {
				/* Multiple key value not supported */
				new = NULL;
				keyPresent = true;
			}
		}
		if (false == keyPresent) {
			/* Unique key - able to create */
			new = createNodeTree(key, NULL, NULL);

			/*
			 * Check what was the last transition between nodes
			 * and then connect corresponding child node of parent.
			 */
			if (false != wentLeft) {
				parent->left = new;
			}
			else {
				parent->right = new;
			}
		}
		else {
			/* Key already exist */
			printf("Error: multiple key value (%d).\n", key);
		}
	}
	else {
		/* Tree is empty */
		*root = createNodeTree(key, NULL, NULL);
	}
}

Node *initTree()
{
	return NULL;
}

void insertXrandomNodesTree(Node **root, int X) {
	int key = 0;

	for(; 0 < X; --X) {
		key = (rand() % 20001) - 10000;
		insertNodeTree(root, key);
	}
}

Stack *searchTree(Node *root, const int key)
{
	Stack *ret = initStack();
	Node *temp = root;

	while ((NULL != temp) && (key != temp->key)) {
		/* Iterate until we find key or leave the tree */
		pushStack(ret, temp);
		if (key < temp->key) {
			temp = temp->left;
		}
		else if (key > temp->key) {
			temp = temp->right;
		}
	}
	if (NULL == temp) {
		/* We left the tree */
		ret = NULL;
		printf("Cannot find node with key (%d).\n", key);
	}
	else if (key == temp->key) {
		/* found the key */
		pushStack(ret, temp);
	}

	return ret;

}

void removeNodeTree(Node **root, const int key)
{
	Stack *path = searchTree(*root, key);
	Node *removed = NULL;
	int removedGrade = 0;

	if (NULL != path) {
		/* Node found in a tree */
		removed = popStack(path);

		removedGrade = calculateGradeOfNodeTree(removed);
		switch (removedGrade) {
		case NODE_IS_LEAF:
			/* Case 1: node is a leaf */
			removeNode0thGradeTree(root, removed, path);
			break;
		case NODE_1ST_GRADE:
			/* Case 2: node has only one child */
			removeNode1stGradeTree(root, removed, path);
			break;
		case NODE_2ND_GRADE:
			/* Case 3: node has both children */
			removeNode2ndGradeTree(root, removed, path);
			break;
		default:
			/* Oops... Something went wrong */
			printf("Wrongly calculated grade of tree node (%d grade).\n", removedGrade);
			break;
		}

		removeStack(path);
	}
	else {
		/* Did not found node in a tree */
		printf("Cannot remove non-existing node.\n");
	}
}

void removeNode0thGradeTree(Node **root, Node *remove, Stack *path)
{
	Node *parent = NULL;

	if (remove != *root) {
		/* Removed node is not a root */
		parent = popStack(path);
		if (remove->key > parent->key) {
			parent->right = NULL;
		}
		else {
			parent->left = NULL;
		}
		free(remove);
	}
	else {
		/* Removed node is a root */
		free(*root);
		*root = NULL;
	}
}

void removeNode1stGradeTree(Node **root, Node *remove, Stack *path)
{
	Node *parent = NULL;

	/* Has left child? If not - it has to have right child. */
	bool hasLeft = ((NULL == remove->left) ? false : true);

	if (remove != *root) {
		/* Removed node is not a root */
		parent = popStack(path);
		if (remove->key > parent->key) {
			parent->right = (hasLeft ? remove->left : remove->right);
		}
		else {
			parent->left = (hasLeft ? remove->left : remove->right);
		}
	}
	else {
		/* Removed node is a root */
		*root = (hasLeft ? remove->left : remove->right);
	}

	free(remove);
}

void removeNode2ndGradeTree(Node **root, Node *remove, Stack *path)
{
	Node *parent = NULL;
	Node *temp = NULL;
	Node *tempParent = NULL;
	int gradeOfTemp = 0;


	if (remove != *root) {
		/* Removed node is not a root, so we can get its parent */
		parent = popStack(path);
	}

	/* Replace removed node with its successor */
	temp = remove->right;
	while (NULL != temp->left) {
		tempParent = temp;
		temp = temp->left;
	}

	gradeOfTemp = calculateGradeOfNodeTree(temp);
	if (NODE_IS_LEAF == gradeOfTemp) {
		/* Went all way left and met an leaf */
		temp->left = remove->left;
		if (NULL != tempParent) {
			/* If went deeper than 2 levels */
			temp->right = remove->right;
			tempParent->left = NULL;
		}
	}
	else if (NODE_1ST_GRADE == gradeOfTemp) {
		/* Went all way left and met a node with right child */
		tempParent->left = temp->right;
		temp->left = remove->left;
		temp->right = remove->right;
	}

	if (remove != *root) {
		/* Connect parent left/right child with replaced node */
		if (remove->key < parent->key) {
			parent->left = temp;
		}
		else {
			parent->right = temp;
		}
	}
	else {
		/* Root was removed */
		*root = temp;
	}

	free(remove);
}

int calculateGradeOfNodeTree(Node *node)
{
	int ret = 0;

	if (NULL != node->left)
		++ret;
	if (NULL != node->right)
		++ret;

	return ret;
}

void printTree(Node *root, int mode)
{
	visitedNodes ^= visitedNodes;

	switch (mode) {
	case INORDER_MODE:
		printInorderTree(root);
		break;

	case PREORDER_MODE:
		printPreorderTree(root);
		break;

	case POSTORDER_MODE:
		printPostorderTree(root);
		break;

	default:
		printf("Unknown printing mode.\n");
		break;
	}

	if (0 != visitedNodes) {
		printf("\nVisited nodes: %d.\n", visitedNodes);
	}
}

void printInorderTree(Node *root)
{
	if (NULL != root) {
		printInorderTree(root->left);

		printf("%d ", root->key);
		++visitedNodes;

		printInorderTree(root->right);
	}
}

void printPreorderTree(Node *root)
{
	if (NULL != root) {
		printf("%d ", root->key);
		++visitedNodes;

		printInorderTree(root->left);

		printInorderTree(root->right);
	}
}

void printPostorderTree(Node *root)
{
	if (NULL != root) {
		printInorderTree(root->left);

		printInorderTree(root->right);

		printf("%d ", root->key);
		++visitedNodes;
	}
}

Stack *initStack()
{
	Stack *new = malloc(sizeof(Stack));
	if (NULL != new) {
		/* Initialize stack with initial size of STACK_SIZE elems */
		new->nodeStack = malloc(sizeof(Node*) * STACK_SIZE);
		if (NULL != new->nodeStack) {
			new->ptr = EMPTY_STACK;
		}
		else {
			/* Delete stack if not enough memory */
			free(new);
			new = NULL;
		}
	}
	return new;
}

void removeStack(Stack *st)
{
	if (NULL != st) {
		free(st->nodeStack);
		free(st);
		st = NULL;
	}
}

bool isEmptyStack(Stack *st)
{
	bool ret = false;
	if (EMPTY_STACK == st->ptr) {
		ret = true;
	}
	return ret;
}

bool isFullStack(Stack *st)
{
	bool ret = false;
	if (FULL_STACK == st->ptr) {
		ret = true;
	}
	return ret;
}

void pushStack(Stack *st, Node *key)
{
	if (isFullStack(st)) {
		/* Is full */
		printf("Error: stack overflow.\n");
		exit(EXIT_FAILURE);
	}
	else {
		/* Is not full */
		st->nodeStack[st->ptr] = key;
		++(st->ptr);
	}
}

Node *popStack(Stack *st)
{
	Node *ret = NULL;
	if (isEmptyStack(st)) {
		/* Stack is empty */
		printf("Error: empty stack.\n");
		exit(EXIT_FAILURE);
	}
	else {
		/* Stack is not empty */
		--(st->ptr);
		ret = st->nodeStack[st->ptr];
	}
	return ret;
}
