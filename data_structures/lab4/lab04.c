/* SDIZO N1 20A LAB04 		*/
/* Tomasz Mączkowski  		*/
/* tom.maczkowski@gmail.com */

/* System headers */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Constant and type defines */
typedef enum { false = 0, true = 1 } bool;
#define FILE_NAME 	"inlab04.txt"
#define ARRAY_SIZE 	10
#define STACK_SIZE 	4096
#define FULL_STACK 	4095
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
	int balance;
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
 * Balance the AVL tree
 */
void balanceTree(Node **, Stack *);
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
 *  INORDER_MODE
 *  PREORDER_MODE
 *  POSTORDER_MODE
 */
void printTree(Node *, int);
/*
 * Calculate the height of  AVL subtree
 */
int calculateHeightTree(Node *);
/*
 * Rotate subtree right
 */
void rotateRightTree(Node **, Node *, Node *, Node *);
/*
 * Rotate subtree left
 */
void rotateLeftTree(Node **, Node *, Node *, Node *);
/*
 * Update balance of a node
 */
void updateBalanceTree(Node *);
/*
 * Get balance o a node
 */
int getBalanceTree(Node *);
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
/*
 * Return the numbers of elems in stack
 */
int getSizeStack(Stack *);
/*
 * Useful function for swapping two nodes
 */
void swapNodes(Node *, Node *);
/*
 * Simple function to pcik higher number
 */
int max(int, int);

//--------------------------------------------------------------------------------------------------
// Main runnable
//--------------------------------------------------------------------------------------------------

//// TODO: for debugging, remove after finish
//void printWithHeightsTree(Node *root)
//{
//	if (NULL != root) {
//		printWithHeightsTree(root->left);
//		updateBalanceTree(root);
//		printf("%d : %dh %2db%c ", root->key, calculateHeightTree(root),
//							      root->balance,
//							      abs(root->balance) > 1 ? '@' : ' ');
//		if (NULL != root->left)
//			printf("left : %d ", root->left->key);
//		if (NULL != root->right)
//			printf("right : %d", root->right->key);
//		printf("\n");
//		printWithHeightsTree(root->right);
//	}
//}

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
		fclose(fp);
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

			printTree(root,	PREORDER_MODE);

			searchTree(root, keys[0]);

			removeNodeTree(&root, keys[1]);

			printTree(root, INORDER_MODE);

			removeNodeTree(&root, keys[2]);

			removeNodeTree(&root, keys[3]);

			printTree(root, INORDER_MODE);

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
		/* Newly inserted nodes always have height = 0 */
		new->balance = 0;
	}
	return new;
}

void insertNodeTree(Node **root, const int key)
{
	Stack *path = initStack();
	Node *new = *root;
	Node *parent = NULL;
	/* Indicator which child node of parent to connect */
	bool wentLeft = false;
	bool keyPresent = false;

	if (NULL != *root) {
		/* Tree is not empty */
		while (NULL != new) {
			pushStack(path, new);
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
			pushStack(path, new);
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

			balanceTree(root, path);
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

	removeStack(path);
	path = NULL;
}

Node *initTree()
{
	return NULL;
}

void balanceTree(Node **root, Stack *path)
{
	int balance = 0;
	int childBalance = 0;
	Node *grand = NULL;
	Node *prnt = NULL;
	Node *chld = NULL;
	Node *temp = NULL;

	if (3 <= getSizeStack(path)) {
		chld = popStack(path);
		prnt = popStack(path);

		while (false == isEmptyStack(path)) {
			/* Update the balances of ancestor nodes */
			grand = popStack(path);
			updateBalanceTree(prnt);
			updateBalanceTree(chld);
			balance = prnt->balance;
			childBalance = chld->balance;

			if (0 == balance) {
				/*
				 * There is no longer need for balancing tree
				 * Removing stack
				 */
				path->ptr = 0;
			}
			else if (-1 > balance) {
				/* Left case */

				if (0 > childBalance) {
					/* Left left case */
					rotateLeftTree(root, grand, prnt, chld);
				}
				else {
					/* Left right case */
					if (NULL != chld->left) {
						rotateRightTree(root, prnt, chld, chld->left);
						chld = prnt->right;
						rotateLeftTree(root, grand, prnt, chld);
					}
					else {
						/* Protect vs. chld is child of
						 * removed node with 1 child */
						rotateRightTree(root, grand, prnt, chld);
						temp = chld;
						chld = prnt;
						prnt = temp;
						rotateLeftTree(root, grand, prnt, chld);
					}
				}
			}
			else if (1 < balance) {
				/* Right case */

				if (0 < childBalance) {
					/* Right right case */
					rotateRightTree(root, grand, prnt, chld);
				}
				else {
					/* Right left case */
					if (NULL != chld->right) {
						rotateLeftTree(root, prnt, chld, chld->right);
						chld = prnt->left;
						rotateRightTree(root, grand, prnt, chld);
					}
					else {
						/* Protect vs. chld is child of
						 * removed node with 1 child */
						rotateLeftTree(root, grand, prnt, chld);
						temp = chld;
						chld = prnt;
						prnt = temp;
						rotateRightTree(root, grand, prnt, chld);
					}
				}
			}
			else {
				chld = prnt;
				prnt = grand;
			}
		}
	}

	/* Update root if needed */
	updateBalanceTree(*root);
	balance = (*root)->balance;

	if (-1 > balance) {
		/* Left case */
		childBalance = (*root)->right->balance;

		if (0 > childBalance) {
			/* Left left case */
			rotateLeftTree(root, NULL, *root, (*root)->right);
		}
		else {
			/* Left right case */
			rotateRightTree(root, *root, (*root)->right, (*root)->right->left);
			rotateLeftTree(root, NULL, *root, (*root)->right);
		}

	}
	else if (1 < balance) {
		/* Right case */
		childBalance = (*root)->left->balance;

		if (0 < childBalance) {
			/* Right right case */
			rotateRightTree(root, NULL, *root, (*root)->left);
		}
		else {
			/* Right left case */
			rotateLeftTree(root, *root, (*root)->left, (*root)->left->right);
			rotateRightTree(root, NULL, *root, (*root)->left);
		}
	}
}

void insertXrandomNodesTree(Node **root, int X)
{
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
	Node *toBePushed = NULL;
	if (remove != *root) {
		/* Removed node is not a root */
		parent = popStack(path);
		if (remove->key > parent->key) {
			parent->right = NULL;
			toBePushed = parent->left;
		}
		else {
			parent->left = NULL;
			toBePushed = parent->right;
		}
		free(remove);
		remove = NULL;
	}
	else {
		/* Removed node is a root */
		free(*root);
		*root = NULL;
	}
	pushStack(path, parent);
	pushStack(path, toBePushed);

	balanceTree(root, path);
}

void removeNode1stGradeTree(Node **root, Node *remove, Stack *path)
{
	Node *parent = NULL;
	Node *toBePushed = NULL;

	/* Has left child? If not - it has to have right child. */
	bool hasLeft = ((NULL == remove->left) ? false : true);

	if (remove != *root) {
		/* Removed node is not a root */
		parent = popStack(path);
		if (remove->key > parent->key) {
			parent->right = (hasLeft ? remove->left : remove->right);
			toBePushed = parent->right;
		}
		else {
			parent->left = (hasLeft ? remove->left : remove->right);
			toBePushed = parent->left;
		}
	}
	else {
		/* Removed node is a root */
		*root = (hasLeft ? remove->left : remove->right);
	}

	free(remove);
	remove = NULL;

	pushStack(path, parent);
	pushStack(path, toBePushed);
	balanceTree(root, path);
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
		if (NULL != tempParent) {
			tempParent->left = temp->right;
			temp->right = remove->right;
		}
		temp->left = remove->left;
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
	remove = NULL;

	/* We have to follow the path to parent of taken node
	 * to balance the subtrees
	 */
	if (NULL != tempParent) {
		path = searchTree(*root, tempParent->key);

		if (NULL != tempParent->left) {
			pushStack(path, tempParent->left);
		}
		else if (NULL != tempParent->right) {
			pushStack(path, tempParent->right);
		}
	}
	else {
		path = searchTree(*root, temp->key);
	}

	balanceTree(root, path);
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

int calculateHeightTree(Node *node)
{
	int ret = 0;
	if (NULL != node) {
		ret = 1 + max(calculateHeightTree(node->left),
					  calculateHeightTree(node->right));
	}
	return ret;
}

void rotateLeftTree(Node **root, Node *prnt, Node *crrnt, Node *child)
{
	Node *temp = NULL;

	if (NULL != prnt) {
		/* Update the parent's pointer */
		if (crrnt == prnt->right) {
			prnt->right = child;
		}
		else {
			prnt->left = child;
		}
	}
	else {
		/* Replace the root */
		*root = child;
	}

	/* Replace current with it's child */
	temp = child->left;
	child->left = crrnt;
	crrnt->right = temp;

	/* Update balances */
	updateBalanceTree(crrnt);
	updateBalanceTree(child);
	updateBalanceTree(prnt);
}

void rotateRightTree(Node **root, Node *prnt, Node *crrnt, Node *child)
{
	Node *temp = NULL;

	if (NULL != prnt) {
		/* Update the parent's pointer */
		if (crrnt == prnt->right) {
			prnt->right = child;
		}
		else {
			prnt->left = child;
		}
	}
	else {
		/* Replace the root */
		*root = child;
	}

	/* Replace current with it's child */
	temp = child->right;
	child->right = crrnt;
	crrnt->left = temp;

	/* Update balances */
	updateBalanceTree(crrnt);
	updateBalanceTree(child);
	updateBalanceTree(prnt);
}

void updateBalanceTree(Node *node)
{
	if (NULL != node) {
		node->balance = (calculateHeightTree(node->left) -
						calculateHeightTree(node->right));
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
	/* Free the stack data and stack structure */
	if (NULL != st) {
		free(st->nodeStack);
		free(st);
	}
}

bool isEmptyStack(Stack *st)
{
	bool ret = false;
	if ((NULL == st) || (EMPTY_STACK == st->ptr)) {
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
		if (NULL != key) {
			st->nodeStack[st->ptr] = key;
			++(st->ptr);
		}
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

int getSizeStack(Stack *st)
{
	return st->ptr;
}

void swapNodes(Node *a, Node *b)
{
	Node *temp = a;
	a = b;
	b = temp;
}

int max(int a, int b)
{
	int ret = 0;
	ret = ((a > b) ? a : b);
	return ret;
}
