# IT-Labs

# ToC
## os2
##### lab1
Write your own `who` command with options `-g` and `-h`.
##### lab2
Rewrite program from lab1 using dynamic shared library for extracting logged users.
##### lab3
Write program that will take as an argument string and using `fork()` function, the program will spawn two children, which each of the child will take half of an argument. Continue spawning until the argument length is equal to 1. The parent process will wait for all of it's children to complete and then write to `stdout` his pid and arguments since beginning, ie:
```
$./program  abcd
26062 abcd ab a
26063 abcd ab b
26060 abcd ab
26065 abcd cd d
26064 abcd cd c
26061 abcd cd
26059 abcd
```
##### lab4
Rewrite your program from lab3. The new program will block `SIGTSTP` signal for the whole execution. After processing it's basic work, every parent process will go into endless loop waiting for `SIGKILL` occurence. If the signal occurs, program should fetch it and propagate to all of its children. After `SIGKILL`, program should check if there was an `SIGTSTP` and write proper message about blocking it.
##### lab5
Write your own implementation of `tree` function. Consider implementing options `-L -d -f -l`.
##### lab6
Write your own program to encode password using SHA-512 and output like in `/etc/shadow`.
Write second program to crack password using dictionary attack. Given input file and extract cypher from `/etc/shadow` and number of pthreads. Next, map file into memory and spawn threads. Threads should go throught the dictionary and try to decode password. Consider also possibility, to calculate optimal pthreads number.
## introduction_to_ai
##### lab1
Write N Queens solution using BFS (Breadth First Search) and DFS (Depth First Search) Searching Algorithms. 
Program should output first solution and number of boards generated.
##### lab2
Write N Queens solution using Simple Hill Climbing Algorithm.
Program should output first solution and number of operations made.
##### lab3
Write N Queens solution using Genetic Algorithm.
Genetic Algorithm should contain functionalities: Selection, Crossing and Mutation.
## programming languages and paradigms
Write programs in C, C++, C++ STL, OOP language other than C++ and non-OOP language different from previous which will be resolving one problem from maths, physics or any other domains.

My problem: calculate integral between `a` and `b` of elementary arithmetic functions using 4 methods: 
- Rectangle's Method 
- Trapeze's Method
- Simpson's Method
- Monte Carlo.
## data structures
Write programs that implement:
- Array of pointers to structures
- Two-sided circular linked list
- BST
- AVL
- List vs BST vs AVL comparison
- AVL vs STL Set comparison
## numeric methods
Implement some common algorithms in numeric methods using MATLAB.
## optimalization methods
Implement some common algorithms related to optimalization (finding local minimum of function) using MATLAB.
## modelling and system simulation
Simulate some common problems using MATLAB+Simulink environment.
## concurrent programming
- Write program, that implemenets matrix multiplication using std::thread.
## basis of data transmission
- Write program, that computes some basic math functions and draw thier plots.
## image processing
Solve common tasks of image processing (image addition, histogram, brightness/contrast measure, etc.) in MATLAB.
## LaTeX
Write CV, polish letter, english letter and front page of diploma in LaTeX environment.

