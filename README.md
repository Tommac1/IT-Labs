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
## introduction_to_ai
##### lab1
Write N Queens solution using BFS (Breadth First Search) and DFS (Depth First Search) Searching Algorithms. 
Program should output first solution and number of boards generated.
##### lab2
Write N Queens solution using Simple Hill Climbing Algorithm.
Program should output first solution and number of operations made.
