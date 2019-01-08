#ifndef PROJECT_H
#define PROJECT_H

#include <vector>
#include <string>

#include "User.h"
#include "Artifact.h"

enum ProjectStatus {
    New, Preparation, In_Progress,
    Integration, Finished
};

const std::string ProjectStatusString[5] = {
    "New", "Preparation", "In_Progress",
    "Integration", "Finished"
};

class Project {
public:
    Project();
    ~Project();
private:
    std::string name;
    User *manager;
    ProjectStatus status;
    std::vector<User *> engineers;
    std::vector<Artifact *> functionalities;
};

#endif
