#ifndef PROJECT_H
#define PROJECT_H

#include <vector>
#include <string>

#include "User.h"
#include "Artifact.h"

enum ProjectStatus {
    PS_New, PS_Preparation, PS_In_Progress,
    PS_Integration, PS_Finished
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
