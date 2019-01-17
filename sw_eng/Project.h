#ifndef PROJECT_H
#define PROJECT_H

#include <vector>
#include <string>
#include <mutex>

#include "User.h"
#include "Artifact.h"
#include "Observer.h"
#include "Database.h"

class Artifact;
class User;
class Observer;
class Database;

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
    static Project *createProject(std::string _n, User *mngr);
    std::string getName();
    void attachFunctionality(Artifact *func);
    void setManager(User *new_manager);
    User *getManager();
    void addEngineer(User *eng);
    void setStatus(ProjectStatus new_status);
    ProjectStatus getStatus();
    std::vector<Observer *> *getSubscribers();
    std::vector<User *> *getEngineers();

    int getId();

    void addSubscriber(User *u);

    ~Project() { 
        delete funcs_data_mutex;
        delete engs_data_mutex;
        delete subs_data_mutex;
    };
private:
    Project(std::string _n, User *mngr) : name(_n), manager(mngr) {
        id = projectsInSystem++;
        status = PS_New;

        funcs_data_mutex = new std::mutex();
        engs_data_mutex = new std::mutex();
        subs_data_mutex = new std::mutex();

        addSubscriber(mngr);
    };

    int id;
    std::string name;
    User *manager;
    ProjectStatus status;
    std::vector<User *> engineers;
    std::vector<Artifact *> functionalities;
    std::vector<Observer *> subscribers;

    std::mutex *funcs_data_mutex;
    std::mutex *engs_data_mutex;
    std::mutex *subs_data_mutex;

    static int projectsInSystem;

    void notifySubscribers(const std::string &action);
    bool isProjectFinishable();
};

#endif
