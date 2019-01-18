#include "Project.h"

int Project::projectsInSystem = 0;

Project *Project::createProject(std::string _n, User *mngr)
{
    Project *proj = nullptr;
    const std::string text = "You do not have permission to create: Project";

    if (mngr != nullptr && mngr->getPermission() != nullptr) {
        proj = new Project(_n, mngr);
    }
    else {
        if (mngr != nullptr)
            mngr->addNotification((Project *)nullptr, text);
    }

    return proj;
}


void Project::attachFunctionality(Artifact *func)
{
    if (func->getType() == AT_Functionality) {
        std::string name = func->getName();

        auto it = std::find_if(functionalities.begin(), functionalities.end(), 
                [&name](Artifact* obj) { return name == obj->getName(); } );

        if (it == functionalities.end()) {
            std::lock_guard<std::mutex> lock(*funcs_data_mutex);
            functionalities.push_back(func);
        }
    }
}

void Project::setManager(User *new_manager)
{
    if (new_manager != nullptr && new_manager->getPermission() != nullptr)
        manager = new_manager;
}

void Project::addEngineer(User *eng)
{
    if (eng != nullptr) {
        std::string username = eng->getUsername();

        auto it = std::find_if(engineers.begin(), engineers.end(), 
                [&username](User *obj) { return username == obj->getUsername(); } );

        if (it == engineers.end()) {
            addSubscriber(eng);
            std::lock_guard<std::mutex> lock(*engs_data_mutex);
            engineers.push_back(eng);
        }
        else {
            std::cout << "Engineer (" << eng->getId() 
                << ") is already in project: " << getName() << std::endl;

        }
    }
}

void Project::notifySubscribers(const std::string &action)
{
    for (auto &sub : subscribers)
        sub->addNotification(this, action);
}

void Project::setStatus(ProjectStatus new_status)
{
    if ((new_status == PS_Preparation && status == PS_New)
            || (new_status == PS_In_Progress && status == PS_Preparation)
            || (new_status == PS_Integration && status == PS_In_Progress)
            || (new_status == PS_Finished && status == PS_Integration && isProjectFinishable()))
    {
        status = new_status;

    }
}

bool Project::isProjectFinishable()
{
    bool result = true; 

    ArtifactStatus finished = AS_Done;

    auto it = std::find_if(functionalities.begin(), functionalities.end(), 
            [&finished](Artifact *obj) { return finished != obj->getStatus(); } );

    if (it != functionalities.end()) {
        result = false;
    }

    return result;
}

void Project::addSubscriber(User *u)
{
    std::lock_guard<std::mutex> lock(*subs_data_mutex);
    subscribers.push_back(u);
}

std::vector<User *> *Project::getEngineers()
{
    return &engineers;
}

std::vector<Observer *> *Project::getSubscribers()
{
    return &subscribers;
}

std::string Project::getName()
{
    return name;
}

User *Project::getManager()
{
    return manager;
}

int Project::getId()
{
    return id;
}

ProjectStatus Project::getStatus()
{
    return status;
}
