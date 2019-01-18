#ifndef OBSERVER_H
#define OBSERVER_H

#include <string>

#include "Artifact.h"
#include "Project.h"

class Artifact;
class Project;

class Observer {
public:
    virtual void addNotification(std::string &action) = 0;
    virtual void addNotification(Artifact *art, const std::string &action) = 0;
    virtual void addNotification(Project *proj, const std::string &action) = 0;
    virtual int getId() = 0;
    virtual std::string getUsername() = 0;
};


#endif
