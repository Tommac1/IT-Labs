#ifndef OBSERVER_H
#define OBSERVER_H

#include <string>

#include "Artifact.h"

class Artifact;

class Observer {
public:
    virtual void addNotification(Artifact *art, std::string &action) = 0;
    virtual int getId() = 0;
    virtual std::string getUsername() = 0;
};


#endif
