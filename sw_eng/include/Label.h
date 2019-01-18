#ifndef LABEL_H
#define LABEL_H

#include <string>
#include <vector>

#include "User.h"
#include "Artifact.h"

class User;
class Artifact;

class Label {
public:
    //void add_shared_user(User *u);
    
    Label();
    ~Label();
private:
    std::string name;
    std::vector<User *> shared_users;
    std::vector<Artifact *> related_artifacts;
};

#endif
