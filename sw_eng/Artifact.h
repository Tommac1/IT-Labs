#ifndef ARTIFACT_H
#define ARTIFACT_H

#include <string>
#include <vector>

#include "Database.h"
#include "User.h"

class User;
class Database;

enum ArtifactType {
    Functionality, Story,
    Task, Defect
};

const std::string ArtifactTypeString[4] = {
    "Functionality", "Story", "Task", "Defect"
};

enum ArtifactStatus {
    New, Analysis, Implementation,
    Validation, Done, Invalid
};

const std::string ArtifactStatusString[6] = {
    "New", "Analysis", "Implementation",
    "Validation", "Done", "Invalid"
};

class Artifact {
public:
    static Artifact *createArtifact(ArtifactType type, std::string name, 
            User *creator, Database *db);
    static Artifact *createArtifact(ArtifactType type, std::string name, 
            User *creator, Database *db, Artifact *parent);


    int getId();
    std::string getName();
    User *getCreator();
    User *getOwner();
    Artifact *getParent();
    ArtifactStatus getStatus();
    ArtifactType getType();

    void setParent(Artifact *parent);
    void addChild(Artifact *child);

    ~Artifact() { };
private:
    Artifact(ArtifactType _t, std::string _n, User *_c, int _id) 
            : name(_n), id(_id), owner(_c), creator(_c), type(_t) { 
        status = New;
    };

    std::string name;
    int id;
    int prio;
    Artifact *parent = nullptr;
    std::vector<Label *> labels;
    ArtifactStatus status;
    User *owner = nullptr;
    User *creator = nullptr;
    std::vector<Artifact *> children;
    ArtifactType type;
//    std::vector<Observer *> subscribers;
};

#endif
