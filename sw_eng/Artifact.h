#ifndef ARTIFACT_H
#define ARTIFACT_H

#include <string>
#include <vector>
#include <mutex>

#include "Database.h"
#include "User.h"
#include "Label.h"
#include "Observer.h"
#include "Notification.h"

class User;
class Database;
class Observer;

enum ArtifactType {
    AT_Functionality, AT_Story,
    AT_Task, AT_Defect
};

const std::string ArtifactTypeString[4] = {
    "Functionality", "Story", "Task", "Defect"
};

enum ArtifactStatus {
    AS_New, AS_Analysis, AS_Implementation,
    AS_Validation, AS_Done, AS_Invalid
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
    std::vector<Observer *> *getSubscribers();

    void setParent(Artifact *parent);
    void addChild(Artifact *child);
    void addSubscriber(User *u);
    int setStatus(ArtifactStatus new_status);

    ~Artifact() { };
private:
    Artifact(ArtifactType _t, std::string _n, User *_c, int _id) 
            : name(_n), id(_id), owner(_c), creator(_c), type(_t) { 
        status = AS_New;
        data_mutex = new std::mutex();
        addSubscriber(creator);
        if (owner != creator)
            addSubscriber(owner);
    };

    void notifySubscribers(std::string);

    std::string name;
    int id;
    int prio;
    Artifact *parent = nullptr;
    std::vector<Label *> labels;
    ArtifactStatus status = AS_New;
    User *owner = nullptr;
    User *creator = nullptr;
    std::vector<Artifact *> children;
    ArtifactType type;
    std::vector<Observer *> subscribers;
    std::mutex *data_mutex;
};

#endif
