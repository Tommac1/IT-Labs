#include "Artifact.h"

int Artifact::artifactsInSystem = 0;


Artifact *Artifact::createArtifact(ArtifactType type, std::string name, User *creator,
        Project *proj, Database *db)
{
    Artifact *art = nullptr;
    if (creator != nullptr) {
        art = new (std::nothrow) Artifact(type, name, proj, creator, artifactsInSystem++); 

        if (art) {
            if (db->addArtifact(art)) {
                delete art;
                art = nullptr;
            }
            else {
                std::string notif = "New " + ArtifactTypeString[type] + " " 
                    + std::to_string(art->getId()) + " created!";
                art->notifySubscribers(notif);
            }
        }
    }

    return art;
}

Artifact *Artifact::createArtifact(ArtifactType type, std::string name, User *creator,
        Project *proj, Database *db, Artifact *parent)
{
    Artifact *art = nullptr;
    bool gotPermission = true;
    const std::string text = "You do not have permission to create: " + ArtifactTypeString[type];

    if (creator != nullptr) {
        art = new (std::nothrow) Artifact(type, name, proj, creator, artifactsInSystem++); 

        if (art) {
            if ((type == AT_Functionality || type == AT_Story) 
                    && creator->getPermission() == nullptr) {
                gotPermission = false;
                creator->addNotification((Artifact *)nullptr, text);
            }
            
            if (gotPermission == false || db->addArtifact(art)) {
                delete art;
                art = nullptr;
            }
            else {
                art->setParent(parent);
                std::string notif = "New " + ArtifactTypeString[type] + " " 
                    + std::to_string(art->getId()) + " created!";
                art->notifySubscribers(notif);
            }
        }
        else {
            art = nullptr;
        }
    }

    return art;
}

void Artifact::addChild(Artifact *child) 
{
    {
        std::lock_guard<std::mutex> lock(*data_mutex);
        children.push_back(child);
    }

    std::string notif = ArtifactTypeString[getType()] + " " 
        + std::to_string(getId()) + " added child "
        + std::to_string(child->getId());

    notifySubscribers(notif);
}

void Artifact::addSubscriber(User *u)
{
    std::string username = u->getUsername();

    auto it = std::find_if(subscribers.begin(), subscribers.end(), 
            [&username](Observer* obj) { return username == obj->getUsername(); } );

    if (it == subscribers.end())
        subscribers.push_back(u);
    else
        std::cout << "User (" << u->getId() << ") is already subscribing artifact ("
            << getId() << ").\n";
}

void Artifact::notifySubscribers(std::string action)
{
    for (auto &sub : subscribers)
        sub->addNotification(this, action);
}

int Artifact::setStatus(ArtifactStatus new_status)
{
    int result = -1;

    if (new_status == AS_Analysis) {
        if ((status == AS_New) || (status == AS_Implementation) 
                || (status == AS_Validation) || (status == AS_Done)) {
            status = new_status;
            result = 0;
        }
    }
    else if (new_status == AS_Implementation) {
        if (status == AS_Analysis) {
            status = new_status;
            result = 0;
        }
    }
    else if (new_status == AS_Validation) {
        if (status == AS_Implementation) {
            status = new_status;
            result = 0;
        }
    }
    else if (new_status == AS_Done) {
        if (allChildrenAreClosed()) {
            if (status == AS_Validation) {
                status = new_status;
                result = 0;
            }
        }
        else {
            std::string notif = ArtifactTypeString[getType()] + " " 
                + std::to_string(getId()) 
                + " cannot close this item. Not all children are resolved.";

            notifySubscribers(notif);
        }
    }
    else if (new_status == AS_Invalid) {
        if ((status == AS_New) || (status == AS_Analysis) || (status == AS_Implementation)) {
            status = new_status;
            result = 0;
        }
    }

    if (!result) {
        std::string notif = ArtifactTypeString[getType()] + " " 
            + std::to_string(getId()) + " got new status: " 
            + ArtifactStatusString[status];

        notifySubscribers(notif);
    }

    return result;
}

int Artifact::allChildrenAreClosed()
{
    int result = 1;

    for (auto &child : children) {
        if (child->getStatus() != AS_Done) {
            result = 0;
            break;
        }
    }

    return result;
}

void Artifact::setOwner(User *new_owner)
{
    if (new_owner != nullptr && new_owner != owner) {
        owner = new_owner;

        std::string notif = ArtifactTypeString[getType()] + " " 
            + std::to_string(getId()) + " got new owner: " 
            + owner->getUsername();

        notifySubscribers(notif);
    }
}

void Artifact::setParent(Artifact *_parent)
{
    parent = _parent;
    _parent->addChild(this);

    std::string notif = ArtifactTypeString[getType()] + " " 
        + std::to_string(getId()) + " got new parent: " 
        + _parent->getName();

    notifySubscribers(notif);
}

int Artifact::getId()
{
    return id;
}

std::string Artifact::getName()
{
    return name;
}

ArtifactType Artifact::getType()
{
    return type;
}

ArtifactStatus Artifact::getStatus()
{
    return status;
}

Artifact *Artifact::getParent()
{
    return parent;
}

User *Artifact::getOwner()
{
    return owner;
}

User *Artifact::getCreator()
{
    return creator;
}

std::vector<Observer *> *Artifact::getSubscribers()
{
    return &subscribers;
}

Project *Artifact::getProject()
{
    return project;
}
