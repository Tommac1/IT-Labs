#include "Artifact.h"

Artifact *Artifact::createArtifact(ArtifactType type, std::string name, User *creator,
        Database *db)
{
    Artifact *art = new (std::nothrow) Artifact(type, name, creator, db->getArtifactsSize()); 

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

    return art;
}

Artifact *Artifact::createArtifact(ArtifactType type, std::string name, User *creator,
        Database *db, Artifact *parent)
{
    Artifact *art = new (std::nothrow) Artifact(type, name, creator, db->getArtifactsSize()); 

    if (art) {
        if (db->addArtifact(art)) {
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
        if (status == AS_Validation) {
            status = new_status;
            result = 0;
        }
    }
    else if (new_status == AS_Invalid) {
        if ((status == AS_New) || (status == AS_Analysis) || (status == AS_Implementation)) {
            status = new_status;
            result = 0;
        }
    }
    return result;
}

void Artifact::setParent(Artifact *_parent)
{
    parent = _parent;
    _parent->addChild(this);
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
