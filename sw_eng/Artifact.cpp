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
        }
    }
    else {
        art = nullptr;
    }

    return art;
}

void Artifact::addChild(Artifact *child) 
{
    children.push_back(child);
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
