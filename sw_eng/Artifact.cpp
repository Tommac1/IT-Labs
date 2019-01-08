#include <Artifact.h>

Artifact *Artifact::createArtifact(ArtifactType type, std::string name, User *creator,
        Database *db)
{
    Artifact *art = new Artifact(type, name, creator, creator, db->getArtifactsSize()); 

    return art;
}


void Artifact::setParent(Artifact *_parent)
{
    parent = _parent;
}

int Artifact::getId()
{
    return id;
}
