---
sidebar_position: 2
---

# Account List

| Username | Default Password | Purpose | User Status | Login Mode | Remarks |
| :----- | :----------------- | :---------------------------------- | :------- | :------------- | :-------------------- |
| admin (openFuyao) | None, randomly generated during installation. | Default platform administrator of the openFuyao community edition core platform. | Enabled | Remote login via browser. | Used to log in to the openFuyao community edition core platform. |
| admin (Harbor) | Harbor@12345 | Default administrator of the local Harbor repository. | Not operated directly by users | Login during API access. | Provides full management permissions on the local Harbor system. |
| harbor_registry_user | regRoot12345 | Account used by the local Harbor repository to interact with the Docker Registry. | Not operated directly by users | Backend login. | Typically used for image uploads and pulls. |
| postgres | dbRoot12345 | Internal database administrator of the local Harbor repository. | Not operated directly by users | Backend login. | Responsible for connecting to and managing the Harbor database service. This account typically has full access to the database. |